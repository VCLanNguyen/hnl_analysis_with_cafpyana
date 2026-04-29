"""XGBoost BDT training for HNL vs SM neutrino background."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, roc_curve,
                             average_precision_score)

# ── Feature columns (MultiIndex tuples → flat name) ──────────────────────────
# 1-shower topology
FEAT_1SHW = {
    ('slc', 'nu_score',              '', '', '', ''): 'nu_score',
    # ('slc', 'vertex',                'x', '', '', ''): 'vtx_x',
    # ('slc', 'vertex',                'y', '', '', ''): 'vtx_y',
    # ('slc', 'vertex',                'z', '', '', ''): 'vtx_z',
    ('slc', 'barycenterFM',          'score', '', '', ''): 'fm_score',
    ('slc', 'barycenterFM',         'chargeTotal', '', '', ''): 'fm_charge',
    ('slc', 'barycenterFM',         'flashPEs', '', '', ''): 'fm_flashpes',
    ('primshw', 'trackScore',        '', '', '', ''): 'trk_score_1',
    ('primshw', 'shw', 'bestplane_energy','', '', ''): 'shw_energy_1',
    ('primshw', 'shw', 'bestplane_dEdx', '', '', ''): 'shw_dedx_1',
    ('primshw', 'shw', 'angle_z',    '', '', ''): 'shw_angle_z_1',
    ('primshw', 'shw', 'conversion_gap', '', '', ''): 'shw_conv_gap_1',
    ('primshw', 'shw', 'open_angle', '', '', ''): 'shw_open_angle_1',
    ('primshw', 'shw', 'density',    '', '', ''): 'shw_density_1',
    ('primshw', 'shw', 'len',        '', '', ''): 'shw_len_1',
    ('slc', 'vertex', 'transverse_distance_beam_2', '', '', ''): 'transv_dist_beam_squared',
}

# 2-shower topology (adds secshw features)
FEAT_2SHW = {
    **FEAT_1SHW,
   # ('secshw', 'trackScore',         '', '', '', ''): 'trk_score_2',
   # ('secshw', 'shw', 'bestplane_energy','', '', ''): 'shw_energy_2',
   # ('secshw', 'shw', 'bestplane_dEdx', '', '', ''): 'shw_dedx_2',
   # ('secshw', 'shw', 'angle_z',     '', '', ''): 'shw_angle_z_2',
    ('secshw', 'shw', 'conversion_gap', '', '', ''): 'shw_conv_gap_2',
    ('secshw', 'shw', 'open_angle',  '', '', ''): 'shw_open_angle_2',
    ('secshw', 'shw', 'density',     '', '', ''): 'shw_density_2',
    ('secshw', 'shw', 'len',         '', '', ''): 'shw_len_2',
    ('slc', 'm_alt', '', '', '', ''): 'm_alt',
}

def score_bdt(df, model, feat_dict):
    """Apply a trained BDT to a new DataFrame and return the scores.

    Parameters
    ----------
    df : pd.DataFrame
        Slice-level DataFrame (same format as used in training).
    model : xgb.XGBClassifier
        Trained model (from train_bdt or load_bdt).
    feat_dict : dict
        Column tuple → feature name mapping (from load_bdt or train_bdt).

    Returns
    -------
    scores : np.ndarray
        BDT score for each row in df.
    """
    flat = df.reset_index()
    avail = [col for col in feat_dict if col in flat.columns]
    missing = [feat_dict[col] for col in feat_dict if col not in avail]
    if missing:
        print(f"Warning: missing features (set to NaN): {missing}")

    X = flat[avail].values.astype(float)
    return model.predict_proba(X)[:, 1]

def train_bdt(hnl_df,
              sm_df,
              hnl_label="HNL 100 MeV",
              sm_label=r"SM $\nu$",
              test_size=0.3,
              random_state=42,
              xgb_params=None,
              normalize_plots=True,
              hyper_search=False,
              hyper_n_iter=30,
              early_stopping_rounds=20,
              scale_pos_weight=None,
              scale_hnl=1.0,
              scale_nu=1.0,
              feat_dict=None,
              model=None,
              save_model=False,
              save_dir="BDT_training",
              save_tag="bdt",
              ):
    """Train XGBoost BDT to separate HNL signal from SM neutrino background.

    Parameters
    ----------
    hnl_df : pd.DataFrame
        HNL signal DataFrame (signal column == 9). Must have MultiIndex columns.
        The ``weights_mc`` column is used as physical event weights.
    sm_df : pd.DataFrame
        SM neutrino background DataFrame. Must have MultiIndex columns.
        The ``weights_mc`` column is used as physical event weights.
    hnl_label, sm_label : str
        Legend labels for plots.
    test_size : float
        Fraction of data used for testing (default 0.3).
    random_state : int
        Random seed for reproducibility.
    xgb_params : dict, optional
        Override any XGBoost hyperparameters.
    normalize_plots : bool
        Normalize BDT score histograms to unit area.
    hyper_search : bool
        Run xgb.cv random hyperparameter search before final training.
    hyper_n_iter : int
        Number of random parameter combinations to try.
    early_stopping_rounds : int
        Early stopping patience for XGBoost.
    scale_pos_weight : float or None
        None  → auto = N_bkg/N_sig (balanced).
        < 1   → penalise background false-positives more aggressively.
        Pass the multiplier relative to balanced; e.g. 0.1 → 10x more weight on bkg.
    scale_hnl : float
        Total POT × U² scale factor for HNL (used as sample weights in plots).
    scale_nu : float
        POT scale factor for SM neutrino MC (used as sample weights in plots).

    Returns
    -------
    model : xgb.XGBClassifier
    feat_names : list[str]
        Flat feature names in the same order as the model input.
    model : xgb.XGBClassifier, optional
        If provided, skip training and use this model directly.
    save_model : bool, default False
        If True, save model and feat_names to save_dir.
    save_dir : str, default "BDT_training"
        Directory (relative to cwd) where the model is saved.
    save_tag : str, default "bdt"
        Filename prefix for saved files.
    """
    import os, json, pickle

    if feat_dict is None:
        feat_dict = FEAT_1SHW

    # ── Flatten MultiIndex to plain columns ───────────────────────────────────
    sig_df = hnl_df.reset_index()
    bkg_df = sm_df.reset_index()

    print(f"Signal slices    : {len(sig_df)}")
    print(f"Background slices: {len(bkg_df)}")

    # ── Keep only feature columns present in both DataFrames ─────────────────
    avail = [c for c in feat_dict
             if c in sig_df.columns and c in bkg_df.columns]
    missing = [feat_dict[c] for c in feat_dict if c not in avail]
    if missing:
        print(f"Warning: missing features (skipped): {missing}")
    feat_names = [feat_dict[c] for c in avail]

    X_sig = sig_df[avail].values.astype(float)
    X_bkg = bkg_df[avail].values.astype(float)
    y_sig = np.ones(len(sig_df))
    y_bkg = np.zeros(len(bkg_df))

    # ── If model provided, skip training entirely ─────────────────────────────
    if model is not None:
        print("Using provided model — skipping training.")
        score_hnl = model.predict_proba(X_sig)[:, 1]
        score_sm  = model.predict_proba(X_bkg)[:, 1]
        return model, feat_names, score_hnl, score_sm

    # ── Uniform weights (no physical weighting) ──────────────────────────────
    w_sig_plot = np.ones(len(sig_df))
    w_bkg_plot = np.ones(len(bkg_df))

    # ── scale_pos_weight ──────────────────────────────────────────────────────
    spw_auto = len(bkg_df) / max(len(sig_df), 1)
    spw = spw_auto if scale_pos_weight is None else scale_pos_weight * spw_auto
    print(f"scale_pos_weight : {spw:.3f}  (balanced = {spw_auto:.3f})")

    # ── Assemble full arrays ──────────────────────────────────────────────────
    n_sig  = len(sig_df)
    idx    = np.arange(len(X_sig) + len(X_bkg))
    X      = np.vstack([X_sig, X_bkg])
    y      = np.concatenate([y_sig, y_bkg])
    w_plot = np.concatenate([w_sig_plot, w_bkg_plot])

    (X_train, X_test,
     y_train, y_test,
     w_plot_train, w_plot_test,
     idx_train, idx_test) = train_test_split(
        X, y, w_plot, idx,
        test_size=test_size, random_state=random_state, stratify=y)
    w_train = np.ones(len(y_train))
    w_test  = np.ones(len(y_test))

    # Boolean masks over the original DataFrames indicating test-set rows
    test_mask_hnl = np.zeros(len(sig_df), dtype=bool)
    test_mask_sm  = np.zeros(len(bkg_df), dtype=bool)
    sig_test_idx  = idx_test[idx_test < n_sig]
    bkg_test_idx  = idx_test[idx_test >= n_sig] - n_sig
    test_mask_hnl[sig_test_idx] = True
    test_mask_sm[bkg_test_idx]  = True

    # ── Hyperparameter search ─────────────────────────────────────────────────
    if hyper_search:
        print(f"\nRunning xgb.cv random search ({hyper_n_iter} iters, "
              f"metric=aucpr, early_stop={early_stopping_rounds})...")
        param_grid = {
            'max_depth':        [4, 6, 8, 10],
            'learning_rate':    [0.02, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [5, 10, 20],
            'gamma':            [0.5, 1.0, 2.0],
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        rng = np.random.RandomState(random_state)
        best_score, best_params, best_n_estimators = -np.inf, None, 300
        for i in range(hyper_n_iter):
            p = {k: rng.choice(v) for k, v in param_grid.items()}
            xgb_p = {**p,
                     'objective':        'binary:logistic',
                     'eval_metric':      'aucpr',
                     'scale_pos_weight': spw,
                     'seed':             random_state,
                     'nthread':          -1}
            cv = xgb.cv(xgb_p, dtrain, num_boost_round=800, nfold=3,
                        stratified=True,
                        early_stopping_rounds=early_stopping_rounds,
                        seed=random_state, verbose_eval=False)
            score = cv['test-aucpr-mean'].iloc[-1]
            n_trees = len(cv)
            print(f"  [{i+1:3d}/{hyper_n_iter}] AUCPR={score:.4f}  trees={n_trees}  {p}")
            if score > best_score:
                best_score, best_params, best_n_estimators = score, p, n_trees
        print(f"\nBest AUCPR (CV): {best_score:.4f}")
        print(f"Best params    : {best_params}")
        params = dict(eval_metric='logloss', random_state=random_state,
                      scale_pos_weight=spw, n_estimators=best_n_estimators)
        params.update(best_params)
    else:
        params = dict(
            n_estimators     = 500,
            max_depth        = 10,
            learning_rate    = 0.03,
            subsample        = 0.6,
            colsample_bytree = 0.8,
            min_child_weight = 50,
            gamma            = 2.0,
            scale_pos_weight = spw,
            eval_metric      = 'logloss',
            random_state     = random_state,
        )

    if xgb_params:
        params.update(xgb_params)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(**params,
                               early_stopping_rounds=early_stopping_rounds)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)

    best_round = model.best_iteration + 1
    evals      = model.evals_result()
    metric_key = list(evals['validation_0'].keys())[0]
    loss_train = evals['validation_0'][metric_key][:best_round]
    loss_test  = evals['validation_1'][metric_key][:best_round]

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_prob       = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc   = roc_auc_score(y_test, y_prob)
    aucpr = average_precision_score(y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'─'*52}")
    print(f"  {'AUC-ROC':<18} {auc:.4f}")
    print(f"  {'AUC-PR':<18} {aucpr:.4f}")
    print(f"  {'Accuracy':<18} {accuracy_score(y_test, y_pred):.4f}")
    print(f"  {'Precision':<18} {precision_score(y_test, y_pred):.4f}")
    print(f"  {'Recall':<18} {recall_score(y_test, y_pred):.4f}")
    print(f"  {'F1':<18} {f1_score(y_test, y_pred):.4f}")
    print(f"{'─'*52}")
    print(f"  Confusion matrix  (rows=true, cols=pred)")
    print(f"              Pred Bkg   Pred Sig")
    print(f"  True Bkg  {tn:>8}   {fp:>8}")
    print(f"  True Sig  {fn:>8}   {tp:>8}")
    print(f"{'─'*52}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Background', 'Signal'])}")

    # ── Plot helpers ──────────────────────────────────────────────────────────
    def _sbnd_label(ax):
        ax.text(0.10, 1.01, "SBND HNL Analysis, Preliminary",
                transform=ax.transAxes, fontsize=11, fontweight='bold')
    def _style(ax):
        for spine in ax.spines.values(): spine.set_linewidth(2)
        ax.tick_params(width=2, length=8, labelsize=11)

    # ── Plot 0: Loss curve ────────────────────────────────────────────────────
    iters = np.arange(1, best_round + 1)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(iters, loss_train, color='royalblue', lw=1.5, label='Train')
    ax.plot(iters, loss_test,  color='tomato',    lw=1.5, label='Test')
    ax.axvline(best_round, color='gray', lw=1.2, ls='--',
               label=f'Best round {best_round}')
    ax.set_xlabel('Boosting round', fontsize=13)
    ax.set_ylabel(metric_key, fontsize=13)
    _style(ax); _sbnd_label(ax)
    ax.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    #plt.savefig('bdt_loss.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    # ── Plot 1: BDT score ─────────────────────────────────────────────────────
    sig_test  = y_prob[y_test == 1];  wt_sig_test  = w_plot_test[y_test == 1]
    bkg_test  = y_prob[y_test == 0];  wt_bkg_test  = w_plot_test[y_test == 0]
    sig_train = y_prob_train[y_train == 1]; wt_sig_train = w_plot_train[y_train == 1]
    bkg_train = y_prob_train[y_train == 0]; wt_bkg_train = w_plot_train[y_train == 0]

    bins_score = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.hist(sig_test,  bins=bins_score, weights=wt_sig_test,  density=normalize_plots,
            histtype='step',      color='royalblue', lw=1.5,
            label=f'{hnl_label} test ({wt_sig_test.sum():.0f})')
    ax.hist(bkg_test,  bins=bins_score, weights=wt_bkg_test,  density=normalize_plots,
            histtype='step',      color='tomato',    lw=1.5,
            label=f'{sm_label} test ({wt_bkg_test.sum():.0f})')
    # ax.hist(sig_train, bins=bins_score, weights=wt_sig_train, density=normalize_plots,
    #         histtype='stepfilled', color='royalblue', alpha=0.25, lw=0,
    #         label=f'{hnl_label} train')
    # ax.hist(bkg_train, bins=bins_score, weights=wt_bkg_train, density=normalize_plots,
    #         histtype='stepfilled', color='tomato',    alpha=0.25, lw=0,
    #         label=f'{sm_label} train')
    ax.set_xlabel('BDT score', fontsize=13)
    ax.set_ylabel('Normalised' if normalize_plots else 'Slices / bin', fontsize=13)
    ax.set_xlim(0, 1); ax.set_yscale('log')
    _style(ax); _sbnd_label(ax)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    plt.tight_layout()
    #plt.savefig('bdt_score.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    # ── Plot 2: ROC curve ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr, tpr, color='royalblue', lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False positive rate', fontsize=13)
    ax.set_ylabel('True positive rate', fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _style(ax); _sbnd_label(ax)
    ax.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    #plt.savefig('bdt_roc.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    # ── Plot 3: Purity & efficiency vs BDT cut ────────────────────────────────
    thresholds  = np.linspace(0, 1, 200)
    sig_mask_t  = y_test == 1
    bkg_mask_t  = y_test == 0
    n_sig_total = wt_sig_test.sum()
    n_bkg_total = wt_bkg_test.sum()
    purity, eff_sig, eff_bkg = [], [], []
    for thr in thresholds:
        sel   = y_prob >= thr
        n_s   = wt_sig_test[(sig_mask_t & sel)[sig_mask_t]].sum()
        n_b   = wt_bkg_test[(bkg_mask_t & sel)[bkg_mask_t]].sum()
        purity.append(n_s / (n_s + n_b) if (n_s + n_b) > 0 else 1.0)
        eff_sig.append(n_s / n_sig_total if n_sig_total > 0 else 0.0)
        eff_bkg.append(n_b / n_bkg_total if n_bkg_total > 0 else 0.0)
    purity, eff_sig, eff_bkg = map(np.array, [purity, eff_sig, eff_bkg])
    zero_bkg = np.where(eff_bkg == 0)[0]
    thr_zero  = thresholds[zero_bkg[0]] if len(zero_bkg) else None

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(thresholds, purity,  color='royalblue', lw=2, label='Purity (sig / total)')
    ax.plot(thresholds, eff_sig, color='green',     lw=2, ls='--', label='Signal efficiency')
    ax.plot(thresholds, eff_bkg, color='tomato',    lw=2, ls=':',  label='Background efficiency')
    if thr_zero is not None:
        ax.axvline(thr_zero, color='gray', lw=1.5, ls='--',
                   label=f'0 bkg at cut = {thr_zero:.2f}')
        ax.annotate(f'sig. eff. = {eff_sig[zero_bkg[0]]:.2f}',
                    xy=(thr_zero, eff_sig[zero_bkg[0]]),
                    xytext=(thr_zero - 0.25, eff_sig[zero_bkg[0]] - 0.12),
                    fontsize=10, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
    ax.set_xlabel('BDT score cut', fontsize=13)
    ax.set_ylabel('Fraction', fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _style(ax); _sbnd_label(ax)
    ax.legend(fontsize=10, frameon=False)
    plt.tight_layout()
    #plt.savefig('bdt_purity_vs_cut.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    # ── Plot 4: Feature importance ────────────────────────────────────────────
    importance = model.feature_importances_
    order      = np.argsort(importance)
    names_ord  = [feat_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(feat_names) + 1.5), dpi=150)
    ax.barh(names_ord, importance[order], color='steelblue', edgecolor='k', lw=0.7)
    ax.set_xlabel('Feature importance (gain)', fontsize=12)
    _style(ax); _sbnd_label(ax)
    plt.tight_layout()
    #plt.savefig('bdt_feature_importance.pdf', bbox_inches='tight')
    plt.show(); plt.close()

    # ── Plot 5: Correlation matrices ──────────────────────────────────────────

    # Remove NaN rows per-class for correlation computation
    df_sig_feat = pd.DataFrame(X_sig, columns=feat_names).dropna()
    df_bkg_feat = pd.DataFrame(X_bkg, columns=feat_names).dropna()

    corr_sig = df_sig_feat.corr()
    corr_bkg = df_bkg_feat.corr()

    n_feat = len(feat_names)
    fig_size = max(6, 0.55 * n_feat)
    cmap = mpl.cm.RdBu_r
    vmin, vmax = -1, 1

    for corr_mat, label_str, fname in [
        (corr_sig, hnl_label, 'bdt_corr_signal.pdf'),
        (corr_bkg, sm_label,  'bdt_corr_background.pdf'),
    ]:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
        im = ax.imshow(corr_mat.values, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='none')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_feat))
        ax.set_yticks(range(n_feat))
        ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(feat_names, fontsize=9)

        # Annotate cells with correlation value
        for i in range(n_feat):
            for j in range(n_feat):
                val = corr_mat.values[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

        ax.set_title(f'Feature correlation — {label_str}', fontsize=12, pad=8)
        #_sbnd_label(ax)
        _style(ax)
        plt.tight_layout()
        #plt.savefig(fname, bbox_inches='tight')
        plt.show(); plt.close()

    score_hnl = model.predict_proba(X_sig)[:, 1]
    score_sm  = model.predict_proba(X_bkg)[:, 1]

    # ── Save model ────────────────────────────────────────────────────────────
    if save_model:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{save_tag}_model.pkl")
        fdict_path = os.path.join(save_dir, f"{save_tag}_feat_dict.pkl")
        masks_path = os.path.join(save_dir, f"{save_tag}_test_masks.pkl")
        avail_feat_dict = {col: feat_dict[col] for col in avail}
        with open(model_path, 'wb') as fh:
            pickle.dump(model, fh)
        with open(fdict_path, 'wb') as fh:
            pickle.dump(avail_feat_dict, fh)
        with open(masks_path, 'wb') as fh:
            pickle.dump({'hnl': test_mask_hnl, 'sm': test_mask_sm}, fh)
        print(f"Model      → {model_path}")
        print(f"feat_dict  → {fdict_path}")
        print(f"test masks → {masks_path}")

    return model, feat_names, score_hnl, score_sm, test_mask_hnl, test_mask_sm


def load_bdt(save_tag, save_dir="BDT_training"):
    """Load a saved BDT model, feat_dict and test masks.

    Returns
    -------
    model : xgb.XGBClassifier
    feat_dict : dict
        Column tuple → feature name mapping used during training.
    feat_names : list[str]
        Flat feature names in training order.
    test_mask_hnl : np.ndarray[bool]
        Boolean mask over the HNL DataFrame rows that were in the test set.
    test_mask_sm : np.ndarray[bool]
        Boolean mask over the SM DataFrame rows that were in the test set.
    """
    import os, pickle
    model_path = os.path.join(save_dir, f"{save_tag}_model.pkl")
    fdict_path = os.path.join(save_dir, f"{save_tag}_feat_dict.pkl")
    masks_path = os.path.join(save_dir, f"{save_tag}_test_masks.pkl")
    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)
    with open(fdict_path, 'rb') as fh:
        feat_dict = pickle.load(fh)
    feat_names = list(feat_dict.values())
    test_mask_hnl, test_mask_sm = None, None
    if os.path.exists(masks_path):
        with open(masks_path, 'rb') as fh:
            masks = pickle.load(fh)
        test_mask_hnl = masks['hnl']
        test_mask_sm  = masks['sm']
    print(f"Loaded model from {model_path}  |  features: {len(feat_names)}")
    return model, feat_dict, feat_names, test_mask_hnl, test_mask_sm

def eval_bdt(model,
             feat_names,
             hnl_df,
             sm_df,
             purity_targets=None,
             plot_var=None,
             plot_bins=None,
             bdt_cut_plot=0.5,
             scale_hnl=1.0,
             scale_nu=1.0,
             target_pot=1e21,
             U2=None,
             U2_ref=1e-7,
             hnl_label="HNL",
             sm_label="SM ν",
             feat_dict=None,
             **plot_kwargs):
    """Evaluate a trained BDT: print efficiency/purity table and optionally plot a variable.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained BDT model returned by train_bdt.
    feat_names : list[str]
        Flat feature names in the same order used during training.
    hnl_df, sm_df : pd.DataFrame
        HNL and SM dataframes (same format used for training).
    purity_targets : list[float], optional
        Purity levels (%) at which to report signal efficiency.
        Default [30, 50, 60, 70, 80, 90, 95, 100].
    plot_var : tuple or str, optional
        MultiIndex column to plot after applying bdt_cut_plot. If None, no plot.
    plot_bins : array-like, optional
        Bin edges for plot_var. Required when plot_var is set.
    bdt_cut_plot : float, default 0.5
        BDT score threshold used for the variable plot.
    scale_hnl : float, default 1.0
        Scale factor for HNL (e.g. target_pot/mchnl_pot * scaleU²).
        Applied on top of weights_mc if present.
    scale_nu : float, default 1.0
        Scale factor for SM MC (e.g. target_pot/mcbnb_pot).
        Applied on top of weights_mc if present.
    U2 : float, optional
        If provided, additionally scales HNL by (U2/U2_ref)².
    U2_ref : float, default 1e-7
        Reference |U|² used in simulation.
    hnl_label, sm_label : str
        Labels for printout and plots.
    **plot_kwargs
        Forwarded to plot_mc_hnl.
    """
    from .plotting import plot_mc_hnl

    if purity_targets is None:
        purity_targets = [0, 10, 30, 50, 60, 70, 80, 90, 95, 100]

    u2_scale  = (U2 / U2_ref) ** 2 if U2 is not None else 1.0
    scale_hnl = scale_hnl * u2_scale

    _wcol = ('weights_mc', '', '', '', '', '')

    def _wsum(df, mask=None):
        sub = df if mask is None else df[mask]
        if _wcol in sub.columns:
            return float(sub[_wcol].fillna(0).sum())
        return float(len(sub))

    # Build feature matrices (MultiIndex → flat names)
    _lookup_dicts = [feat_dict] if feat_dict is not None else [FEAT_1SHW, FEAT_2SHW]
    name_to_col   = {v: k for d in _lookup_dicts for k, v in d.items()}
    ordered_cols  = [name_to_col[n] for n in feat_names if n in name_to_col]

    hnl_flat = hnl_df.reset_index()
    sm_flat  = sm_df.reset_index()

    X_hnl = hnl_flat[ordered_cols].values.astype(float)
    X_sm  = sm_flat[ordered_cols].values.astype(float)

    score_hnl = model.predict_proba(X_hnl)[:, 1]
    score_sm  = model.predict_proba(X_sm)[:, 1]

    W_hnl_0 = _wsum(hnl_df) * scale_hnl
    W_sm_0  = _wsum(sm_df)  * scale_nu

    title_str = f"BDT evaluation — {hnl_label}  |  target POT = {target_pot:.0e}"
    if U2 is not None:
        title_str += f"  |  |U2| = {U2:.2e}"

    # ── Dense sweep to build purity vs threshold curve ────────────────────────
    thresholds = np.linspace(0, 1, 2000)
    purity_curve = []
    for thr in thresholds:
        wh = _wsum(hnl_df, score_hnl >= thr) * scale_hnl
        ws = _wsum(sm_df,  score_sm  >= thr) * scale_nu
        purity_curve.append(100 * wh / (wh + ws) if (wh + ws) > 0 else 0.)
    purity_curve = np.array(purity_curve)

    # ── For each purity target find the lowest threshold that achieves it ─────
    cnt_fmt = "{:.3e}".format
    sep     = "─" * 82
    header  = (f"  {'target pur%':>11}  {'BDT cut':>8}  {'HNL':>14}  "
               f"{'HNL eff%':>9}  {'SM':>14}  {'SM eff%':>8}  {'purity%':>8}  {'Punzi':>10}")
    print(f"\n{title_str}")
    print(sep)
    print(header)
    print(sep)

    for pur_target in purity_targets:
        # find indices where purity >= target, take the one with lowest threshold
        idx = np.where(purity_curve >= pur_target)[0]
        if len(idx) == 0:
            print(f"  {pur_target:>10.0f}%  {'no cut achieves this purity':>60}")
            continue
        best_idx = idx[0]          # lowest threshold (most signal kept)
        cut      = thresholds[best_idx]
        mh = score_hnl >= cut
        ms = score_sm  >= cut
        wh = _wsum(hnl_df, mh) * scale_hnl
        ws = _wsum(sm_df,  ms) * scale_nu
        eff_h  = 100 * wh / W_hnl_0 if W_hnl_0 > 0 else 0.
        eff_s  = 100 * ws / W_sm_0  if W_sm_0  > 0 else 0.
        purity = 100 * wh / (wh + ws) if (wh + ws) > 0 else 0.
        punzi  = wh / (0.5 + np.sqrt(ws)) if ws > 0 else 0.
        print(f"  {pur_target:>10.0f}%  {cut:>8.3f}  {cnt_fmt(wh):>14}  "
              f"{eff_h:>8.1f}%  {cnt_fmt(ws):>14}  {eff_s:>7.1f}%  "
              f"{purity:>7.2f}%  {punzi:>10.4f}")
    print(sep)

    # ── BDT score distribution (POT-scaled) ───────────────────────────────────
    _score_col = ('slc', 'bdt_score', '', '', '', '')
    _wcol      = ('weights_mc', '', '', '', '', '')

    def _plot_weights(df, scale):
        if _wcol in df.columns:
            return df[_wcol].fillna(0).values * scale
        return np.ones(len(df)) * scale

    if _score_col in hnl_df.columns and _score_col in sm_df.columns:
        bins_score = np.linspace(0, 1, 51)
        fig_score, ax_score = plt.subplots(figsize=(7, 5), dpi=150)
        ax_score.hist(sm_df[_score_col].values,  bins=bins_score,
                      weights=_plot_weights(sm_df,  scale_nu),
                      histtype='stepfilled', color='gray',      alpha=0.5,
                      label=sm_label)
        ax_score.hist(hnl_df[_score_col].values, bins=bins_score,
                      weights=_plot_weights(hnl_df, scale_hnl),
                      histtype='step',       color='royalblue', lw=2,
                      label=hnl_label)
        ax_score.set_xlabel('BDT score', fontsize=13)
        ax_score.set_ylabel('Events / bin', fontsize=13)
        ax_score.set_yscale('log')
        ax_score.set_xlim(0, 1)
        ax_score.legend(fontsize=10, frameon=False, ncol=1)
        ax_score.set_title(f'BDT score — {title_str}', fontsize=10)
        plt.tight_layout()
        plt.show()

    # Variable plot after BDT cut
    if plot_var is not None:
        if plot_bins is None:
            raise ValueError("plot_bins is required when plot_var is set")

        mh_plot = score_hnl >= bdt_cut_plot
        ms_plot = score_sm  >= bdt_cut_plot

        hnl_sel = hnl_df[np.array(mh_plot)]
        sm_sel  = sm_df[np.array(ms_plot)]

        # Temporarily update the global HNL legend label used by plot_var
        from . import plotting as _plt_mod
        _old_label = _plt_mod.signal_labels[0]
        _plt_mod.signal_labels[0] = hnl_label
        # Merge legend_kwargs: force ncol=1 unless caller overrides it
        _lkw = {'ncol': 1, **plot_kwargs.pop('legend_kwargs', {})}
        try:
            fig, ax = plot_mc_hnl(
                mc_df         = sm_sel,
                hnl_df        = hnl_sel,
                var           = plot_var,
                bins          = plot_bins,
                scale_nu      = scale_nu,
                scale_hnl     = scale_hnl,
                legend_kwargs = _lkw,
                **plot_kwargs,
            )
        finally:
            _plt_mod.signal_labels[0] = _old_label

        ax.set_title(f'{hnl_label} — BDT score ≥ {bdt_cut_plot}')
        plt.show()
        return fig, ax
