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
    #('slc', 'barycenterFM',         'chargeTotal', '', '', ''): 'fm_charge',
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
    ('secshw', 'trackScore',         '', '', '', ''): 'trk_score_2',
    ('secshw', 'shw', 'bestplane_energy','', '', ''): 'shw_energy_2',
    ('secshw', 'shw', 'bestplane_dEdx', '', '', ''): 'shw_dedx_2',
    ('secshw', 'shw', 'angle_z',     '', '', ''): 'shw_angle_z_2',
    ('secshw', 'shw', 'conversion_gap', '', '', ''): 'shw_conv_gap_2',
    ('secshw', 'shw', 'open_angle',  '', '', ''): 'shw_open_angle_2',
    ('secshw', 'shw', 'density',     '', '', ''): 'shw_density_2',
    ('secshw', 'shw', 'len',         '', '', ''): 'shw_len_2',
    ('slc', 'm_alt', '', '', '', ''): 'm_alt',
}


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
              N_shw=1,
              scale_hnl=1.0,
              scale_nu=1.0,
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
    N_shw : int
        1 → use 1-shower feature set, 2 → use 2-shower feature set.
    scale_hnl : float
        Total POT × U² scale factor for HNL (used as sample weights in plots).
    scale_nu : float
        POT scale factor for SM neutrino MC (used as sample weights in plots).

    Returns
    -------
    model : xgb.XGBClassifier
    feat_names : list[str]
        Flat feature names in the same order as the model input.
    """
    feat_dict = FEAT_2SHW if N_shw == 2 else FEAT_1SHW

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

    # ── Uniform weights (no physical weighting) ──────────────────────────────
    w_sig_plot = np.ones(len(sig_df))
    w_bkg_plot = np.ones(len(bkg_df))

    # ── scale_pos_weight ──────────────────────────────────────────────────────
    spw_auto = len(bkg_df) / max(len(sig_df), 1)
    spw = spw_auto if scale_pos_weight is None else scale_pos_weight * spw_auto
    print(f"scale_pos_weight : {spw:.3f}  (balanced = {spw_auto:.3f})")

    # ── Assemble full arrays ──────────────────────────────────────────────────
    X      = np.vstack([X_sig, X_bkg])
    y      = np.concatenate([y_sig, y_bkg])
    w_plot = np.concatenate([w_sig_plot, w_bkg_plot])

    (X_train, X_test,
     y_train, y_test,
     w_plot_train, w_plot_test) = train_test_split(
        X, y, w_plot,
        test_size=test_size, random_state=random_state, stratify=y)
    w_train = np.ones(len(y_train))
    w_test  = np.ones(len(y_test))

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

    return model, feat_names