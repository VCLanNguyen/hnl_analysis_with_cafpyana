import numpy as np
import pandas as pd
from dataclasses import replace
import warnings
import pickle
from tqdm import tqdm
from .utils import ensure_lexsorted
from .histogram import get_hist1d, get_hist2d
from .syst import *
from .classes import SystematicsOutput, XSecInputs
from .constants import integrated_flux, signal_dict

def get_corr_from_cov(cov):
    sigma = np.sqrt(np.diag(cov))
    denom = np.outer(sigma, sigma)

    corr = np.divide(
        cov,
        denom,
        out=np.zeros_like(cov, dtype=float),
        where=denom > 0
    )

    np.fill_diagonal(corr, 1.0)
    return corr

def get_fractional_covariance(cov, cv_hist):
    cv_hist = np.asarray(cv_hist)
    denom = np.outer(cv_hist, cv_hist)

    frac_cov = np.divide(
        cov,
        denom,
        out=np.zeros_like(cov, dtype=float),
        where=denom > 0
    )
    return frac_cov


def add_uncertainty(
    result: SystematicsOutput,
    cov: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
    unc: np.ndarray | None = None,
    sum_value: float | None = None,
    top5: bool = False,
):
    """
    Add a user-defined covariance contribution to a SystematicsOutput.

    Parameters
    ----------
    result
        Existing systematics result object.
    cov
        Covariance matrix contribution to add. Must match (nbins, nbins).
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    target
        Where to apply this uncertainty: "rate", "xsec", or "both".
    unc
        Optional per-bin fractional uncertainty array to store in the df.
        Defaults to sqrt(diag(cov))/hist_cv.
    sum_value
        Optional summary scalar for the df. Defaults to mean(unc).
    top5
        Value for the `top5` column in the added row.
    """
    if not key:
        raise ValueError("key must be a non-empty string")
    if target not in {"rate", "xsec", "both"}:
        raise ValueError("target must be one of: 'rate', 'xsec', 'both'")
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (cv_hist.size, cv_hist.size):
        raise ValueError(
            f"cov shape {cov.shape} does not match expected {(cv_hist.size, cv_hist.size)}"
        )

    if target in {"xsec", "both"} and not result.has_xsec:
        raise ValueError("xsec covariance is not available in this SystematicsOutput")

    if unc is None:
        unc = np.divide(np.sqrt(np.diag(cov)), cv_hist, out=np.zeros_like(cv_hist, dtype=float), where=cv_hist > 0)
    else:
        unc = np.asarray(unc, dtype=float)
        if unc.shape != cv_hist.shape:
            raise ValueError(f"unc shape {unc.shape} does not match hist_cv shape {cv_hist.shape}")

    if sum_value is None:
        sum_value = float(np.mean(unc))

    syst_row = pd.DataFrame(
        {
            "key": [key],
            "category": [category],
            "unc": [unc],
            "sum": [sum_value],
            "top5": [top5],
        }
    )

    syst_entry = {
        "cov": cov,
        "cov_frac": get_fractional_covariance(cov, cv_hist),
        "corr": get_corr_from_cov(cov),
    }

    updates = {}

    if target in {"rate", "both"}:
        updates["rate_cov"] = result.rate_cov + cov
        updates["rate_syst_df"] = pd.concat([result.rate_syst_df, syst_row], ignore_index=True)
        updates["rate_syst_dict"] = {**result.rate_syst_dict, key: syst_entry}

    if target in {"xsec", "both"}:
        updates["xsec_cov"] = result.xsec_cov + cov
        updates["xsec_syst_df"] = pd.concat([result.xsec_syst_df, syst_row], ignore_index=True)
        updates["xsec_syst_dict"] = {**result.xsec_syst_dict, key: syst_entry}

    return replace(result, **updates)


def add_flat_norm_uncertainty(
    result: SystematicsOutput,
    frac_unc: float,
    key: str,
    category: str | None = None,
):
    """
    Add a fully correlated flat normalization uncertainty to a SystematicsOutput.

    Parameters
    ----------
    result
        Existing systematics result object.
    frac_unc
        Fractional uncertainty (e.g. 0.02 for 2%).
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    """
    if frac_unc < 0:
        raise ValueError("frac_unc must be non-negative")
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    flat_cov = (frac_unc ** 2) * np.outer(cv_hist, cv_hist)
    unc = np.full(cv_hist.shape, frac_unc, dtype=float)
    return add_uncertainty(
        result=result,
        cov=flat_cov,
        key=key,
        category=category,
        target="both" if result.has_xsec else "rate",
        unc=unc,
        sum_value=float(frac_unc),
    )


def add_fractional_uncertainty(
    result: SystematicsOutput,
    frac_unc: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
    correlation: str = "diagonal",
):
    """
    Add a per-bin fractional uncertainty array with configurable correlation.

    Parameters
    ----------
    result
        Existing systematics result object.
    frac_unc
        Per-bin fractional uncertainties (e.g. [0.05, 0.2, 0.2, 0.2]).
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    target
        Where to apply this uncertainty: "rate", "xsec", or "both".
    correlation
        Correlation model for bin-to-bin structure:
        - "diagonal": uncorrelated between bins.
        - "fully_correlated": 100% correlated across bins.
    """
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    frac_unc = np.asarray(frac_unc, dtype=float)
    if frac_unc.shape != cv_hist.shape:
        raise ValueError(
            f"frac_unc shape {frac_unc.shape} does not match hist_cv shape {cv_hist.shape}"
        )
    if np.any(frac_unc < 0):
        raise ValueError("frac_unc entries must be non-negative")
    if correlation not in {"diagonal", "fully_correlated"}:
        raise ValueError("correlation must be one of: 'diagonal', 'fully_correlated'")

    sigma = frac_unc * cv_hist
    if correlation == "diagonal":
        cov = np.diag(sigma ** 2)
    else:
        cov = np.outer(sigma, sigma)

    return add_uncertainty(
        result=result,
        cov=cov,
        key=key,
        category=category,
        target=target,
        unc=frac_unc,
        sum_value=float(np.mean(frac_unc)),
    )


def add_unisim_uncertainty(
    result: SystematicsOutput,
    alt_hist: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
):
    """
    Add a single-universe (unisim) uncertainty to a SystematicsOutput.

    The alternate histogram is interpreted as one shifted prediction.
    The covariance contribution is built from:
        delta = alt_hist - hist_cv
        cov = outer(delta, delta)

    Parameters
    ----------
    result
        Existing systematics result object.
    alt_hist
        Alternate prediction histogram with the same shape as result.hist_cv.
    key
        Dictionary/dataframe key label for the new source.
    category
        Category label for the dataframe entry. Defaults to `key`.
    target
        Where to apply this uncertainty: "rate", "xsec", or "both".
    """
    if category is None:
        category = key

    cv_hist = np.asarray(result.hist_cv, dtype=float)
    alt_hist = np.asarray(alt_hist, dtype=float)
    if alt_hist.shape != cv_hist.shape:
        raise ValueError(
            f"alt_hist shape {alt_hist.shape} does not match hist_cv shape {cv_hist.shape}"
        )

    delta = alt_hist - cv_hist
    uni_cov = np.outer(delta, delta)
    unc = np.divide(np.abs(delta), cv_hist, out=np.zeros_like(cv_hist, dtype=float), where=cv_hist > 0)

    return add_uncertainty(
        result=result,
        cov=uni_cov,
        key=key,
        category=category,
        target=target,
        unc=unc,
        sum_value=float(np.mean(unc)),
    )

def get_total_cov(reco_df, reco_var, bins, mcbnb_pot,
                  normalize=False, selection_kwargs=None, projected_pot=1e20, 
                  xsec_inputs: XSecInputs | None = None):
    """
    Get the total event-rate covariance matrix and systematic dataframe for a
    given variable. Optionally also compute the xsec covariance matrix and
    systematic dataframe when xsec_inputs are provided.

    The data statistical uncertainty is added as a separate "Datastat" entry in
    the returned event-rate dataframe, and in the xsec dataframe when requested.

    The returned result also includes systematic dictionaries:
    - rateate_syst_dict (includes DetVar keys)
    - xsec_syst_dict (includes DetVar keys, when xsec_inputs are provided)
    """

    def _sum_covariances(syst_dicts, nbins):
        total_cov = np.zeros((nbins - 1, nbins - 1))
        for syst_dict in syst_dicts:
            for entry in syst_dict.values():
                total_cov += entry['cov']
        return total_cov

    if selection_kwargs is None:
        selection_kwargs = {}

    # detvar_dict = {}
    with open('/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/mc/dfs/detvars/detvar_dict_updated.pkl', 'rb') as f:
        detvar_dict = pickle.load(f)
    with open('/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/mc/dfs/detvars/recomb_dict_updated2.pkl', 'rb') as f:
        recomb_dict = pickle.load(f)
    # combine recomb_dict and detvar_dict
    for key in recomb_dict.keys():
        detvar_dict[key] = recomb_dict[key]    # DetVar systematics from selected/control, then concatenate and matrixify.

    sorted_df = ensure_lexsorted(reco_df, axis=1)
    cv_hist = get_hist1d(data=sorted_df[reco_var], weights=sorted_df.flux_pot_norm, bins=bins)

    detv_syst_dict = get_detvar_systs(detvar_dict, reco_var, bins, normalize=normalize, **selection_kwargs)
    rate_syst_dict = get_syst(reco_df=reco_df, reco_var=reco_var, bins=bins, normalize=normalize)
    rate_total_syst_dict = {**rate_syst_dict, **detv_syst_dict}
    rate_syst_df = get_syst_df([rate_syst_dict, detv_syst_dict], cv_hist)

    data_err = np.sqrt(get_hist1d(data=sorted_df[reco_var], weights=reco_df.weights_mc, bins=bins)* (projected_pot / mcbnb_pot))
    flux_scale = integrated_flux * (projected_pot / 1e6)
    data_unc = np.divide(data_err, flux_scale * cv_hist, out=np.zeros_like(data_err, dtype=float), where=cv_hist > 0)
    data_syst_df = pd.DataFrame({'key': ['Datastat'], 'category': ['Datastat'], 'unc': [data_unc], 'sum': [np.mean(data_unc)], 'top5': [False]})
    
    offbeam_mask = sorted_df.signal == signal_dict['offbeam']
    offbeam_hist = get_hist1d(data=sorted_df[offbeam_mask][reco_var], weights=sorted_df[offbeam_mask].flux_pot_norm, bins=bins)
    offbeam_cov  = np.diag(offbeam_hist)

    rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    rate_cov = _sum_covariances([rate_syst_dict, detv_syst_dict], len(bins))

    if xsec_inputs is None:
        return SystematicsOutput(
            hist_cv=cv_hist,
            rate_cov=rate_cov,
            rate_syst_df=rate_syst_df,
            rate_syst_dict=rate_total_syst_dict,
        )

    xsec_syst_dict = get_syst(
        reco_df=reco_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
        xsec_inputs=xsec_inputs,
    )
    xsec_total_syst_dict = {**xsec_syst_dict, **detv_syst_dict}
    xsec_syst_df = get_syst_df([xsec_syst_dict, detv_syst_dict], cv_hist)
    xsec_syst_df = pd.concat([xsec_syst_df, data_syst_df], ignore_index=True)
    xsec_cov = _sum_covariances([xsec_syst_dict, detv_syst_dict], len(bins))
    
    return SystematicsOutput(
        hist_cv=cv_hist,
        rate_cov=rate_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_total_syst_dict,
        xsec_cov=xsec_cov,
        xsec_syst_df=xsec_syst_df,
        xsec_syst_dict=xsec_total_syst_dict,
    )
    
'''
Control Region
'''

def _combine_syst_hist_dicts(sel_dict, ctrl_dict):
    combined = {}
    shared_keys = set(sel_dict).intersection(set(ctrl_dict))

    missing_sel = sorted(set(ctrl_dict) - set(sel_dict))
    missing_ctrl = sorted(set(sel_dict) - set(ctrl_dict))
    if missing_sel or missing_ctrl:
        raise ValueError(
            "Systematic keys do not match between selected and control regions. "
            f"Missing in selected: {missing_sel}; missing in control: {missing_ctrl}"
        )

    for key in shared_keys:
        sel_h = sel_dict[key]["hists"]
        ctrl_h = ctrl_dict[key]["hists"]

        if sel_h.ndim == 1:
            sel_h = sel_h.reshape(-1, 1)
        if ctrl_h.ndim == 1:
            ctrl_h = ctrl_h.reshape(-1, 1)

        if sel_h.shape[1] != ctrl_h.shape[1]:
            raise ValueError(
                f"Universe-count mismatch for {key}: "
                f"selected={sel_h.shape[1]}, control={ctrl_h.shape[1]}"
            )

        combined[key] = {"hists": np.concatenate([sel_h, ctrl_h], axis=0)}

    return combined

def _add_matrices_in_place(syst_hist_dict, cv_hist):
    for key in syst_hist_dict:
        cov, cov_frac, corr = calc_matrices(syst_hist_dict[key]["hists"], cv_hist)
        syst_hist_dict[key]["cov"] = cov
        syst_hist_dict[key]["cov_frac"] = cov_frac
        syst_hist_dict[key]["corr"] = corr
    return syst_hist_dict

def _sum_covariances_from_dicts(syst_dicts, n_combined_bins):
    total_cov = np.zeros((n_combined_bins, n_combined_bins))
    for syst_dict in syst_dicts:
        for entry in syst_dict.values():
            total_cov += entry["cov"]
    return total_cov


def get_total_cov_combined(
    reco_df,
    reco_control_df,
    reco_var,
    bins,
    normalize=False,
    selected_selection_kwargs=None,
    control_selection_kwargs=None,
    xsec_inputs=None,
):
    # ! TODO: add data statistics systematics
    """
    Build covariance/results for a concatenated selected+control measurement.

    Workflow:
    1) run get_syst_hists separately in selected and control regions,
    2) concatenate per-systematic histograms,
    3) run calc_matrices on the concatenated histograms.

    Notes:
    - `reco_var` and `bins` are shared between selected/control.
    - selection kwargs can still differ between selected/control.
    - Returned result includes systematic dictionaries:
      rateate_syst_dict includes DetVar keys, and xsec_syst_dict (if requested)
      includes DetVar keys.
    """
    if selected_selection_kwargs is None:
        selected_selection_kwargs = {}
    if control_selection_kwargs is None:
        control_selection_kwargs = {}

    # CV histograms for selected and control, then concatenate.
    reco_df = ensure_lexsorted(reco_df, axis=1)
    reco_control_df = ensure_lexsorted(reco_control_df, axis=1)

    cv_sel = get_hist1d(data=reco_df[reco_var], weights=reco_df.flux_pot_norm, bins=bins)
    cv_ctrl = get_hist1d(data=reco_control_df[reco_var], weights=reco_control_df.flux_pot_norm, bins=bins)
    cv_hist = np.concatenate([cv_sel, cv_ctrl])

    # Event-rate reweight systematics from selected/control, then concatenate and matrixify.
    rate_sel_hists, _ = get_syst_hists(
        reco_df=reco_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
    )
    rate_ctrl_hists, _ = get_syst_hists(
        reco_df=reco_control_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
    )
    rate_syst_dict = _add_matrices_in_place(
        _combine_syst_hist_dicts(rate_sel_hists, rate_ctrl_hists),
        cv_hist,
    )
    
    with open('/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/mc/dfs/detvars/detvar_dict_updated.pkl', 'rb') as f:
        detvar_dict = pickle.load(f)
    with open('/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/mc/dfs/detvars/recomb_dict_updated2.pkl', 'rb') as f:
        recomb_dict = pickle.load(f)
    # combine recomb_dict and detvar_dict
    for key in recomb_dict.keys():
        detvar_dict[key] = recomb_dict[key]    # DetVar systematics from selected/control, then concatenate and matrixify.
    # detvar_dict = {}
    detv_sel_hists = get_detvar_systs(
        detvar_dict,
        reco_var,
        bins,
        normalize=normalize,
        **selected_selection_kwargs,
    )
    detv_ctrl_hists = get_detvar_systs(
        detvar_dict,
        reco_var,
        bins,
        normalize=normalize,
        **control_selection_kwargs,
    )
    detv_syst_dict = _add_matrices_in_place(
        _combine_syst_hist_dicts(detv_sel_hists, detv_ctrl_hists),
        cv_hist,
    )
    rate_total_syst_dict = {**rate_syst_dict, **detv_syst_dict}

    rate_syst_df = get_syst_df([rate_syst_dict, detv_syst_dict], cv_hist)
    # rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    n_combined_bins = len(cv_hist)
    rate_cov = _sum_covariances_from_dicts([rate_syst_dict, detv_syst_dict], n_combined_bins)

    if xsec_inputs is None:
        return SystematicsOutput(
            hist_cv=cv_hist,
            rate_cov=rate_cov,
            rate_syst_df=rate_syst_df,
            rate_syst_dict=rate_total_syst_dict,
        )
    
    print("Calculating cross-section systematics...")
    xsec_sel_hists, _ = get_syst_hists(
        reco_df=reco_df,
        reco_var=reco_var,
        bins=bins,
        normalize=normalize,
        xsec_inputs=xsec_inputs,
    )
    xsec_syst_dict = _add_matrices_in_place(
        _combine_syst_hist_dicts(xsec_sel_hists, rate_ctrl_hists),
        cv_hist,
    )
    xsec_total_syst_dict = {**xsec_syst_dict, **detv_syst_dict}
    xsec_syst_df = get_syst_df([xsec_syst_dict, detv_syst_dict], cv_hist)
    # rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    n_combined_bins = len(cv_hist)
    xsec_cov = _sum_covariances_from_dicts([xsec_syst_dict, detv_syst_dict], n_combined_bins)
    
    return SystematicsOutput(
        hist_cv=cv_hist,
        rate_cov=rate_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_total_syst_dict,
        xsec_cov=xsec_cov,
        xsec_syst_df=xsec_syst_df,
        xsec_syst_dict=xsec_total_syst_dict,
    )
    
    
    