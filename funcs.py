import numpy as np
import pandas as pd
from dataclasses import replace
import warnings
import pickle
from tqdm import tqdm
from .utils import ensure_lexsorted, apply_event_mask
from .io import load_dfs
from .selection import select, select_sideband
from .histogram import get_hist1d, get_hist2d
from .syst import *
from .classes import SystematicsOutput, XSecInputs
from .constants import integrated_flux, signal_dict
from . import config

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


def _sum_covariances_from_dicts(syst_dicts, n_bins):
    total_cov = np.zeros((n_bins, n_bins))
    for syst_dict in syst_dicts:
        for entry in syst_dict.values():
            total_cov += entry["cov"]
    return total_cov


def _load_detvar_dicts(
    detvar_files=None,
):
    """Load and combine detector variation dictionaries from pickle files.
    
    Parameters
    ----------
    detvar_files : list of str, optional
        List of paths to detvar dictionary pickle files. If None, uses config.DETVAR_DICT_FILES.
    
    Returns
    -------
    dict
        Combined detector variation and recombination dictionary.
    """
    if detvar_files is None:
        detvar_files = config.DETVAR_DICT_FILES
        
    combined_dict = {}
    for detvar_file in detvar_files:
        with open(detvar_file, 'rb') as f:
            file_dict = pickle.load(f)
            combined_dict.update(file_dict)
    
    return combined_dict


def _block_diag_cov(cov_a, cov_b):
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)
    if cov_a.ndim != 2 or cov_b.ndim != 2 or cov_a.shape[0] != cov_a.shape[1] or cov_b.shape[0] != cov_b.shape[1]:
        raise ValueError("cov_a and cov_b must be square 2D covariance matrices")
    n_a = cov_a.shape[0]
    n_b = cov_b.shape[0]
    out = np.zeros((n_a + n_b, n_a + n_b), dtype=float)
    out[:n_a, :n_a] = cov_a
    out[n_a:, n_a:] = cov_b
    return out


def _hists_from_frac_unc(cv_hist: np.ndarray, frac_unc: np.ndarray) -> np.ndarray:
    cv_hist = np.asarray(cv_hist, dtype=float)
    frac_unc = np.asarray(frac_unc, dtype=float)
    if frac_unc.shape != cv_hist.shape:
        raise ValueError(f"frac_unc shape {frac_unc.shape} does not match hist_cv shape {cv_hist.shape}")
    # One shifted universe whose bin-wise fractional shift matches frac_unc.
    return (cv_hist * (1.0 + frac_unc)).reshape(-1, 1)


def _apply_norm_and_intime_uncertainties(
    result: SystematicsOutput,
    intime_cov: np.ndarray | None = None,
    pot_norm_unc: float = 0.02,
    ntargets_unc: float = 0.01,
):
    updated = add_flat_norm_uncertainty(
        result=result,
        frac_unc=pot_norm_unc,
        key="BeamExposure",
        category="BeamExposure",
    )
    updated = add_flat_norm_uncertainty(
        result=updated,
        frac_unc=ntargets_unc,
        key="NTargets",
        category="NTargets",
    )

    if intime_cov is not None:
        cv_hist = np.asarray(updated.hist_cv, dtype=float)
        intime_unc = np.divide(
            np.sqrt(np.diag(intime_cov)),
            cv_hist,
            out=np.zeros_like(cv_hist, dtype=float),
            where=cv_hist > 0,
        )
        updated = add_uncertainty(
            result=updated,
            cov=np.asarray(intime_cov, dtype=float),
            key="Cosmic",
            category="Cosmic",
            target="both" if updated.has_xsec else "rate",
            unc=intime_unc,
            hists=_hists_from_frac_unc(cv_hist, intime_unc),
            sum_value=float(np.mean(intime_unc)),
        )

    return updated


def add_uncertainty(
    result: SystematicsOutput,
    cov: np.ndarray,
    key: str,
    category: str | None = None,
    target: str = "both",
    unc: np.ndarray | None = None,
    hists: np.ndarray | None = None,
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
    hists
        Optional universe histogram array stored in the systematic dictionary.
        Shape must be (nbins, nuniverses) or (nbins,).
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

    if hists is not None:
        hists = np.asarray(hists, dtype=float)
        if hists.ndim == 1:
            hists = hists.reshape(-1, 1)
        if hists.ndim != 2 or hists.shape[0] != cv_hist.size:
            raise ValueError(
                f"hists must have shape (nbins, nuniverses); got {hists.shape} for nbins={cv_hist.size}"
            )

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
    if hists is not None:
        syst_entry["hists"] = hists

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
        hists=_hists_from_frac_unc(cv_hist, unc),
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
        hists=alt_hist.reshape(-1, 1),
        sum_value=float(np.mean(unc)),
    )

def get_intime_cov (selected_df, var, bins, 
                    mcbnb_ngen,
                    mcbnb_pot,
                    threshold = 0.05,
                    event_mask: str | None = "all",
                    select_region: str = "signal",
                    **selection_kwargs):
    mcint_dfs = load_dfs(config.INTIME_FILE,['histgenevtdf','nuecc'])
    scale = mcbnb_ngen/mcint_dfs['histgenevtdf'].TotalGenEvents.sum()
    if select_region == "signal":
        mcint_df = select(mcint_dfs['nuecc'], savedict=False)
    elif select_region == "control":
        mcint_df = select_sideband(mcint_dfs['nuecc'], savedict=False)
    else:
        mcint_df = select(mcint_dfs['nuecc'], savedict=False, **selection_kwargs)
    mcint_df[('weights_mc', '', '', '', '', '')] = scale
    mcint_df[('flux_pot_norm', '', '', '', '', '')] = mcint_df.weights_mc/(integrated_flux * (mcbnb_pot / 1e6))
    # sort to avoid performance warning
    selected_df = ensure_lexsorted(selected_df,axis=1)
    mcint_df = ensure_lexsorted(mcint_df,axis=1)
    selected_df = apply_event_mask(selected_df, event_mask)
    mcint_df = apply_event_mask(mcint_df, event_mask)
    
    cv_hist = get_hist1d(data=selected_df[var], bins=bins, 
                             weights=selected_df.flux_pot_norm)
    # remove offbeam contribution
    selected_no_offbeam_df = selected_df[selected_df.signal!=signal_dict['offbeam']]
    cv_hist_removed = get_hist1d(data = selected_no_offbeam_df[var],
                                     bins=bins, 
                                     weights = selected_no_offbeam_df.flux_pot_norm)
    
    # add the intime contribution
    int_hist = get_hist1d(data=mcint_df[var], bins=bins,
                              weights=mcint_df.flux_pot_norm)
    dv_hist = cv_hist_removed + int_hist

    matrices = calc_matrices(dv_hist.reshape(len(bins)-1,-1),cv_hist)
    cov = matrices[0]
    unc = np.sqrt(np.diag(cov))/cv_hist
    # if the uncertainty is large enough, keep it for that bin
    # otherwise, we use the largest non-large uncertainty as a uniform uncertainty for all 
    large_unc = unc > threshold
    if np.any(~large_unc):
        uniform_unc_val = np.max(unc[~large_unc])
    else:
        uniform_unc_val = np.max(unc)
    unc_final = np.where(large_unc, unc, uniform_unc_val)
    # apply fully correlated uncertainty
    cov_final = np.outer(unc_final*cv_hist, unc_final*cv_hist)
    return cov_final
    
def get_total_cov(reco_df, reco_var, bins, mcbnb_pot,
                  selection_kwargs=None, projected_pot=1e20, 
                  mcbnb_ngen: float | None = None,
                  intime_threshold: float = 0.05,
                  event_mask: str | None = "all",
                  select_region: str = "signal",
                  xsec_inputs: XSecInputs | None = None):
    """
    Get the total event-rate covariance matrix and systematic dataframe for a
    given variable. Optionally also compute the xsec covariance matrix and
    systematic dataframe when xsec_inputs are provided.

    The data statistical uncertainty is added as a separate "Datastat" entry in
    the returned event-rate dataframe, and in the xsec dataframe when requested.

    Parameters
    ----------
    reco_df : pd.DataFrame
        Reconstructed event data
    reco_var : str or tuple
        Variable to histogram
    bins : np.ndarray
        Bin edges
    mcbnb_pot : float
        Monte Carlo BNB POT (or the main sample to normalize to)
    selection_kwargs : dict, optional
        Additional selection cuts to apply
    projected_pot : float, optional
        Projected POT for data statistics calculation
    mcbnb_ngen : float, optional
        Number of generated events for in-time calculation
    intime_threshold : float, optional
        Threshold for in-time uncertainty handling, default is 0.05 (5%)
    event_mask : str or None, optional
        Event mask ('all', 'signal', 'background'), default is 'all'
    select_region : str, optional
        Which detector variation dictionary to use: 'signal' (default), 'control', or 'all'.
    xsec_inputs : XSecInputs, optional
        Cross-section calculation inputs

    Returns
    -------
    SystematicsOutput
        Systematic uncertainties with rate (and optionally cross-section) covariances.
    
    Notes
    -----
    - rateate_syst_dict (includes DetVar keys)
    - xsec_syst_dict (includes DetVar keys, when xsec_inputs are provided)
    """

    if selection_kwargs is None:
        selection_kwargs = {}

    # Map select_region to appropriate config path
    select_region_map = {
        "signal": config.DETVAR_DICT_SIGNAL,
        "control": config.DETVAR_DICT_CONTROL,
        "all": config.DETVAR_DICT_FILES,
    }
    
    if select_region not in select_region_map:
        raise ValueError(f"select_region must be one of {list(select_region_map.keys())}, got '{select_region}'")
    
    detvar_path = select_region_map[select_region]
    
    # Load the appropriate detvar dictionary
    print(f"Loading detvar dictionary for region: {select_region}")
    print(f"  Path: {detvar_path}")
    if select_region == "all":
        detvar_dict = _load_detvar_dicts(detvar_path)
    else:
        # Load single file
        with open(detvar_path, 'rb') as f:
            detvar_dict = pickle.load(f)
    print(f"  Loaded {len(detvar_dict)} detector variation entries")

    sorted_df = apply_event_mask(ensure_lexsorted(reco_df, axis=1), event_mask)
    cv_hist = get_hist1d(data=sorted_df[reco_var], weights=sorted_df.flux_pot_norm, bins=bins)

    detv_syst_dict = get_detvar_systs(
        detvar_dict,
        reco_var,
        bins,
        event_mask=event_mask,
        **selection_kwargs,
    )
    rate_syst_dict = get_syst(reco_df=sorted_df, reco_var=reco_var, bins=bins)
    rate_total_syst_dict = {**rate_syst_dict, **detv_syst_dict}
    rate_syst_df = get_syst_df([rate_syst_dict, detv_syst_dict], cv_hist)

    data_err = np.sqrt(get_hist1d(data=sorted_df[reco_var], weights=reco_df.weights_mc, bins=bins)* (projected_pot / mcbnb_pot))
    flux_scale = integrated_flux * (projected_pot / 1e6)
    data_unc = np.divide(data_err, flux_scale * cv_hist, out=np.zeros_like(data_err, dtype=float), where=cv_hist > 0)
    data_syst_df = pd.DataFrame({'key': ['Datastat'], 'category': ['Datastat'], 'unc': [data_unc], 'sum': [np.mean(data_unc)], 'top5': [False]})

    rate_syst_df = pd.concat([rate_syst_df, data_syst_df], ignore_index=True)
    rate_cov = _sum_covariances_from_dicts([rate_syst_dict, detv_syst_dict], cv_hist.size)

    intime_cov = None
    if mcbnb_ngen is not None:
        intime_cov = get_intime_cov(
            selected_df=sorted_df,
            var=reco_var,
            bins=bins,
            mcbnb_ngen=mcbnb_ngen,
            mcbnb_pot=mcbnb_pot,
            threshold=intime_threshold,
            event_mask=event_mask,
            select_region=select_region,
            **selection_kwargs,
        )

    if xsec_inputs is None:
        base_output = SystematicsOutput(
            hist_cv=cv_hist,
            rate_cov=rate_cov,
            rate_syst_df=rate_syst_df,
            rate_syst_dict=rate_total_syst_dict,
        )
        return _apply_norm_and_intime_uncertainties(base_output, intime_cov=intime_cov)

    xsec_syst_dict = get_syst(
        reco_df=sorted_df,
        reco_var=reco_var,
        bins=bins,
        xsec_inputs=xsec_inputs,
    )
    xsec_total_syst_dict = {**xsec_syst_dict, **detv_syst_dict}
    xsec_syst_df = get_syst_df([xsec_syst_dict, detv_syst_dict], cv_hist)
    xsec_syst_df = pd.concat([xsec_syst_df, data_syst_df], ignore_index=True)
    xsec_cov = _sum_covariances_from_dicts([xsec_syst_dict, detv_syst_dict], cv_hist.size)

    base_output = SystematicsOutput(
        hist_cv=cv_hist,
        rate_cov=rate_cov,
        rate_syst_df=rate_syst_df,
        rate_syst_dict=rate_total_syst_dict,
        xsec_cov=xsec_cov,
        xsec_syst_df=xsec_syst_df,
        xsec_syst_dict=xsec_total_syst_dict,
    )
    return _apply_norm_and_intime_uncertainties(base_output, intime_cov=intime_cov)