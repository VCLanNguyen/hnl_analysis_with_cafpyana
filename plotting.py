"""Useful plotting helpers for nueana: stacked MC, PDG breakdowns, and data overlays.

This module provides:
- plot_var: unified function to plot either signal-type stacks or PDG-type stacks.
- data_plot_overlay: draw data points with Poisson errors on top of MC stacks.
- plot_mc_data: convenience function that builds an MC+data figure with ratio subplot.

All functions accept both plain and MultiIndex DataFrames (the code will attempt to
ensure lexsorted axes via ``ensure_lexsorted`` imported from ``.utils``).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pandas as pd
import warnings

from .constants import signal_dict, signal_labels, pdg_dict, signal_colors, generic_dict, generic_labels, generic_colors
from .utils import ensure_lexsorted
from .syst import *
from .histogram import *

def annotate_internal(ax):
    ax.annotate("SBND Internal", xy=(0.0, 1.02), xycoords='axes fraction', ha='left',color='gray',fontweight='bold')

def plot_var(df: pd.DataFrame,
             var: tuple | str,
             bins: np.ndarray,
             ax = None,
             xlabel: str = "",
             ylabel: str = "",
             title: str = "",
             counts: bool = False,
             percents: bool = False,
             scale: float = 1.0,
             normalize: bool = False,
             mult_factor: float = 1.0,
             cut_val: list[float] | None = None,
             plot_err: bool = True,
             systs: bool | np.ndarray = None,
             pdg: bool = False,
             pdg_col: tuple | str = 'pfp_shw_truth_p_pdg',
             hatch: list[str] | None = None,
             bin_labels : list[str] | None = None,
             generic: bool = False,
             overflow: bool = True,
             hist_filled: bool = True,
             error_legend: bool = True,
             legend_kwargs: dict | None = None,
             
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot a variable as stacked histograms for signal categories or PDG types.

    This function supports three modes controlled by ``pdg`` and ``generic``:
    - pdg=False, generic=False (default): stack by interaction type using ``signal_dict``.
    - pdg=True: stack by particle PDG using ``pdg_dict``; adds 'cosmic' and
      'other' as the last two categories.
    - generic=True: stack by broad category using ``generic_dict`` (nuFV, nonFV,
      dirt, cosmic). Takes precedence over ``pdg`` if both are True.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    var : tuple | str
        Column name (or multi-index tuple) to histogram.
    bins : np.ndarray
        Bin edges for the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None the current axis is used.
    xlabel : str, optional
        X axis label. Defaults to the variable name when empty.
    ylabel : str, optional
        Y axis label. Defaults to 'Counts' when empty.
    title : str, optional
        Plot title. Defaults to the variable name when empty.
    counts : bool, default False
        If True, append event counts to legend labels.
    scale : float, default 1.0
        Scale factor applied to the histogram.
    normalize : bool, default False
        If True, normalize histograms so integral equals 1 (uses bin widths from ``bins``).
    mult_factor : float, default 1.0
        Multiplicative factor applied to the first category (index 0). Intended for quick
        visual scaling only; error propagation is not adjusted at all. 
    cut_val : list, optional
        List of x-values at which to draw vertical cut lines.
    plot_err : bool, default True
        If True, draw MC statistical (and optional systematic) error bands.
    systs : bool | np.ndarray, optional
        if True, calculates and plots systematic uncertainties stored in the input dataframe. 
        if given as a numpy array, uses the provided values as total uncertainties 
        (e.g. from an external calculation) and plots them without attempting to separate stat/syst.
        if False or None, no error bands are plotted.
    pdg : bool, default False
        When True, split histograms by PDG (uses ``pdg_col``). Otherwise split by signal type.
    pdg_col : tuple | str, default 'pfp_shw_truth_p_pdg'
        Column (or multi-index tuple) containing the PDG code per particle (used when ``pdg``
        is True).
    hatch : list, optional
        Optional hatch patterns per category.
    generic : bool, default False
        When True, stack by broad category (FV neutrino, non-FV, dirt, cosmic) using
        ``generic_dict`` / ``generic_labels`` / ``generic_colors``. Takes precedence
        over ``pdg`` if both are True.
    overflow : bool, optional
        If True (default), values above bins[-1] are clipped to bins[-1] - 1e-10
        to fold overflow into the last bin. If False, uses standard numpy histogram
        behavior with no clipping.
    hist_filled : bool, default True
        If True, use filled histograms. If False, use step histograms with no fill.
    error_legend : bool, default True
        If True, include a legend entry of MC stat./syst. uncertainties when ``plot_err`` is True.
    legend_kwargs : dict, optional
        Dictionary of keyword arguments to pass to ax.legend(). These will override
        the default legend settings (ncol=2, loc='upper right').
    
    Returns
    -------
    bins, steps, total_err
        - bins: the input bin edges
        - steps: array of cumulative step values used for plotting (shape (n_categories, len(bins)))
        - total_err: combined stat + syst per bin (length = n_bins)
    """
    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)

    def _col_has_token(col, token: str) -> bool:
        if isinstance(col, tuple):
            return token in "_".join(map(str, col))
        return token in str(col)

    weight = any(_col_has_token(col, "weights_mc") for col in df.columns)

    # Convert once to numpy arrays to avoid repeated pandas filtering per category.
    var_vals = np.asarray(df[var])
    if overflow:
        var_vals = np.clip(var_vals, bins[0], bins[-1] - 1e-10)
    signal_vals = np.asarray(df['signal'])
    weights_vals = np.asarray(df['weights_mc']) if weight else None
    abs_pdg_vals = np.abs(np.asarray(df[pdg_col])) if pdg else None
    
    colors = generic_colors if generic else signal_colors
    if ax is None: ax = plt.gca()
    category_dict = generic_dict if generic else (pdg_dict if pdg else signal_dict)
    category_labels = generic_labels if generic else signal_labels
    ncategories = len(generic_dict) if generic else (len(pdg_dict)+3 if pdg else len(signal_dict))
    if hatch == None: hatch = [""]*ncategories
    alpha = 0.25 if pdg else 0.4
    
    hists       = np.zeros((ncategories,len(bins)-1)) # this is for storing the histograms
    steps       = np.zeros((ncategories,len(bins))) # this is for plotting
    stats_var   = np.zeros((ncategories,len(bins)-1))
    bin_widths  = np.diff(bins)
    
    stats_err   = np.zeros(len(bins)-1)
    systs_err   = np.zeros(len(bins)-1)
    
    # Check if systs is provided as array (already includes stats)
    systs_is_array = isinstance(systs, np.ndarray)

    if pdg==False:
        for i, entry in enumerate(category_dict):
            this_cat = category_dict[entry]
            cat_mask = (signal_vals == this_cat)
            cat_weights = weights_vals[cat_mask] if weight else None
            hists[i] = np.histogram(var_vals[cat_mask], bins=bins, weights=cat_weights)[0]
            if weight:
                stats_var[i] = np.histogram(var_vals[cat_mask], bins=bins, weights=cat_weights**2)[0]
            else:
                stats_var[i] = hists[i]
    else: 
        nu_mask = signal_vals < signal_dict['cosmic']
        cosmic_mask = signal_vals == signal_dict['cosmic']
        offbeam_mask = signal_vals == signal_dict['offbeam']
        # "other" starts from nu entries then removes known PDGs.
        other_mask = nu_mask.copy()

        for i, key in enumerate(list(pdg_dict.keys())):
            pdg_value = pdg_dict[key]['pdg']
            pdg_mask = nu_mask & (abs_pdg_vals == pdg_value)
            cat_weights = weights_vals[pdg_mask] if weight else None
            hists[i] = np.histogram(var_vals[pdg_mask], bins=bins, weights=cat_weights)[0]
            if weight:
                stats_var[i] = np.histogram(var_vals[pdg_mask], bins=bins, weights=cat_weights**2)[0]
            else:
                stats_var[i] = hists[i]
            other_mask &= (abs_pdg_vals != pdg_value)

        other_weights = weights_vals[other_mask] if weight else None
        cosmic_weights = weights_vals[cosmic_mask] if weight else None
        offbeam_weights = weights_vals[offbeam_mask] if weight else None
        hists[-1] = np.histogram(var_vals[other_mask], bins=bins, weights=other_weights)[0]
        hists[-2] = np.histogram(var_vals[offbeam_mask], bins=bins, weights=offbeam_weights)[0]
        hists[-3] = np.histogram(var_vals[cosmic_mask], bins=bins, weights=cosmic_weights)[0]
        if weight:
            stats_var[-1] = np.histogram(var_vals[other_mask], bins=bins, weights=other_weights**2)[0]
            stats_var[-2] = np.histogram(var_vals[offbeam_mask], bins=bins, weights=offbeam_weights**2)[0]
            stats_var[-3] = np.histogram(var_vals[cosmic_mask], bins=bins, weights=cosmic_weights**2)[0]
        else:
            stats_var[-1] = hists[-1]
            stats_var[-2] = hists[-2]
            stats_var[-3] = hists[-3]
    
    # ! THIS ASSUMES that the PDG of interest and the signal type of interest are both index 0
    # ! e.g. for nueCC (signal==0), e- is the first entry in the pdg_dict
    hists    *= scale 
    hists[0] = mult_factor*hists[0]

    stats_var *= scale**2
    stats_var[0] *= mult_factor**2

    # storing the sum of each category in case we want to display it
    hist_counts = np.sum(hists,axis=1)
    total_hist_count = np.sum(hist_counts)

    # check if systematic cols are inside the df
    found_systs = False
    if (systs_is_array == False) and (systs == True):
        found_systs = any(_col_has_token(col, "univ_") for col in df.columns)
        
    if systs_is_array:
        # systs array already includes statistical uncertainty
        found_systs = True
        systs_arr = systs
        syst_dict = {}
    elif (systs==True) & (found_systs): 
        syst_dict = get_syst(indf=df,var=var,bins=bins,scale=True)
        total_cov = np.zeros(len(bins)-1)
        for key in syst_dict.keys():
            total_cov += np.diag(syst_dict[key]['cov'])
        systs_arr = np.sqrt(total_cov)
    else:
        if (systs==True) & (found_systs==False):
            print("can't find universes in the input df, ignoring systematic error bars")
            systs=False
        systs_arr = np.zeros(len(bins)-1)
        syst_dict = {}

    # Only calculate statistical error if systs not provided as array
    if not systs_is_array:
        stats_err = np.sqrt(np.sum(stats_var, axis=0))

    # Systematic error calculation
    systs_err = systs_arr * scale

    if normalize:
        total_integral = np.sum(hists * bin_widths)
        hists = hists / total_integral
        if not systs_is_array:
            stats_err = stats_err / total_integral
        systs_err = systs_err / total_integral
        
    for i in range(ncategories):
        color = colors[i]
        if pdg: 
            plot_label = (list(pdg_dict.keys())+['cosmic']+['offbeam']+['other'])[i]
            if 'cosmic' in plot_label:
                color = colors[signal_dict['cosmic']]
            if 'offbeam' in plot_label:
                color = colors[signal_dict['offbeam']]
        else: plot_label = category_labels[i]
        if (mult_factor!= 1.0) & (i==0): plot_label +=  f" [x{mult_factor}]"
        if counts:
            plot_label += f" ({int(hist_counts[i]):,})" if hist_counts[i] < 1e6 else f" ({hist_counts[i]:.2e})"
        if percents and total_hist_count > 0:
            plot_label += f" ({hist_counts[i]/total_hist_count*100:.1f}%)"
        bottom=steps[i-1] if i>0 else 0
        # steps needs the first entry to be repeated!
        steps[i] = np.insert(hists[i],obj=0,values=hists[i][0]) + bottom;

        #if zero contribution to histogram don't plot
        if hist_counts[i] == 0: continue

        if hist_filled:
            ax.fill_between(bins, bottom, steps[i], step="pre", 
                             facecolor=mpl.colors.to_rgba(color,alpha),
                             edgecolor=mpl.colors.to_rgba(color,1.0),  
                             lw=1.5, 
                             hatch=hatch[i],zorder=(ncategories-i),label=plot_label)
        else:
            edge_baseline = steps[i-1][1:] if i > 0 else 0.0
            ax.stairs(hists[i], bins, baseline=edge_baseline, color=color, lw=2.0,
                      label=plot_label, zorder=(ncategories-i))
    
    if plot_err: 
        systs_options = {"step":"pre", "color":mpl.colors.to_rgba("gray", alpha=0.75),
                         "lw":0.0,"facecolor":"none","hatch":"xxx",
                         "zorder":ncategories+1}
        
        # fill_between needs the *first* entry to be repeated...
        if systs_is_array:
            # systs array already includes both stat + syst
            min_total_err = steps[-1] - np.append(systs_err[0], systs_err)
            pls_total_err = steps[-1] + np.append(systs_err[0], systs_err)
            pltlabel = "MC stat.+syst." if error_legend else ""
            ax.fill_between(bins, min_total_err, pls_total_err, **systs_options, label=pltlabel)
        elif found_systs:
            # Separate stat and syst bands
            stats_options = {"step":"pre", "color":mpl.colors.to_rgba("gray", alpha=0.9),
                             "lw":0.0,"facecolor":"none","hatch":"....",
                             "zorder":ncategories+1}
            min_systs_err = steps[-1]     - np.append(systs_err[0],systs_err)
            pls_systs_err = steps[-1]     + np.append(systs_err[0],systs_err)
            min_stats_err = min_systs_err - np.append(stats_err[0],stats_err)
            pls_stats_err = pls_systs_err + np.append(stats_err[0],stats_err)
            pltlabel = "MC syst" if error_legend else ""
            ax.fill_between(bins, min_systs_err, pls_systs_err, **systs_options,label=pltlabel)
            pltlabel = "MC stat" if error_legend else ""
            ax.fill_between(bins, min_systs_err, min_stats_err, **stats_options,label=pltlabel)
            ax.fill_between(bins, pls_systs_err, pls_stats_err, **stats_options)
        else: 
            # Only stat errors
            stats_options = {"step":"pre", "color":mpl.colors.to_rgba("gray", alpha=0.9),
                             "lw":0.0,"facecolor":"none","hatch":"....",
                             "zorder":ncategories+1}
            min_stats_err = steps[-1] - np.append(stats_err,stats_err[-1])
            pls_stats_err = steps[-1] + np.append(stats_err,stats_err[-1])
            pltlabel = "MC stat." if error_legend else ""
            ax.fill_between(bins, min_stats_err, pls_stats_err, **stats_options,label=pltlabel)

    cut_line_zorder = ncategories + 2
    if cut_val != None:
        for i in range(len(cut_val)):
            ax.axvline(cut_val[i],lw=2,color="gray",linestyle="--",zorder=cut_line_zorder)
    
    # Total error is just systs_err if array provided, otherwise stat + syst
    total_err = systs_err if systs_is_array else (stats_err + systs_err)

    ax.set_xlabel('_'.join(var)) if xlabel == "" else ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")      if ylabel == "" else ax.set_ylabel(ylabel)
    ax.set_title ('_'.join(var)) if title  == "" else ax.set_title (title)
    #annotate_internal(ax)
    
    if bin_labels is not None:
        ax.set_xticks(bins)
        ax.set_xticklabels(bin_labels)
    
    # Apply legend with custom kwargs
    default_legend_kwargs = {'ncol': 2, 'loc': 'upper right'}
    if legend_kwargs:
        default_legend_kwargs.update(legend_kwargs)
    legend = ax.legend(**default_legend_kwargs)
    legend.set_zorder(cut_line_zorder + 1)

    return bins, steps, total_err, syst_dict

def plot_var_pdg(**args):
    """Backward-compatible wrapper for plotting by PDG.

    Parameters
    ----------
    **args : dict
        All keyword arguments are forwarded to :func:`plot_var`. Key arguments are
        documented there; this wrapper simply calls ``plot_var(pdg=True, **args)``.

    Returns
    -------
    tuple
        The same (bins, steps, total_err) tuple returned by :func:`plot_var`.
    """
    return plot_var(pdg=True,**args)

def data_plot_overlay(df: pd.DataFrame,
                      var: str | tuple,
                      bins: list[float] | np.ndarray,
                      ax = None,
                      normalize: bool = False,
                      overflow: bool = True) -> tuple[np.ndarray, np.ndarray, object]:
    """Overlay data as points with Poisson errors on an axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to plot. ``var`` must be a column name or
        a tuple for MultiIndex columns.
    var : str | tuple
        Column to histogram.
    bins : array-like
        Bin edges for the histogram.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None the current axis is used.
    normalize : bool, default False
        If True, normalize the histogram by its integral (uses bin widths).

    Returns
    -------
    hist, errors, plot
        - hist: per-bin counts (or normalized values)
        - errors: per-bin sqrt(hist) (or normalized errors)
        - plot: the Artist returned by ax.errorbar
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(df, pd.DataFrame):
        df = ensure_lexsorted(df, axis=0)
        df = ensure_lexsorted(df, axis=1)

    hist = get_hist1d(data=df[var], bins=bins, overflow=overflow)
    errors = np.sqrt(hist)
    bin_widths = np.diff(bins)

    label = "data" 
    label += f" ({np.sum(hist,dtype=int):,})" if np.sum(hist) < 1e6 else f"({np.sum(hist):.2e})"
    
    if normalize:
        total_integral = np.sum(hist * bin_widths)
        hist = hist / total_integral
        errors = errors / total_integral
    
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    plot = ax.errorbar(bin_centers, hist, yerr=errors, fmt='.',color='black',zorder=1e3,label=label)
    return hist, errors, plot

def plot_mc_data(mc_df: pd.DataFrame,
                 data_df: pd.DataFrame,
                 var: str | tuple,
                 bins: list[float] | np.ndarray,
                 bin_labels: list[str] | None = None,
                 figsize: tuple[int, int] = (7, 6),
                 ratio_min: float = 0.0,
                 ratio_max: float = 2.0,
                 savefig: str = "",
                 **kwargs) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a combined MC stack + data overlay plot with data/MC ratio subplot.

    Parameters
    ----------
    mc_df : pandas.DataFrame
        MC dataframe to be stacked.
    data_df : pandas.DataFrame
        Dataframe containing observed data to overlay as points with errors.
    var : str | tuple
        Column (or multi-index tuple) to histogram.
    bins : array-like
        Bin edges for the histograms.
    figsize : tuple, default (7, 6)
        Figure size.
    ratio_min, ratio_max : float, default (0.0, 2.0)
        y-limits for the ratio subplot.
    savefig : str, optional
        If provided, path where the figure will be saved (bbox_inches='tight').
    **kwargs
        All other arguments (scale, pdg, pdg_col, xlabel, ylabel, title, counts, normalize,
        systs, hatch, etc.) are forwarded to :func:`plot_var`.

    Returns
    -------
    fig, ax_main, ax_sub
        The created matplotlib Figure and the main and ratio Axes.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.4)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])

    data_args = dict(df=data_df, var=var, bins=bins, ax=ax_main, normalize=kwargs.get('normalize', False), overflow=kwargs.get('overflow',True))
    mc_args   = dict(df=mc_df, var=var, bins=bins, ax=ax_main, **kwargs)

    data_hist, data_err, data_plot = data_plot_overlay(**data_args)
    mc_bins, mc_steps, mc_err, mc_dict = plot_var(**mc_args)
    
    xmin, xmax = ax_main.get_xlim()
    
    # plot the ratio
    mc_tot = mc_steps[-1][1:]  # last step contains the total MC counts

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in divide")
        # ratio is (data bin content) / (mc bin content)
        ratio = data_hist / mc_tot
        # error in ratio is just (data error) / (mc bin content)
        ratio_err = data_err / mc_tot
        # error in shading should just be (mc error) / (mc bin content)
        mc_contribution = mc_err/mc_tot
        # shading is around unity    
        ps_err = 1 + np.append(mc_contribution[0],mc_contribution)
        ms_err = 1 - np.append(mc_contribution[0],mc_contribution)
        
    bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])
    nbins = len(bins)-1
    
    ax_sub.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='s', markersize=3,color='black', zorder=1e3, label='data/MC ratio')
    # fill_between needs last entry to be repeated 
    ax_sub.fill_between(mc_bins,ms_err, ps_err, step="pre", color=mpl.colors.to_rgba("gray", alpha=0.4), lw=0.0, label='MC err.')
    
    ax_sub.axhline(1, color='red', linestyle='--', linewidth=1, zorder=0,label="y=1.0")
    ax_sub.set_xlim(xmin, xmax)
    ax_sub.set_ylim(ratio_min, ratio_max)
    ax_sub.set_ylabel("Data/MC")
    ax_sub.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                  ncol=3,fontsize='small',frameon=False)
    cut_val = kwargs.get('cut_val', None)
    if cut_val is not None:
        for cut in cut_val:
            # ax_main.axvline(cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
            ax_sub.axvline (cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
    
    if savefig!="":
        plt.savefig(savefig,bbox_inches='tight')
    
    return fig, ax_main, ax_sub, mc_dict


def plot_mc_hnl_data(mc_df: pd.DataFrame,
                 hnl_df: pd.DataFrame,
                 data_df: pd.DataFrame,
                 var: str | tuple,
                 bins: list[float] | np.ndarray,
                 figsize: tuple[int, int] = (7, 6),
                 ratio_min: float = 0.0,
                 ratio_max: float = 2.0,
                 savefig: str = "",
                 scale_nu: float = 1.0,
                 scale_hnl: float = 1.0,
                 bin_labels: list[str] | None = None,
                 **kwargs) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a combined MC stack filled histogram + HNL step histogram + data overlay plot with data/MC ratio subplot.

    Parameters
    ----------
    mc_df : pandas.DataFrame
        MC dataframe to be stacked.
    hnl_df : pandas.DataFrame
        HNL dataframe to be overlaid as a step histogram.
    data_df : pandas.DataFrame
        Dataframe containing observed data to overlay as points with errors.
    var : str | tuple
        Column (or multi-index tuple) to histogram.
    bins : array-like
        Bin edges for the histograms.
    figsize : tuple, default (7, 6)
        Figure size.
    ratio_min, ratio_max : float, default (0.0, 2.0)
        y-limits for the ratio subplot.
    savefig : str, optional
        If provided, path where the figure will be saved (bbox_inches='tight').
    scale_nu : float, default 1.0
        Scale factor for neutrino MC.
    scale_hnl : float, default 1.0
        Scale factor for HNL MC.
    **kwargs
        All other arguments (scale, pdg, pdg_col, xlabel, ylabel, title, counts, normalize,
        systs, hatch, etc.) are forwarded to :func:`plot_var`.

    Returns
    -------
    fig, ax_main, ax_sub
        The created matplotlib Figure and the main and ratio Axes.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.4)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[1])

    data_args = dict(df=data_df, var=var, bins=bins, ax=ax_main, normalize=kwargs.get('normalize', False), overflow=kwargs.get('overflow',True))
    mc_args   = dict(df=mc_df, var=var, bins=bins, ax=ax_main, hist_filled=True, error_legend=False, scale = scale_nu, **kwargs)
    hnl_args = dict(df=hnl_df, var=var, bins=bins, ax=ax_main, hist_filled=False, error_legend=True, scale = scale_hnl, **kwargs)

    data_hist, data_err, data_plot = data_plot_overlay(**data_args)
    mc_bins, mc_steps, mc_err, mc_dict = plot_var(**mc_args)
    hnl_bins, hnl_steps, hnl_err, hnl_dict = plot_var(**hnl_args)

    xmin, xmax = ax_main.get_xlim()
    
    # plot the ratio
    mc_tot = mc_steps[-1][1:]  # last step contains the total MC counts

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in divide")
        # ratio is (data bin content) / (mc bin content)
        ratio = data_hist / mc_tot
        # error in ratio is just (data error) / (mc bin content)
        ratio_err = data_err / mc_tot
        # error in shading should just be (mc error) / (mc bin content)
        mc_contribution = mc_err/mc_tot
        # shading is around unity    
        ps_err = 1 + np.append(mc_contribution[0],mc_contribution)
        ms_err = 1 - np.append(mc_contribution[0],mc_contribution)
        
    bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])
    nbins = len(bins)-1
    
    ax_sub.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='s', markersize=3,color='black', zorder=1e3, label='data/MC ratio')
    # fill_between needs last entry to be repeated 
    ax_sub.fill_between(mc_bins,ms_err, ps_err, step="pre", color=mpl.colors.to_rgba("gray", alpha=0.4), lw=0.0, label='MC err.')
    
    ax_sub.axhline(1, color='red', linestyle='--', linewidth=1, zorder=0,label="y=1.0")
    ax_sub.set_xlim(xmin, xmax)
    ax_sub.set_ylim(ratio_min, ratio_max)
    ax_sub.set_ylabel("Data/MC")
    ax_sub.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                  ncol=3,fontsize='small',frameon=False)
    cut_val = kwargs.get('cut_val', None)
    if cut_val is not None:
        for cut in cut_val:
            # ax_main.axvline(cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
            ax_sub.axvline (cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)
    if bin_labels is not None:
        ax_main.set_xticks(bins)
        ax_main.set_xticklabels(bin_labels)
        ax_sub.set_xticks(bins)
        ax_sub.set_xticklabels(bin_labels)
    #annotate_internal(ax_main)

    if savefig!="":
        plt.savefig(savefig,bbox_inches='tight')


def plot_mc_hnl(mc_df: pd.DataFrame,
                hnl_df: pd.DataFrame,
                var: str | tuple,
                bins: list[float] | np.ndarray,
                figsize: tuple[int, int] = (7, 5),
                savefig: str = "",
                scale_nu: float = 1.0,
                scale_hnl: float = 1.0,
                log_y: bool = False,
                show_fom: bool = False,
                fom_nsigma: float = 1.0,
                **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """MC BNB stacked histogram + HNL step overlay, without data points."""
    if show_fom:
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2))
        gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
        ax  = fig.add_subplot(gs[0])
        ax_fom = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_fom  = None

    mc_args  = dict(df=mc_df,  var=var, bins=bins, ax=ax, hist_filled=True,  error_legend=False, scale=scale_nu,  **kwargs)
    hnl_args = dict(df=hnl_df, var=var, bins=bins, ax=ax, hist_filled=False, error_legend=True,  scale=scale_hnl, **kwargs)

    mc_bins, mc_steps, mc_err, mc_dict = plot_var(**mc_args)
    hnl_bins, hnl_steps, hnl_err, _   = plot_var(**hnl_args)

    cut_val = kwargs.get('cut_val', None)
    if cut_val is not None:
        for cut in cut_val:
            ax.axvline(cut, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1e2)

    if log_y:
        ax.set_yscale('log')

    if show_fom:
        S_bins = hnl_steps[-1][1:]
        B_bins = mc_steps[-1][1:]
        bin_centers = 0.5 * (np.asarray(bins)[1:] + np.asarray(bins)[:-1])
        a = fom_nsigma
        S_gt = np.cumsum(S_bins[::-1])[::-1]
        B_gt = np.cumsum(B_bins[::-1])[::-1]
        S_lt = np.cumsum(S_bins)
        B_lt = np.cumsum(B_bins)
        with np.errstate(invalid='ignore', divide='ignore'):
            fom_gt = np.where(B_gt > 0, S_gt / (a / 2 + np.sqrt(B_gt)), 0)
            fom_lt = np.where(B_lt > 0, S_lt / (a / 2 + np.sqrt(B_lt)), 0)
        best_gt = bin_centers[np.argmax(fom_gt)]
        best_lt = bin_centers[np.argmax(fom_lt)]
        ax_fom.plot(bin_centers, fom_gt, color='steelblue',  label=f'keep > x  (best={best_gt:.3g})')
        ax_fom.plot(bin_centers, fom_lt, color='darkorange', label=f'keep < x  (best={best_lt:.3g})')
        ax_fom.axvline(best_gt, color='steelblue',  linestyle='--', linewidth=1, alpha=0.7)
        ax_fom.axvline(best_lt, color='darkorange', linestyle='--', linewidth=1, alpha=0.7)
        ax_fom.set_ylabel(f'Punzi FOM\n(a={a:.0f})')
        ax_fom.set_xlim(ax.get_xlim())
        ax_fom.legend(fontsize='small', frameon=False)
        ax_fom.set_xlabel(ax.get_xlabel() or str(var))
        ax.set_xlabel('')
        if cut_val is not None:
            for cut in cut_val:
                ax_fom.axvline(cut, color='black', linestyle='--', linewidth=2, alpha=0.5)

    if savefig != "":
        plt.savefig(savefig, bbox_inches='tight')

    if show_fom:
        return fig, ax, ax_fom
    return fig, ax

    return fig, ax_main, ax_sub, mc_dict, hnl_dict