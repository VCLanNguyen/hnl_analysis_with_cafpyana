"""Generic cut infrastructure for nue CC selection.

:class:`~nueana.classes.CutSpec` (defined in :mod:`nueana.classes`) and the
functions to build, modify, and apply cut sequences live here.

The analysis-specific cut sequences (:data:`~nueana.analysis.DEFAULT_CUTS`,
:data:`~nueana.analysis.SIDEBAND_CUTS`) and signal categorisation live in
:mod:`nueana.analysis`.

**Tighten or loosen a cut**::

    from nueana import modify_cut, select, DEFAULT_CUTS

    cuts = modify_cut(DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
    df_sel = select(df, cuts=cuts)

**Drop a cut**::

    from nueana import DEFAULT_CUTS, drop_cuts

    cuts = drop_cuts(DEFAULT_CUTS, "muon_rejection")
    cuts = drop_cuts(DEFAULT_CUTS, "direction", "shower_length")

**Add a custom cut**::

    from nueana.classes import CutSpec
    cuts = DEFAULT_CUTS + [CutSpec("my_cut", fn=lambda df: df.x > 10)]

**Adjust a parameter of an fn-based cut** (combine with ``functools.partial``)::

    from functools import partial
    cuts = modify_cut(DEFAULT_CUTS, "muon_rejection",
                      fn=partial(cut_muon_rejection, max_track_length=100))
"""

import warnings

import numpy as np
import pandas as pd
from dataclasses import replace
from functools import reduce

from makedf.util import InFV, InAV
from .classes import CutSpec
from .utils import ensure_lexsorted


__all__ = ['drop_cuts', 'modify_cut', 'select', 'select_sideband', 'union_cut',
           'define_signal_pi0', 'define_signal_hnl']


# ---------------------------------------------------------------------------
# Cut application
# ---------------------------------------------------------------------------

def _mask(df, spec):
    """Return a boolean mask for spec applied to df."""
    if spec.fn is not None:
        return spec.fn(df)
    series = spec.accessor(df) if spec.accessor is not None else reduce(getattr, spec.variable, df)
    return (series > spec.min) & (series < spec.max)


def _and_mask(df, cuts):
    """AND a list of CutSpec masks, all evaluated against the same (unnarrowed) df."""
    mask = pd.Series(True, index=df.index)
    for spec in cuts:
        mask &= _mask(df, spec)
    return mask


def union_cut(name, cuts_a, cuts_b, label=None):
    """Build a single CutSpec that is the OR of two CutSpec chains.

    Lets a selection like "(1shw AND ...) OR (2shw AND ...)" be expressed as one
    CutSpec, so it folds into select()'s ordinary sequential-AND cuts= list (and
    therefore into load_mc/load_data/load_mchnl's load-time cuts= too) instead of
    needing a separate post-load union step.
    """
    return CutSpec(name, fn=lambda df: _and_mask(df, cuts_a) | _and_mask(df, cuts_b),
                    label=label or name)


def drop_cuts(cuts, *names):
    """Return a copy of cuts with the named cut(s) removed.

    Parameters
    ----------
    cuts : list of CutSpec
        The cut sequence to modify.
    *names : str
        Names of cuts to drop.

    Examples
    --------
    >>> cuts = drop_cuts(DEFAULT_CUTS, "muon_rejection")
    >>> cuts = drop_cuts(DEFAULT_CUTS, "direction", "shower_length")
    """
    unknown = set(names) - {c.name for c in cuts}
    if unknown:
        raise ValueError(f"No cuts named {sorted(unknown)}. Available: {[c.name for c in cuts]}")
    return [c for c in cuts if c.name not in names]


def modify_cut(cuts, name, **kwargs):
    """Return a copy of cuts with the named CutSpec updated.

    Parameters
    ----------
    cuts : list of CutSpec
        The cut sequence to modify.
    name : str
        Name of the cut to update.
    **kwargs
        Fields to update on the matching CutSpec (passed to
        ``dataclasses.replace``).

    Returns
    -------
    list of CutSpec
        New list with the named entry replaced.

    Raises
    ------
    ValueError
        If no cut with the given name exists.

    Examples
    --------
    >>> cuts = modify_cut(DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
    >>> cuts = modify_cut(cuts, "muon_rejection",
    ...                   fn=partial(cut_muon_rejection, max_track_length=100))
    """
    names = [c.name for c in cuts]
    if name not in names:
        raise ValueError(f"No cut named '{name}'. Available: {names}")
    return [replace(c, **kwargs) if c.name == name else c for c in cuts]


# ---------------------------------------------------------------------------
# Selection pipeline
# ---------------------------------------------------------------------------

def select(indf,
           cuts=None,
           stage=None,
           savedict=False,
           check_preprocessed=True):
    """Apply a sequence of cuts to a DataFrame.

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame. Must have ``primshw.shw.reco_energy`` set — call
        :func:`~nueana.preprocess.preprocess_mc` or
        :func:`~nueana.preprocess.preprocess_data` first.
    cuts : list of CutSpec, optional
        Ordered cut sequence. Defaults to ``DEFAULT_CUTS`` from
        :mod:`nueana.cuts`. Use :func:`modify_cut` to adjust individual cuts,
        or build a list from scratch using :class:`~nueana.classes.CutSpec`.
    stage : str, optional
        Stop and return after this cut (matched by ``CutSpec.name``).
    savedict : bool, default False
        If True, return a dict of DataFrames keyed by cut name instead
        of the final DataFrame.
    check_preprocessed : bool, default True
        If True, warn when no preprocessing fixes (``_fix_*`` columns)
        are detected on ``indf``. Suppress with ``check_preprocessed=False``
        for DataFrames where preprocessing is not applicable.

    Returns
    -------
    pandas.DataFrame or dict
        Final selected DataFrame, or per-stage dict when
        ``savedict=True`` or ``stage`` is set.
    """
    if cuts is None:
        from .analysis import DEFAULT_CUTS
        cuts = DEFAULT_CUTS

    if check_preprocessed:
        from .preprocess import applied_fixes
        if not applied_fixes(indf):
            warnings.warn(
                "No preprocessing fixes detected on this DataFrame. "
                "Call preprocess_mc() or preprocess_data() before select().",
                stacklevel=2,
            )

    if stage is not None and stage not in {c.name for c in cuts}:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid options: {[c.name for c in cuts]}"
        )

    df = indf.copy()

    df_dict = {}
    for spec in cuts:
        df = df[_mask(df, spec)]
        if savedict:
            df_dict[spec.name] = df
        if stage == spec.name:
            return df_dict if savedict else df

    return df_dict if savedict else df

def define_signal_pi0(indf: pd.DataFrame, prefix=None):
    """Define signal/background categories for the generic pi0 analysis.

    Categorizes events into CCpi0 signal and various background categories
    based on truth information and fiducial volume, using the HNL/pi0 signal
    scheme (:data:`~nueana.analysis.signal_dict_hnl`).

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'signal' column indicating event category using
        ``signal_dict_hnl``.
    """
    from .analysis import signal_dict_hnl

    # Keep lexsorted axes for robust multi-index access without forcing a full copy.
    nudf = ensure_lexsorted(ensure_lexsorted(indf, 0), 1)

    if prefix is None:
        mcdf = nudf
    else:
        mcdf = nudf[prefix]

    whereFV = InFV(mcdf.position,det="SBND",inzback=0)
    whereAV = InAV(df=mcdf.position)
    whereCCpi0 = ((mcdf.iscc==1)  # require CC interaction
                & (abs(mcdf.pdg)==14)  # require neutrino to be a nue
                & (mcdf.npi0>0) # require at least one pi0 in the final state
                )

    if "signal" in nudf.columns:
        signal = nudf["signal"].to_numpy(copy=True)
    else:
        signal = np.full(len(nudf), -1, dtype=np.int16)

    # background
    signal[whereFV & (mcdf.iscc==0) & (mcdf.npi0 > 0)] = signal_dict_hnl["NCpi0"] # nc pi0 FV
    signal[whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==14) & (mcdf.npi0 == 0)] = signal_dict_hnl["othernumuCC"] # numu cc other FV
    signal[whereFV & (mcdf.iscc==0) & (mcdf.npi0 == 0)] = signal_dict_hnl["otherNC"] # nc other FV
    signal[whereFV & (mcdf.iscc==1) & (abs(mcdf.pdg)==12)] = signal_dict_hnl["CCnue"] # nue cc FV
    signal[whereAV & (signal < 0)] = signal_dict_hnl["nonFV"] # nonFV
    signal[whereAV == False] = signal_dict_hnl["dirt"] # dirt
    signal[np.isnan(mcdf.E)] = signal_dict_hnl['cosmic']

    signal[whereFV & whereCCpi0] = signal_dict_hnl["CCpi0"]
    nudf["signal"] = signal
    if ((nudf.signal < 0) | (nudf.signal >= len(signal_dict_hnl))).any():
        print("Warning: unidentified signal/bacgkr channels present.")
    return nudf


def define_signal_hnl(indf):
    """Define signal for the HNL sample: everything (HNL and HNL cosmic) is signal."""
    from .analysis import signal_dict_hnl

    signal_col = ('signal', '', '', '', '', '')
    # HNL cosmic (mask via slc.prtl.E == NaN) is deliberately labeled HNL too, not
    # signal_dict_hnl['hnlcosmic'] -- both are part of the signal region, so every row
    # gets the same value regardless of the cosmic mask (no need to branch on it).
    indf[signal_col] = signal_dict_hnl['hnl']

    return indf


def select_sideband(indf, cuts=None, **kwargs):
    """Apply the sideband cut sequence. Accepts the same kwargs as ``select``."""
    if cuts is None:
        from .analysis import SIDEBAND_CUTS
        cuts = SIDEBAND_CUTS
    return select(indf, cuts=cuts, **kwargs)
