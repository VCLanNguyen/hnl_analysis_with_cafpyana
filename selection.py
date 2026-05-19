"""nue CC selection.

Each cut is a :class:`CutSpec`. The full sequence is a plain list, so
customising is straightforward with :func:`modify_cut`:

**Tighten or loosen a cut**::

    from nueana.selection import DEFAULT_CUTS, modify_cut, select

    cuts = modify_cut(DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
    df_sel = select(df, cuts=cuts)

**Drop a cut**::

    cuts = drop_cuts(DEFAULT_CUTS, "muon_rejection")
    cuts = drop_cuts(DEFAULT_CUTS, "direction", "shower_length")  # drop multiple

**Add a custom cut**::

    from nueana.selection import CutSpec
    cuts = DEFAULT_CUTS + [CutSpec("my_cut", fn=lambda df: df.x > 10)]

**Adjust a parameter of an fn-based cut** (combine with ``functools.partial``)::

    from functools import partial
    cuts = modify_cut(DEFAULT_CUTS, "muon_rejection",
                      fn=partial(cut_muon_rejection, max_track_length=100))
"""

import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from functools import partial, reduce
from typing import Callable, Optional

from . import config
from makedf.util import *
from pyanalib.pandas_helpers import *

from .utils import ensure_lexsorted
from .constants import signal_dict, generic_dict
from .geometry import whereTPC


# ---------------------------------------------------------------------------
# CutSpec: declarative description of a single selection cut.
# ---------------------------------------------------------------------------

@dataclass
class CutSpec:
    """Declarative description of a single selection cut.

    Exactly one of ``variable``, ``accessor``, or ``fn`` must be set.

    Parameters
    ----------
    name : str
        Unique identifier used for stage stopping and savedict keys.
    variable : tuple, optional
        MultiIndex key resolved via getattr chaining, e.g.
        ``("primshw", "shw", "len")``. Cut passes when
        ``min < df.<variable> < max``.
    min, max : float
        Lower and upper bounds for variable/accessor cuts.
        Default to ``-inf`` / ``+inf`` (i.e. open-ended).
    accessor : callable, optional
        ``lambda df: <Series>`` — use when column access is more
        complex than a simple tuple key. Cut passes when
        ``min < accessor(df) < max``.
    fn : callable, optional
        ``fn(df) -> bool mask`` — full override for cuts that are not
        a simple min/max comparison. Takes precedence over
        ``variable`` and ``accessor``.
    label : str, optional
        Human-readable description, e.g. for cut-flow tables and plots.
        Defaults to ``name`` if not set.
    """
    name: str
    variable: tuple = None
    min: float = -np.inf
    max: float = np.inf
    accessor: Callable = None
    fn: Callable = None
    label: str = None

    def __post_init__(self):
        if self.fn is None and self.accessor is None and self.variable is None:
            raise ValueError(
                f"CutSpec '{self.name}': at least one of variable, accessor, or fn must be set."
            )
        if self.label is None:
            self.label = self.name


def _mask(df, spec):
    """Return a boolean mask for spec applied to df."""
    if spec.fn is not None:
        return spec.fn(df)
    series = spec.accessor(df) if spec.accessor is not None else reduce(getattr, spec.variable, df)
    return (series > spec.min) & (series < spec.max)


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
# Cut functions for cuts that are not simple min/max comparisons.
# Use these directly for fine-grained control (e.g. in detvar loops),
# or pass them as fn= in a CutSpec.
# ---------------------------------------------------------------------------

def cut_muon_rejection(df, max_track_length=200):
    return np.isnan(df.primtrk.trk.len) | (df.primtrk.trk.len < max_track_length)


def InSpill(df, spill_start=0.335, spill_end=0.335 + 1.6):
    return (
        (df.slc.barycenterFM.flashTime > spill_start)
        & (df.slc.barycenterFM.flashTime < spill_end)
    )


def InScore(df, score_cut=0.02):
    return df.slc.barycenterFM.score > score_cut


# ---------------------------------------------------------------------------
# Default cut sequence
# ---------------------------------------------------------------------------

DEFAULT_CUTS = [
    CutSpec("flash_pe",        variable=("slc", "barycenterFM", "flashPEs"),  min=2e3,   label="flash PE > 2000"),
    CutSpec("nu_score",        variable=("slc", "nu_score"),                  min=0.5,   label="nu score > 0.5"),
    CutSpec("clear_cosmic",    fn=lambda df: df.slc.is_clear_cosmic == 0,                label="clear cosmic"),
    CutSpec("fiducial_volume", fn=lambda df: InFV(df.slc.vertex, det="SBND_nu26", inzback=0), label="fiducial volume"),
    CutSpec("flash_time",      variable=("slc", "barycenterFM", "flashTime"), min=0.335, max=1.935, label="flash time [0.335, 1.935] µs"),
    CutSpec("flash_score",     variable=("slc", "barycenterFM", "score"),     min=0.02,             label="flash score > 0.02"),
    CutSpec("shower_energy",   variable=("primshw", "shw", "reco_energy"),    min=0.5,              label="shower energy > 0.5 GeV"),
    CutSpec("muon_rejection",  fn=cut_muon_rejection,                                              label="track length < 200 cm"),
    CutSpec("conversion_gap",  variable=("primshw", "shw", "conversion_gap"), min=0.001, max=2,    label="conversion gap [0.001, 2] cm"),
    CutSpec("dedx",            variable=("primshw", "shw", "bestplane_dEdx"), min=1.25,  max=2.5,  label="dE/dx [1.25, 2.5] MeV/cm"),
    CutSpec("opening_angle",   variable=("primshw", "shw", "open_angle"),     min=0.03,  max=0.15, label="opening angle [0.03, 0.15] rad"),
    CutSpec("shower_length",   variable=("primshw", "shw", "len"),            min=10,    max=200,  label="shower length [10, 200] cm"),
    ]

# Sideband: built from DEFAULT_CUTS by overriding the cuts that differ.
SIDEBAND_CUTS = DEFAULT_CUTS.copy()
SIDEBAND_CUTS = drop_cuts(SIDEBAND_CUTS, "muon_rejection")
SIDEBAND_CUTS = drop_cuts(SIDEBAND_CUTS, "shower_length")
SIDEBAND_CUTS = modify_cut(SIDEBAND_CUTS,  "conversion_gap", min=2,    max=np.inf,  label="conversion gap > 2 cm")
SIDEBAND_CUTS = modify_cut(SIDEBAND_CUTS,  "dedx",           min=3,    max=6,       label="dE/dx [3, 6] MeV/cm")
SIDEBAND_CUTS = modify_cut(SIDEBAND_CUTS,  "opening_angle",  min=0,    max=1.0,     label="opening angle [0, 1.0] rad")


# ---------------------------------------------------------------------------
# Selection pipeline
# ---------------------------------------------------------------------------

def select(indf,
           cuts=None,
           stage=None,
           savedict=False,
           spring=True,
           shower_scale=1.17,
           check_preprocessed=True):
    """Apply a sequence of cuts to a DataFrame.

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame.
    cuts : list of CutSpec, optional
        Ordered cut sequence. Defaults to ``DEFAULT_CUTS``.
        Use :func:`modify_cut` to adjust individual cuts, or build a
        list from scratch using :class:`CutSpec`.
    stage : str, optional
        Stop and return after this cut (matched by ``CutSpec.name``).
    savedict : bool, default False
        If True, return a dict of DataFrames keyed by cut name instead
        of the final DataFrame.
    spring : bool, default True
        Use max-plane shower energy (True) or best-plane (False) as
        ``primshw.shw.reco_energy`` before cuts are applied.
    shower_scale : float, default 1.17
        Scale factor applied to shower energy.
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
    cuts = DEFAULT_CUTS if cuts is None else cuts

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

    energy_col = ("primshw", "shw", "reco_energy", '', '', '')
    src = df.primshw.shw.maxplane_energy if spring else df.primshw.shw.bestplane_energy
    df[energy_col] = src * shower_scale

    df_dict = {}
    for spec in cuts:
        df = df[_mask(df, spec)]
        if savedict:
            df_dict[spec.name] = df
        if stage == spec.name:
            return df_dict if savedict else df

    return df_dict if savedict else df


def select_sideband(indf, cuts=None, **kwargs):
    """Apply the sideband cut sequence. Accepts the same kwargs as ``select``."""
    return select(indf, cuts=SIDEBAND_CUTS if cuts is None else cuts, **kwargs)


# ---------------------------------------------------------------------------
# Truth categorisation
# ---------------------------------------------------------------------------

def define_signal(indf: pd.DataFrame, prefix=None):
    """Define signal/background categories for neutrino interactions.

    Categorizes events into signal (CC nue) and background categories
    based on truth information and fiducial volume.

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added ``signal`` column (values from ``signal_dict``).
    """
    nudf = ensure_lexsorted(ensure_lexsorted(indf, 0), 1)

    mcdf = nudf[prefix] if prefix is not None else nudf

    whereFV = InFV(mcdf.position, det="SBND_nu26", inzback=0)
    whereAV = InAV(df=mcdf.position)
    whereCCnue = (
        (mcdf.iscc == 1)
        & (abs(mcdf.pdg) == 12)
        & (abs(mcdf.e.pdg) == 11)
        & (mcdf.e.genE > 0.5)
    )

    if "signal" in nudf.columns:
        signal = nudf["signal"].to_numpy(copy=True)
    else:
        signal = np.full(len(nudf), -1, dtype=np.int16)

    signal[whereFV & (mcdf.iscc == 1) & (abs(mcdf.pdg) == 14) & (mcdf.npi0 > 0)]   = signal_dict["numuCCpi0"]
    signal[whereFV & (mcdf.iscc == 0) & (mcdf.npi0 > 0)]                           = signal_dict["NCpi0"]
    signal[whereFV & (mcdf.iscc == 1) & (abs(mcdf.pdg) == 12)]                     = signal_dict["othernueCC"]
    signal[whereFV & (mcdf.iscc == 1) & (abs(mcdf.pdg) == 14) & (mcdf.npi0 == 0)]  = signal_dict["othernumuCC"]
    signal[whereFV & (mcdf.iscc == 0) & (mcdf.npi0 == 0)]                          = signal_dict["otherNC"]
    signal[whereAV & (signal < 0)]                                                 = signal_dict["nonFV"]
    signal[whereAV == False]                                                       = signal_dict["dirt"]
    signal[np.isnan(mcdf.E)]                                                       = signal_dict['cosmic']
    signal[whereFV & whereCCnue]                                                   = signal_dict["nueCC"]

    nudf["signal"] = signal
    if ((nudf.signal < 0) | (nudf.signal >= len(signal_dict))).any():
        print("Warning: unidentified signal/background channels present.")
    return nudf


def define_generic(indf: pd.DataFrame, prefix=None):
    """Define broad signal/background categories (CC nu, NC nu, non-FV, dirt, cosmic).

    Parameters
    ----------
    indf : pandas.DataFrame
        Input DataFrame with MultiIndex columns containing truth information.
    prefix : str or tuple, optional
        Column prefix to access truth information. If None, uses top-level columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added ``signal`` column (values from ``generic_dict``).
    """
    indf = ensure_lexsorted(indf, 0)
    nudf = ensure_lexsorted(indf.copy(), 1)

    mcdf = nudf[prefix] if prefix is not None else nudf

    whereFV = InFV(df=mcdf.position, inzback=0, det="SBND")
    whereAV = InAV(df=mcdf.position)

    if "signal" not in nudf.columns:
        nudf["signal"] = -1

    nudf["signal"] = np.where(whereAV == False,           generic_dict["dirt"],   nudf["signal"])
    nudf["signal"] = np.where(whereAV,                    generic_dict["nonFV"],  nudf["signal"])
    nudf["signal"] = np.where(whereFV & (mcdf.iscc == 0), generic_dict["NCnu"],   nudf["signal"])
    nudf["signal"] = np.where(whereFV & (mcdf.iscc == 1), generic_dict["CCnu"],   nudf["signal"])
    nudf["signal"] = np.where(np.isnan(mcdf.E),           generic_dict["cosmic"], nudf["signal"])

    if ((nudf.signal < 0) | (nudf.signal >= len(generic_dict))).any():
        print("Warning: unidentified signal/background channels present.")
    indf["signal"] = nudf["signal"]
    return indf
