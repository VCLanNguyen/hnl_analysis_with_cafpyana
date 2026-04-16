"""PFP filtering, topology counting, and slice-level DataFrame construction."""

import numpy as np
import pandas as pd

from .utils import ensure_lexsorted

# Slice-level index columns
_SLC_LEVELS = ['__ntuple', 'entry', 'rec.slc..index']

# PFP column keys (6-tuple MultiIndex)
_C_SCORE   = ('pfp', 'trackScore',       '', '', '', '')
_C_SHW_E   = ('pfp', 'shw', 'bestplane_energy', '', '', '')
_C_SHW_SX  = ('pfp', 'shw', 'start', 'x', '', '')
_C_SHW_SY  = ('pfp', 'shw', 'start', 'y', '', '')
_C_SHW_SZ  = ('pfp', 'shw', 'start', 'z', '', '')
_C_TRK_LEN = ('pfp', 'trk', 'len',       '', '', '')
_C_TRK_SX  = ('pfp', 'trk', 'start', 'x', '', '')
_C_TRK_SY  = ('pfp', 'trk', 'start', 'y', '', '')
_C_TRK_SZ  = ('pfp', 'trk', 'start', 'z', '', '')


def filter_valid_pfp(df, sentinel_vals=(-999, -5), verbose=True):
    """Remove PFPs that carry sentinel values in any pfp-level variable.

    All ``('pfp', ...)`` columns are checked **except** plane-specific ones
    (``('pfp', 'shw', 'plane', ...)``), which can legitimately hold -999
    for inactive wire planes.

    Also requires track length > 0 and shower bestplane_energy > 0 when
    those columns are present.

    This filter should run *before* shower/track classification so that
    only genuinely reconstructed PFPs enter the topology counting.

    Parameters
    ----------
    df : pd.DataFrame
        PFP-level DataFrame with 6-tuple MultiIndex columns.
    sentinel_vals : tuple of float, default (-999, -5)
        Values treated as missing/invalid.
    verbose : bool, default True
        Print how many PFPs are removed and why.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with bad PFPs removed.
    """
    check_cols = [c for c in df.columns if c[0] == 'pfp' and c[2] != 'plane']

    bad = pd.Series(False, index=df.index)
    col_hits = {}
    for col in check_cols:
        col_bad = pd.Series(False, index=df.index)
        for val in sentinel_vals:
            col_bad |= (df[col] == val)
        if col_bad.any():
            col_hits[col] = int(col_bad.sum())
        bad |= col_bad

    # Require trk.len > 0 and shw.bestplane_energy > 0
    if _C_TRK_LEN in df.columns:
        bad_trk = df[_C_TRK_LEN] <= 0
        if bad_trk.any():
            col_hits[_C_TRK_LEN] = col_hits.get(_C_TRK_LEN, 0) + int(bad_trk.sum())
        bad |= bad_trk
    if _C_SHW_E in df.columns:
        bad_shw = df[_C_SHW_E] <= 0
        if bad_shw.any():
            col_hits[_C_SHW_E] = col_hits.get(_C_SHW_E, 0) + int(bad_shw.sum())
        bad |= bad_shw

    n_bad = bad.sum()
    if verbose:
        if n_bad == 0:
            print("filter_valid_pfp: no sentinel values found — all PFPs kept")
        else:
            print(f"filter_valid_pfp: removed {n_bad} / {len(df)} PFPs "
                  f"({n_bad / len(df) * 100:.1f}%) with sentinel values {sentinel_vals}")
            for col, n in sorted(col_hits.items(), key=lambda x: -x[1]):
                print(f"  {col}  →  {n} ({n / len(df) * 100:.1f}%)")

    return df[~bad].copy()


def count_topo(df, trkscore_cut: float = 0.51, filter_sentinel: bool = True) -> pd.DataFrame:
    """Count showers and tracks per slice and add topology columns.

    PFPs are optionally cleaned via :func:`filter_valid_pfp` first.

    A PFP is classified as a **shower** if:
      - 0 < trackScore < trkscore_cut

    A PFP is classified as a **track** if:
      - trkscore_cut <= trackScore < 1

    Parameters
    ----------
    df : pd.DataFrame
        PFP-level DataFrame with MultiIndex columns and index levels
        ['__ntuple', 'entry', 'rec.slc..index', 'rec.slc.reco.pfp..index'].
    trkscore_cut : float, default 0.51
        trackScore threshold separating showers from tracks.
    filter_sentinel : bool, default True
        If True, call :func:`filter_valid_pfp` before classifying PFPs.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with four new columns added:
          ('slc', 'n_shws', '', '', '', '')
          ('slc', 'n_trks', '', '', '', '')
          ('pfp', 'is_shw', '', '', '', '')
          ('pfp', 'is_trk', '', '', '', '')
    """
    if filter_sentinel:
        df = filter_valid_pfp(df)
    else:
        df = df.copy()

    trkscore = df[_C_SCORE]
    is_shw = (trkscore > 0) & (trkscore < trkscore_cut)
    is_trk = (trkscore >= trkscore_cut) & (trkscore < 1)

    flat = df.reset_index()
    topo = (
        pd.DataFrame({
            '__ntuple':       flat['__ntuple'].values,
            'entry':          flat['entry'].values,
            'rec.slc..index': flat['rec.slc..index'].values,
            'is_shw':         is_shw.values,
            'is_trk':         is_trk.values,
        })
        .groupby(_SLC_LEVELS)
        .agg(n_shws=('is_shw', 'sum'), n_trks=('is_trk', 'sum'))
    )

    idx = df.index.to_frame(index=False)[_SLC_LEVELS]
    merged = idx.merge(topo.reset_index(), on=_SLC_LEVELS, how='left')
    merged.index = df.index

    df[('slc', 'n_shws', '', '', '', '')] = merged['n_shws'].values
    df[('slc', 'n_trks', '', '', '', '')] = merged['n_trks'].values
    df[('pfp', 'is_shw', '', '', '', '')] = is_shw.values
    df[('pfp', 'is_trk', '', '', '', '')] = is_trk.values

    return df


def _pick_pfp_rank(pfp_df, sort_col, rank, ascending=False):
    """Return one PFP row per slice at the given within-slice rank.

    Parameters
    ----------
    pfp_df : pd.DataFrame
        PFP-level subset (already filtered to the desired PFP type).
    sort_col : tuple
        MultiIndex column key used for ranking (e.g. energy or length).
    rank : int
        0-based rank to select (0 = leading, 1 = subleading, ...).
    ascending : bool
        Sort direction. Default False (highest value = rank 0).

    Returns
    -------
    pd.DataFrame
        One row per slice at the requested rank; slices with fewer PFPs than
        rank+1 are silently dropped.
    """
    idx = pfp_df.index.to_frame(index=False)
    idx_cols = list(idx.columns)
    idx['_val'] = pfp_df[sort_col].values

    asc_flags = [True] * len(_SLC_LEVELS) + [ascending]
    idx_sorted = idx.sort_values(_SLC_LEVELS + ['_val'], ascending=asc_flags)
    idx_sorted['_rank'] = idx_sorted.groupby(_SLC_LEVELS).cumcount()

    sel = idx_sorted[idx_sorted['_rank'] == rank][idx_cols]
    mi = pd.MultiIndex.from_frame(sel)
    return pfp_df.loc[mi].copy()


def _rename_branch(df, old='pfp', new='primshw'):
    """Rename the first column-level from *old* to *new* for matching columns."""
    new_cols = pd.MultiIndex.from_tuples(
        [(new if c[0] == old else c[0],) + c[1:] for c in df.columns],
        names=df.columns.names,
    )
    return df.set_axis(new_cols, axis=1)


def make_topo_df(df, n_shws=None, n_trks=None):
    """Collapse a PFP-level DataFrame to slice level with named PFP branches.

    Expects a df already processed by :func:`count_topo`, which adds the
    ``is_shw``/``is_trk`` per-PFP flags and ``n_shws``/``n_trks`` per-slice
    counts.

    Showers are ranked by ``bestplane_energy`` (descending): the
    highest-energy shower becomes ``primshw``, the second becomes ``secshw``.
    Tracks are ranked by ``trk.len`` (descending): the longest becomes
    ``primtrk``.

    Parameters
    ----------
    df : pd.DataFrame
        PFP-level DataFrame already processed by :func:`count_topo`.
    n_shws : int or None, default None
        Keep only slices with exactly this many showers. None = all.
    n_trks : int or None, default None
        Keep only slices with exactly this many tracks. None = all.

    Returns
    -------
    pd.DataFrame
        Slice-level DataFrame with columns under ``primshw``, ``secshw``,
        ``primtrk``, and ``slc``.
    """
    _C_IS_SHW = ('pfp', 'is_shw', '', '', '', '')
    _C_IS_TRK = ('pfp', 'is_trk', '', '', '', '')
    _C_N_SHWS = ('slc', 'n_shws', '', '', '', '')
    _C_N_TRKS = ('slc', 'n_trks', '', '', '', '')

    mask = pd.Series(True, index=df.index)
    if n_shws is not None:
        mask &= (df[_C_N_SHWS] == n_shws)
    if n_trks is not None:
        mask &= (df[_C_N_TRKS] == n_trks)
    df = df[mask]

    shw_df = df[df[_C_IS_SHW].astype(bool)]
    trk_df = df[df[_C_IS_TRK].astype(bool)]

    lead_shw    = _pick_pfp_rank(shw_df, _C_SHW_E,   rank=0)
    sublead_shw = _pick_pfp_rank(shw_df, _C_SHW_E,   rank=1)
    lead_trk    = _pick_pfp_rank(trk_df, _C_TRK_LEN, rank=0)

    def _drop_pfp_idx(sub):
        n_extra = sub.index.nlevels - len(_SLC_LEVELS)
        if n_extra > 0:
            return sub.droplevel(list(range(len(_SLC_LEVELS), sub.index.nlevels)))
        return sub

    lead_shw    = _drop_pfp_idx(lead_shw)
    sublead_shw = _drop_pfp_idx(sublead_shw)
    lead_trk    = _drop_pfp_idx(lead_trk)

    pfp_cols = [c for c in df.columns if c[0] == 'pfp']
    lead_shw    = _rename_branch(lead_shw[pfp_cols],    old='pfp', new='primshw')
    sublead_shw = _rename_branch(sublead_shw[pfp_cols], old='pfp', new='secshw')
    lead_trk    = _rename_branch(lead_trk[pfp_cols],    old='pfp', new='primtrk')

    non_pfp_cols = [c for c in df.columns
                    if not (isinstance(c, tuple) and c[0] == 'pfp')]
    slc_df = df[non_pfp_cols].groupby(level=_SLC_LEVELS).first()

    result = slc_df.join(lead_shw,    how='left')
    result = result.join(sublead_shw, how='left')
    result = result.join(lead_trk,    how='left')

    return result.reset_index(drop=True)
