"""
Merge ar23-plus weights from a secondary .df file into an orig nuecc .df file.

The merge uses the same two-step event matching as detvar/store.py:
  Step 1 — external match on physics-level nulite columns
  Step 2 — internal match on file-level index columns (__ntuple, entry)

The ar23-only weight columns are added to both nuecc (as slc.truth.* columns,
consistent with existing truth weights) and mcnuecc (as top-level columns).
The output is a new single-split .df file.

Usage
-----
    from nueana.exclusive import merge_ar23_weights

    merge_ar23_weights(
        orig_file = 'mc_nuecc.df',
        ar23_file = 'mc_nuecc_ar23.df',
        outfile   = 'mc_nuecc_merged.df',
    )
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..io import load_dfs, get_n_split

__all__ = ['merge_ar23_weights']

_EXT_MATCH_COL = ['E', 'rec.mc.nu..index', 'run', 'subrun', 'evt']
_HDF_KW = dict(format='fixed')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_all(file: str, keys: list) -> dict:
    n = get_n_split(file)
    return load_dfs(file, keys, n_max_concat=n)


def _prep_lite(nulite_df: pd.DataFrame, ext_match_col: list) -> pd.DataFrame:
    """Reset nulite index, reindex on ext_match_col, drop ambiguous events."""
    df = nulite_df.reset_index().set_index(ext_match_col)
    return df[~df.index.duplicated(keep=False)]


def _build_mapping(
    orig_lite: pd.DataFrame,
    ar23_lite: pd.DataFrame,
    common_idx,
    ext_match_col: list,
) -> pd.DataFrame:
    """Return a mapping DataFrame with columns:
        ar23_ntuple, ar23_entry, rec.mc.nu..index, orig_ntuple, orig_entry, orig_file_idx

    Each row is unique on (ar23_ntuple, ar23_entry, rec.mc.nu..index) because
    _prep_lite already deduplicates both sides on ext_match_col (which includes
    rec.mc.nu..index).  Multiple neutrinos per event (different rec.mc.nu..index
    values with the same ntuple/entry) are preserved as separate rows.
    """
    orig_r = orig_lite.loc[common_idx].reset_index()[
        ext_match_col + ['__ntuple', 'entry', 'file_idx']
    ]
    ar23_r = ar23_lite.loc[common_idx].reset_index()[
        ext_match_col + ['__ntuple', 'entry']
    ]

    m = ar23_r.merge(orig_r, on=ext_match_col, suffixes=('_ar23', '_orig'))
    m = m.rename(columns={
        '__ntuple_ar23': 'ar23_ntuple',
        'entry_ar23':    'ar23_entry',
        '__ntuple_orig': 'orig_ntuple',
        'entry_orig':    'orig_entry',
        'file_idx':      'orig_file_idx',
    })[['ar23_ntuple', 'ar23_entry', 'rec.mc.nu..index', 'orig_ntuple', 'orig_entry', 'orig_file_idx']]

    return m.reset_index(drop=True)


def _remap_weights(ar23_weights: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Remap ar23 weight rows from ar23 internal coords to orig internal coords.

    ar23_weights has index ['__ntuple', 'entry', 'rec.mc.nu..index'] and
    MultiIndex columns.  Works by remapping the index only (never merging the
    MultiIndex-column DataFrame with the flat mapping table).

    Returns a DataFrame indexed on ['__ntuple', 'entry', 'rec.mc.nu..index', 'file_idx']
    in orig coordinate space.
    """
    # Drop cross-split duplicates in ar23 (same ntuple/entry/nu-index in multiple files)
    n_before = len(ar23_weights)
    ar23_weights = ar23_weights[~ar23_weights.index.duplicated(keep=False)]
    n_dropped = n_before - len(ar23_weights)
    if n_dropped:
        print(f"  Warning: dropped {n_dropped} ambiguous ar23 mcnuecc rows (cross-split duplicates)")

    # Flatten the ar23 index to a plain DataFrame for merging
    idx_flat = ar23_weights.index.to_frame(index=False)
    # columns: __ntuple, entry, rec.mc.nu..index

    # Rename mapping keys to match idx_flat column names
    map_renamed = mapping.rename(columns={
        'ar23_ntuple': '__ntuple',
        'ar23_entry':  'entry',
    })[['__ntuple', 'entry', 'rec.mc.nu..index', 'orig_ntuple', 'orig_entry', 'orig_file_idx']]

    idx_mapped = idx_flat.merge(map_renamed, on=['__ntuple', 'entry', 'rec.mc.nu..index'], how='inner')
    # columns: __ntuple(ar23), entry(ar23), rec.mc.nu..index, orig_ntuple, orig_entry, orig_file_idx

    # Select matching rows from ar23_weights (preserve MultiIndex columns)
    ar23_idx = pd.MultiIndex.from_frame(idx_mapped[['__ntuple', 'entry', 'rec.mc.nu..index']])
    matched  = ar23_weights.loc[ar23_idx].copy()

    # Assign new index in orig coordinate space
    new_idx = pd.MultiIndex.from_arrays(
        [
            idx_mapped['orig_ntuple'].values,
            idx_mapped['orig_entry'].values,
            idx_mapped['rec.mc.nu..index'].values,
            idx_mapped['orig_file_idx'].values,
        ],
        names=['__ntuple', 'entry', 'rec.mc.nu..index', 'file_idx'],
    )
    matched.index = new_idx
    return matched


def _join_to_mcnuecc(orig_mc: pd.DataFrame, remapped: pd.DataFrame) -> pd.DataFrame:
    """Left-join ar23 weight columns onto orig mcnuecc by its 4-level index."""
    return orig_mc.join(remapped, how='left')


def _join_to_nuecc(orig_nuecc: pd.DataFrame, remapped: pd.DataFrame) -> pd.DataFrame:
    """Left-join ar23 weight columns onto orig nuecc.

    Matches via (__ntuple, entry, file_idx) from nuecc's index and
    rec.mc.nu..index from nuecc's columns.  Weight columns are renamed
    to ('slc', 'truth', ...) to match existing truth-weight format.
    """
    n = orig_nuecc.columns.nlevels
    nu_idx_col = ('rec.mc.nu..index',) + ('',) * (n - 1)

    nu_idx_vals = orig_nuecc[nu_idx_col].values
    # NaN nu_idx → use -1 sentinel (won't match any real neutrino index)
    nu_idx_safe = np.where(pd.isna(nu_idx_vals), -1.0, nu_idx_vals.astype(float))

    join_idx = pd.MultiIndex.from_arrays(
        [
            orig_nuecc.index.get_level_values('__ntuple'),
            orig_nuecc.index.get_level_values('entry'),
            orig_nuecc.index.get_level_values('file_idx'),
            nu_idx_safe,
        ],
        names=['__ntuple', 'entry', 'file_idx', 'rec.mc.nu..index'],
    )

    wgts_reordered = remapped.reorder_levels(['__ntuple', 'entry', 'file_idx', 'rec.mc.nu..index'])
    wgt_values = wgts_reordered.reindex(join_idx)
    wgt_values.index = orig_nuecc.index

    # Rename weight columns: 3-level tuple → n-level with ('slc', 'truth', ...) prefix
    new_cols = []
    for c in wgt_values.columns:
        new_c = ('slc', 'truth') + c
        if len(new_c) < n:
            new_c = new_c + ('',) * (n - len(new_c))
        new_cols.append(new_c)
    wgt_values.columns = pd.MultiIndex.from_tuples(new_cols)

    return pd.concat([orig_nuecc, wgt_values], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_ar23_weights(
    orig_file: str,
    ar23_file: str,
    outfile: str,
    ext_match_col: list | None = None,
    nulite_key: str = 'nulite',
    nuecc_key: str = 'nuecc',
    mcnuecc_key: str = 'mcnuecc',
) -> None:
    """Merge ar23-plus weights into orig and write a new merged .df file.

    Events are matched via a two-step process: external match on physics-level
    nulite columns (default: E, rec.mc.nu..index, run, subrun, evt) followed by
    an internal remap of file-level coordinates (__ntuple, entry).  Both files
    are deduplicated on the external columns first, so the match is 1-to-1 per
    neutrino (including multi-neutrino events with different rec.mc.nu..index).

    The ar23-only weight columns are added to:
    - ``nuecc``: as ``('slc', 'truth', weight_name, ...)`` columns
    - ``mcnuecc``: as top-level weight columns

    All other tables (hdr, histpotdf, histgenevtdf, nulite) are copied from
    ``orig_file`` unchanged, preserving the original split structure.

    Parameters
    ----------
    orig_file : str
        Path to the original .df file (nulite, nuecc, mcnuecc tables).
    ar23_file : str
        Path to the ar23 .df file (nulite, mcnuecc with extra weights).
    outfile : str
        Path for the merged output .df file.
    ext_match_col : list of str, optional
        Physics-level columns for external event matching.
        Defaults to ``['E', 'rec.mc.nu..index', 'run', 'subrun', 'evt']``.
    nulite_key, nuecc_key, mcnuecc_key : str
        HDF5 table-name prefixes in the .df files.
    """
    import gc

    if ext_match_col is None:
        ext_match_col = _EXT_MATCH_COL

    n_orig = get_n_split(orig_file)
    n_ar23 = get_n_split(ar23_file)

    # ---- Step 1: nulite matching (cheap) -------------------------------------
    print(f"Loading nulite ({n_orig} orig splits, {n_ar23} ar23 splits)...")
    orig_nulite = load_dfs(orig_file, [nulite_key], n_max_concat=n_orig)[nulite_key]
    ar23_nulite = load_dfs(ar23_file, [nulite_key], n_max_concat=n_ar23)[nulite_key]

    orig_lite  = _prep_lite(orig_nulite, ext_match_col)
    ar23_lite  = _prep_lite(ar23_nulite, ext_match_col)
    common_idx = ar23_lite.index.intersection(orig_lite.index)
    n_com, n_ar23_ev, n_orig_ev = len(common_idx), len(ar23_lite), len(orig_lite)
    print(
        f"  {n_com} common events before dedup "
        f"({n_com/n_ar23_ev*100:.1f}% of ar23, {n_com/n_orig_ev*100:.1f}% of orig)"
    )
    if n_com == 0:
        raise ValueError("Empty event intersection — check that ext_match_col is correct.")

    mapping = _build_mapping(orig_lite, ar23_lite, common_idx, ext_match_col)
    overlap_frac = len(mapping) / n_orig_ev
    print(f"  {len(mapping)} unambiguous matched events after dedup")
    print(f"  overlap with orig: {len(mapping)}/{n_orig_ev} = {overlap_frac*100:.2f}%")

    # Compute everything from orig_lite (already deduplicated) before releasing it
    _orig_lite_reset = orig_lite.reset_index()

    # Valid internal indices for lite-dedup filter
    _orig_lite_flat   = _orig_lite_reset[['__ntuple', 'entry', 'rec.mc.nu..index', 'file_idx']]
    orig_valid_mc_idx  = pd.MultiIndex.from_frame(_orig_lite_flat)
    orig_valid_evt_idx = pd.MultiIndex.from_frame(
        _orig_lite_flat[['__ntuple', 'entry', 'file_idx']].drop_duplicates()
    )

    # global_event_id: hash on (run, subrun, evt, maxE_index, maxE).
    # Rename E→maxE and rec.mc.nu..index→maxE_index to disambiguate from mcnuecc column names.
    # Index by full 4-level (ntuple, entry, rec.mc.nu..index, file_idx) — unique after lite dedup.
    _id_cols = _orig_lite_reset.rename(columns={'E': 'maxE', 'rec.mc.nu..index': 'maxE_index'})
    global_event_id = pd.util.hash_pandas_object(
        _id_cols[['run', 'subrun', 'evt', 'maxE_index', 'maxE']], index=False
    )
    global_event_id.index = pd.MultiIndex.from_frame(
        _id_cols[['__ntuple', 'entry', 'maxE_index', 'file_idx']]
        .rename(columns={'maxE_index': 'rec.mc.nu..index'})
    )
    global_event_id.name = 'global_event_id'

    del orig_nulite, ar23_nulite, orig_lite, ar23_lite, _orig_lite_reset, _orig_lite_flat, _id_cols
    gc.collect()

    # ---- Step 2: load all ar23 mcnuecc, remap, then free ---------------------
    print("Loading ar23 mcnuecc and remapping weights...")
    ar23_mc = load_dfs(ar23_file, [mcnuecc_key], n_max_concat=n_ar23)[mcnuecc_key]

    orig_mc_col_set = set(
        load_dfs(orig_file, [mcnuecc_key], n_max_concat=1)[mcnuecc_key].columns.tolist()
    )
    ar23_only_cols = [c for c in ar23_mc.columns if c not in orig_mc_col_set]
    if not ar23_only_cols:
        raise ValueError("No ar23-only columns found; nothing to merge.")
    print(f"  {len(ar23_only_cols)} ar23-only weight columns")

    remapped = _remap_weights(ar23_mc[ar23_only_cols], mapping)
    del ar23_mc
    gc.collect()
    print(f"  {len(remapped)} remapped weight rows")

    # ---- Step 3: join onto orig tables (all splits at once) ------------------
    print("Loading orig mcnuecc and nuecc...")
    orig_mc    = load_dfs(orig_file, [mcnuecc_key], n_max_concat=n_orig)[mcnuecc_key]
    orig_nuecc = load_dfs(orig_file, [nuecc_key],   n_max_concat=n_orig)[nuecc_key]

    print("Merging weights...")
    n_orig_mc = len(orig_mc)
    new_mc    = _join_to_mcnuecc(orig_mc,    remapped)
    if len(new_mc) != n_orig_mc:
        print(f"  Warning: mcnuecc row count changed {n_orig_mc} -> {len(new_mc)} (index not unique)")
    n_orig_nuecc = len(orig_nuecc)
    new_nuecc = _join_to_nuecc(orig_nuecc, remapped)
    del orig_mc, orig_nuecc

    # Capture matched indices from remapped before freeing it
    matched_mc_idx  = remapped.index  # (ntuple, entry, rec.mc.nu..index, file_idx)
    matched_evt_idx = pd.MultiIndex.from_frame(
        remapped.index.to_frame(index=False)[['__ntuple', 'entry', 'file_idx']].drop_duplicates()
    )
    del remapped
    gc.collect()

    # Drop rows whose lite_col key was duplicated in orig (ambiguous event identity)
    print("Dropping lite-duplicated rows from output...")
    new_mc    = new_mc[new_mc.index.isin(orig_valid_mc_idx)]
    nuecc_evt = pd.MultiIndex.from_arrays([
        new_nuecc.index.get_level_values('__ntuple'),
        new_nuecc.index.get_level_values('entry'),
        new_nuecc.index.get_level_values('file_idx'),
    ], names=['__ntuple', 'entry', 'file_idx'])
    new_nuecc = new_nuecc[nuecc_evt.isin(orig_valid_evt_idx)]
    del orig_valid_mc_idx, orig_valid_evt_idx, nuecc_evt
    gc.collect()

    # Add global_event_id: stable per-event hash keyed by (ntuple, entry, rec.mc.nu..index, file_idx)
    n_mc    = new_mc.columns.nlevels
    n_nuecc = new_nuecc.columns.nlevels
    id_col_mc    = ('global_event_id',) + ('',) * (n_mc    - 1)
    id_col_nuecc = ('global_event_id',) + ('',) * (n_nuecc - 1)

    # mcnuecc: same 4-level index — direct reindex
    new_mc[id_col_mc] = global_event_id.reindex(new_mc.index).values

    # nuecc: index has a slice level instead of rec.mc.nu..index — look up via the nu index column
    nu_idx_col  = ('rec.mc.nu..index',) + ('',) * (n_nuecc - 1)
    nu_idx_vals = new_nuecc[nu_idx_col].values
    nu_idx_safe = np.where(pd.isna(nu_idx_vals), -1.0, nu_idx_vals.astype(float))
    lookup_idx = pd.MultiIndex.from_arrays([
        new_nuecc.index.get_level_values('__ntuple'),
        new_nuecc.index.get_level_values('entry'),
        new_nuecc.index.get_level_values('file_idx'),
        nu_idx_safe,
    ], names=['__ntuple', 'entry', 'file_idx', 'rec.mc.nu..index'])
    geid_reord = global_event_id.reindex(
        global_event_id.index.reorder_levels(['__ntuple', 'entry', 'file_idx', 'rec.mc.nu..index'])
    )
    new_nuecc[id_col_nuecc] = geid_reord.reindex(lookup_idx).values
    del global_event_id, geid_reord, lookup_idx
    gc.collect()

    n_mc_total      = len(new_mc)
    _n = new_nuecc.columns.nlevels
    _nuecc_wgt_col  = (('slc', 'truth') + ar23_only_cols[0] + ('',) * (_n - 2 - len(ar23_only_cols[0])))[:_n]
    n_mc_matched    = int(new_mc[ar23_only_cols[0]].notna().sum())
    n_nuecc_total   = len(new_nuecc)
    n_nuecc_matched = int(new_nuecc[_nuecc_wgt_col].notna().sum())
    print(f"  mcnuecc: {n_mc_matched}/{n_mc_total} rows matched ({n_mc_matched/n_mc_total*100:.1f}%)"
          f"  [{n_orig_mc - n_mc_total} lite-duplicated rows dropped]")
    print(f"  nuecc:   {n_nuecc_matched}/{n_nuecc_total} rows matched ({n_nuecc_matched/n_nuecc_total*100:.1f}%)"
          f"  [{n_orig_nuecc - n_nuecc_total} lite-duplicated rows dropped]")

    # Keep only rows with ar23 overlap (consistent with the scaled POT), filtered by index
    new_mc = new_mc[new_mc.index.isin(matched_mc_idx)]
    nuecc_evt2 = pd.MultiIndex.from_arrays([
        new_nuecc.index.get_level_values('__ntuple'),
        new_nuecc.index.get_level_values('entry'),
        new_nuecc.index.get_level_values('file_idx'),
    ], names=['__ntuple', 'entry', 'file_idx'])
    new_nuecc = new_nuecc[nuecc_evt2.isin(matched_evt_idx)]
    del matched_mc_idx, matched_evt_idx, nuecc_evt2
    gc.collect()

    # Apply signal selection to mcnuecc and preprocess+select nuecc
    print("Applying signal selection and preprocessing...")
    from ..analysis  import define_signal
    from ..preprocess import preprocess_mc
    from ..selection  import select

    n_mc_pre = len(new_mc)
    new_mc = define_signal(new_mc)
    new_mc = new_mc[new_mc.signal == 0]
    print(f"  mcnuecc signal: {len(new_mc)}/{n_mc_pre} ({len(new_mc)/n_mc_pre*100:.1f}%)")

    n_nuecc_pre = len(new_nuecc)
    new_nuecc = select(preprocess_mc(new_nuecc))
    print(f"  nuecc selected: {len(new_nuecc)}/{n_nuecc_pre} ({len(new_nuecc)/n_nuecc_pre*100:.1f}%)")
    gc.collect()

    # ---- Step 4: write output ------------------------------------------------
    with pd.HDFStore(orig_file, mode='r') as _s:
        orig_keys = {k.lstrip('/') for k in _s.keys()}
    skip = {nuecc_key, mcnuecc_key, 'split'}
    passthrough: set[str] = set()
    for k in orig_keys:
        parts = k.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit() and parts[0] not in skip:
            passthrough.add(parts[0])

    print(f"Writing {outfile}...")
    with warnings.catch_warnings(), pd.HDFStore(outfile, mode='w') as store:
        warnings.filterwarnings('ignore', '.*not a valid Python identifier.*')
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

        store.put(f'{mcnuecc_key}_0', new_mc,    **_HDF_KW)
        store.put(f'{nuecc_key}_0',   new_nuecc, **_HDF_KW)

        for prefix in sorted(passthrough):
            try:
                df = load_dfs(orig_file, [prefix], n_max_concat=n_orig)[prefix]
                if prefix == 'histpotdf':
                    df = df.copy()
                    print(f"  Applying overlap fraction {overlap_frac:.4f} to histpotdf TotalPOT")
                    print(f"    Original TotalPOT: {df['TotalPOT'].iloc[0]:.3e}")
                    df['TotalPOT'] *= overlap_frac
                    print(f"    Scaled TotalPOT: {df['TotalPOT'].iloc[0]:.3e}")
                store.put(f'{prefix}_0', df, **_HDF_KW)
            except Exception as e:
                print(f"  Warning: could not copy '{prefix}': {e}")

        store.put('split', pd.DataFrame({'n_split': [1]}), **_HDF_KW)

    print(f"Done — written to {outfile}")
