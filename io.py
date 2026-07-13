"""File input/output utilities for loading HDF5 data files."""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import numpy as np

__all__ = ['get_n_split', 'print_keys', 'load_dfs', 'load_mc', 'load_data',
           'load_mchnl', 'correct_cosmic_weight_mevprtl_df']

# Sentinel distinguishing "caller didn't pass this" from "caller explicitly
# passed None" -- load_mc/load_data use None to mean "skip this step", so the
# unset-default can't also be None. See load_mc's rec_key/preprocess_fn/
# define_signal_fn and load_data's rec_key/preprocess_fn/offbeam_signal_value.
_UNSET = object()

# credit for first three functions to Mun! 
def get_n_split(file):
    """Get the number of splits in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    
    Returns
    -------
    int
        Number of splits in the file.
    """
    this_split_df = pd.read_hdf(file, key="split")
    this_n_split = this_split_df.n_split.iloc[0]
    return this_n_split

def print_keys(file):
    """Print all keys available in an HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    """
    with pd.HDFStore(file, mode='r') as store:
        keys = store.keys()       # list of all keys in the file
        print("Keys:", keys)
        
def load_dfs(file, keys2load, n_max_concat=10, start_split=0):
    """Load DataFrames from split HDF5 file.
    
    Parameters
    ----------
    file : str
        Path to HDF5 file.
    keys2load : list
        List of key names to load from the file.
    n_max_concat : int, optional
        Maximum number of splits to concatenate (default: 10).
    start_split : int, optional
        Starting split index to load from (default: 0).
    
    Returns
    -------
    dict
        Dictionary mapping key names to concatenated DataFrames.
    """
    out_df_dict = {}
    # Single open HDFStore for the whole read, instead of one open+close per
    # pd.read_hdf(file, key=...) call (get_n_split's own read plus one per
    # key-per-split) -- meaningfully fewer file opens for files with several keys
    # and/or several splits, and cheaper on this project's NFS-backed filesystem.
    with pd.HDFStore(file, mode='r') as store:
        this_n_keys = store['split'].n_split.iloc[0] - start_split
        n_concat = min(n_max_concat, this_n_keys)
        for key in keys2load:
            dfs = [store[f'{key}_{i}'] for i in range(start_split, start_split + n_concat)]
            out_df_dict[key] = pd.concat(dfs, ignore_index=False)
    return out_df_dict

def correct_cosmic_weight_mevprtl_df(indf_rec, indf_truth, indf_hdr):
    """Correct the cosmic-background event weight in an HNL MeVPrtl reco DataFrame
    using the matching truth-level total weight.

    Reco and truth don't share a common natural row index -- their sub-object index
    levels differ (``rec.slc..index`` vs. ``rec.mc.prtl..index``) -- so they're joined
    via a composite key built from ``evt`` (looked up per-row from the header table)
    instead of the natural index.

    When present, ``file_idx`` is included in that composite key alongside
    ``(__ntuple, entry, evt)``. This matters for any file produced by
    ``concat_hdf.py``: it resets ``__ntuple``/``entry`` numbering independently per
    source shard and adds a ``file_idx`` level specifically to disambiguate the
    result, so omitting ``file_idx`` from the join key causes rows from different
    shards to collide on the same ``(__ntuple, entry, evt)`` triple -- ``reindex``
    then raises ``ValueError: cannot handle a non-unique multi-index`` (this crashed
    on any concatenated multi-shard HNL sample prior to this fix; verified against
    the real ``mchnl_nupi0_m260.df``, 15 shards). Falls back to the
    ``(__ntuple, entry, evt)``-only key when ``file_idx`` isn't present on either
    side, matching the original behaviour for a raw, non-concatenated single-shard
    file.

    Parameters
    ----------
    indf_rec : pd.DataFrame
        HNL reco (slice-level) DataFrame.
    indf_truth : pd.DataFrame
        HNL MeVPrtl truth DataFrame.
    indf_hdr : pd.DataFrame
        Header DataFrame, used to look up ``evt`` for both ``indf_rec`` and
        ``indf_truth``.
    """

    #Step 1: Build total event weight in reco and truth HNL dataframes

    # Build total event weight in reco and truth HNL dataframes
    fluxw_column = ('slc', 'prtl', 'flux_weight', '', '', '')
    rayw_column = ('slc', 'prtl', 'ray_weight', '', '', '')
    decayw_column = ('slc', 'prtl', 'decay_weight', '', '', '')
    totalw_column = ('weights_mc', '', '', '', '', '')
    indf_rec[totalw_column] = indf_rec[rayw_column] * indf_rec[decayw_column] * indf_rec[fluxw_column]

    truth_fluxw_column = ('flux_weight', '')
    truth_rayw_column = ('ray_weight', '')
    truth_decayw_column = ('decay_weight', '')
    truth_totalw_column = ('weights_mc_truth', '')
    indf_truth[truth_totalw_column] = indf_truth[truth_rayw_column] * indf_truth[truth_decayw_column] * indf_truth[truth_fluxw_column]

    #-------------------------------------------------------------------------------------#
    #Step 2: Join evt from header dataframe into reco dataframe using reco's own full index
    evt_col = ('evt', '', '', '', '', '')
    indf_rec[evt_col] = indf_hdr['evt'].reindex(indf_rec.index)

    #-------------------------------------------------------------------------------------#
    # Step 3: Join truth total weight into reco dataframe on (file_idx, __ntuple, entry, evt)
    # -- file_idx included only when present on both sides, see docstring.
    totalw_truth_col = ('weights_mc_truth', '', '', '', '', '')

    truth_total_weight = indf_truth[truth_totalw_column]
    truth_evt = indf_hdr['evt'].reindex(indf_truth.index).values

    join_levels = ['__ntuple', 'entry']
    if 'file_idx' in indf_truth.index.names and 'file_idx' in indf_rec.index.names:
        join_levels = ['file_idx'] + join_levels
    join_names = join_levels + ['evt']

    truth_key = pd.MultiIndex.from_arrays(
        [indf_truth.index.get_level_values(lvl) for lvl in join_levels] + [truth_evt],
        names=join_names,
    )
    truth_lookup = pd.Series(truth_total_weight.values, index=truth_key)

    reco_key = pd.MultiIndex.from_arrays(
        [indf_rec.index.get_level_values(lvl) for lvl in join_levels] + [indf_rec[evt_col].values],
        names=join_names,
    )

    # .map() instead of Series.reindex(): a direct lookup against reco_key's own order
    # rather than reindex's stricter re-alignment machinery -- same result, cheaper
    # since it doesn't need to build/validate reco_key as a standalone index in its
    # own right the way reindex does.
    indf_rec[totalw_truth_col] = reco_key.map(truth_lookup)

    #-------------------------------------------------------------------------------------#
    # Step 4: Correct cosmic weight in reco dataframe using truth total weight for cosmic entries

    mask_cosmic = np.isnan(indf_rec[('slc', 'prtl', 'E', '', '', '')])
    indf_rec.loc[mask_cosmic, totalw_column] = indf_rec.loc[mask_cosmic, truth_totalw_column]

    #-------------------------------------------------------------------------------------#
    #Drop event column from reco dataframe
    indf_rec = indf_rec.drop(columns=[evt_col])

    return indf_rec


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_mc(
    file: str,
    keys: list | None = None,
    cuts=None,
    max_splits: int | None = None,
    chunk_splits: int = 1,
    add_pi0: bool = False,
    excl_mc_df=None,
    rec_key: str = 'nuecc',
    preprocess_fn=_UNSET,
    define_signal_fn=_UNSET,
) -> tuple:
    """Load, preprocess, and optionally select an MC HDF5 file in chunks.

    Splits are loaded in batches of chunk_splits to balance memory and I/O
    overhead.  POT and generated-event counts are accumulated across all
    splits.  Header columns (run/subrun/event) are merged into the output
    DataFrame.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    keys : list of str, optional
        Table keys to load.  Defaults to
        ``['hdr', rec_key, 'histpotdf', 'histgenevtdf']``.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~nueana.selection.select`.
        When None the full preprocessed DataFrame is returned.
    max_splits : int, optional
        Cap on the number of splits to load.  Defaults to all splits.
    chunk_splits : int, default 1
        Number of splits to load per iteration.  Increase to reduce I/O
        overhead at the cost of higher peak memory per chunk.
    add_pi0 : bool, default False
        If True, compute pi0 kinematics via :func:`~nueana.preprocess.add_pi0`
        for each chunk after preprocessing.
    excl_mc_df : pd.DataFrame, optional
        Exclusive mcnuecc DataFrame with ``define_signal`` already applied
        (i.e. has a top-level ``signal`` column).  When provided,
        :func:`~nueana.exclusive.remove_signal_overlap` is called on the
        final concatenated result to strip events that are already covered
        by the exclusive sample and would otherwise be double-counted.
    rec_key : str, default 'nuecc'
        Key of the main slc-level table within ``keys``/the HDF5 file (e.g.
        ``'rec'`` for makers that don't use the nueCC-style ``'nuecc'`` key).
    preprocess_fn : callable or None, optional
        Called as ``preprocess_fn(df)`` on each chunk's ``rec_key`` table
        before selection. Defaults to :func:`~nueana.preprocess.preprocess_mc`
        (nueCC's flash PE-scale/flash-time fixes). Pass ``None`` to skip
        preprocessing entirely -- e.g. when the caller applies its own
        analysis-specific fixes after loading instead.
    define_signal_fn : callable, optional
        Called as ``define_signal_fn(df)`` on each merged chunk to stamp
        signal categories. Defaults to
        ``lambda df: analysis.define_signal(df, prefix=('slc', 'truth'))``.
        Pass a different single-argument callable (e.g.
        ``nue.define_signal_hnl`` or
        ``functools.partial(nue.define_signal_pi0, prefix=('slc', 'truth'))``)
        for a non-nueCC signal scheme.

    Returns
    -------
    df : pd.DataFrame
        Concatenated, preprocessed (and optionally selected) MC DataFrame
        with header columns merged in and signal categories defined.
    pot : float
        Accumulated POT.
    ngen : float
        Accumulated generated-event count.
    """
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    from .preprocess import preprocess_mc, add_pi0 as _add_pi0
    from .selection import select
    from .analysis import define_signal
    from .utils import merge_hdr

    if keys is None:
        keys = ['hdr', rec_key, 'histpotdf', 'histgenevtdf']
    if preprocess_fn is _UNSET:
        preprocess_fn = preprocess_mc
    if define_signal_fn is _UNSET:
        define_signal_fn = lambda df: define_signal(df, prefix=('slc', 'truth'))

    n_total  = get_n_split(file)
    n_splits = min(max_splits, n_total) if max_splits is not None else n_total
    starts   = range(0, n_splits, chunk_splits)
    iterator = _tqdm(starts) if _tqdm is not None else starts

    pot    = 0.0
    ngen   = 0.0
    chunks = []

    for i in iterator:
        n_load = min(chunk_splits, n_splits - i)
        dfs = load_dfs(file, keys2load=keys, n_max_concat=n_load, start_split=i)

        if 'histpotdf' in dfs:    pot  += dfs['histpotdf'].TotalPOT.sum()
        elif 'hdr' in dfs:        pot  += dfs['hdr'].pot.sum()
        if 'histgenevtdf' in dfs: ngen += dfs['histgenevtdf'].TotalGenEvents.sum()
        elif 'hdr' in dfs:        ngen += dfs['hdr'][dfs['hdr'].first_in_subrun == 1].ngenevt.sum()

        df    = preprocess_fn(dfs[rec_key]) if preprocess_fn is not None else dfs[rec_key]
        sel   = select(df, cuts=cuts) if cuts is not None else df
        chunk = merge_hdr(dfs['hdr'], sel)
        del dfs
        chunk = define_signal_fn(chunk)
        if add_pi0:
            chunk = _add_pi0(chunk)
        chunks.append(chunk)
        del chunk
        gc.collect()

    result = pd.concat(chunks, ignore_index=False).copy()
    if excl_mc_df is not None:
        from .exclusive import remove_signal_overlap
        result = remove_signal_overlap(result, excl_mc_df)
    return result, pot, ngen


def load_data(
    file: str,
    keys: list | None = None,
    onbeam: bool = True,
    cuts=None,
    rec_key: str = 'nuecc',
    preprocess_fn=_UNSET,
    offbeam_signal_value=_UNSET,
) -> tuple:
    """Load, preprocess, and optionally select a data HDF5 file.

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    keys : list of str, optional
        Table keys to load.  Defaults to ``['hdr', rec_key, 'histpotdf']``.
    onbeam : bool, default True
        True for on-beam (BNB) data; False for off-beam.  Controls which
        gate counter is returned and whether the offbeam signal category is
        stamped on the output DataFrame.
    cuts : list of CutSpec, optional
        If supplied, passed to :func:`~nueana.selection.select`.
        When None the full preprocessed DataFrame is returned.
    rec_key : str, default 'nuecc'
        Key of the main slc-level table within ``keys``/the HDF5 file (e.g.
        ``'rec'`` for makers that don't use the nueCC-style ``'nuecc'`` key).
    preprocess_fn : callable or None, optional
        Called as ``preprocess_fn(df)`` on the merged table before selection.
        Defaults to :func:`~nueana.preprocess.preprocess_data` (nueCC's
        flash-time offset fix). Pass ``None`` to skip preprocessing entirely.
    offbeam_signal_value : int, optional
        Signal value stamped on off-beam rows when ``onbeam=False``. Defaults
        to nueCC's ``analysis.signal_dict['offbeam']``. Pass e.g.
        ``nue.signal_dict_hnl['offbeam']`` for a non-nueCC signal scheme.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed (and optionally selected) data DataFrame with header
        columns merged in and pi0 kinematics added.  Off-beam DataFrames
        also have ``signal`` set to ``offbeam_signal_value``.
    pot : float
        Accumulated on-beam POT (0.0 for off-beam files).
    ngates : float
        BNB gate count (on-beam) or off-beam gate count.
    """
    from pyanalib.pandas_helpers import multicol_add
    from .preprocess import preprocess_data, add_pi0
    from .selection import select
    from .utils import merge_hdr
    from .analysis import signal_dict

    if keys is None:
        keys = ['hdr', rec_key, 'histpotdf']
    if preprocess_fn is _UNSET:
        preprocess_fn = preprocess_data
    if offbeam_signal_value is _UNSET:
        offbeam_signal_value = signal_dict['offbeam']

    dfs = load_dfs(file, keys2load=keys)
    df  = merge_hdr(dfs['hdr'], dfs[rec_key])
    df  = preprocess_fn(df) if preprocess_fn is not None else df

    pot    = 0.0
    ngates = 0.0
    if onbeam:
        pot    = dfs['histpotdf'].TotalPOT.sum() if 'histpotdf' in dfs else dfs['hdr'].pot.sum()
        ngates = dfs['hdr'].nbnbinfo.sum()
    else:
        ngates = dfs['hdr'].noffbeambnb.sum()
        signal = pd.Series(
            np.ones(len(df), dtype=np.int16) * offbeam_signal_value,
            name="signal", index=df.index,
        )
        df = multicol_add(df, signal)

    sel = select(df, cuts=cuts) if cuts is not None else df
    return sel, pot, ngates


def load_mchnl(
    file: str,
    keys: list | None = None,
    mevprtl_key: str = 'mevprtl',
    rec_key: str = 'rec',
    preprocess_fn=_UNSET,
) -> tuple:
    """Load and preprocess an HNL MeVPrtl MC HDF5 file.

    Applies the MeVPrtl-specific cosmic-weight correction
    (:func:`correct_cosmic_weight_mevprtl_df`), stamps HNL signal categories
    (:func:`~nueana.selection.define_signal_hnl`), and merges header columns
    in. Also extracts the simulated mixing angle and HNL mass from the truth
    table, since every HNL sample needs these for scaling/labeling.

    Replaces a load sequence that was copy-pasted across 7 HNL analysis
    notebooks (``load_dfs`` + ``correct_cosmic_weight_mevprtl_df`` +
    ``define_signal_hnl`` + ``merge_hdr``, plus manual POT/simU/hnlM
    extraction) -- the copy-paste drift had already caused a real bug in one
    of them (a missing ``merge_hdr`` call).

    Parameters
    ----------
    file : str
        Path to the HDF5 file.
    keys : list of str, optional
        Table keys to load, not including ``mevprtl_key`` (added automatically
        if missing). Defaults to ``['hdr', rec_key, 'histpotdf']``.
    mevprtl_key : str, default 'mevprtl'
        Key of the MeVPrtl truth table.
    rec_key : str, default 'rec'
        Key of the main slc-level table.
    preprocess_fn : callable or None, optional
        Called as ``preprocess_fn(df)`` on the raw ``rec_key`` table before the
        cosmic-weight correction. Defaults to
        :func:`~nueana.preprocess.preprocess_mchnl`. Pass ``None`` to skip.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed HNL DataFrame with header columns merged in and signal
        categories defined.
    pot : float
        HNL MC POT (unscaled).
    info : dict
        ``{'simU': ..., 'hnlM': ...}`` -- simulated mixing angle squared and
        HNL mass in MeV (already converted from the truth table's GeV),
        extracted from the truth table. Build a legend label from these at
        the call site -- exact wording/formatting varies by analysis (nu_ee
        vs. nu_pi0 channel, mass-dependent scale factors, ...), so it isn't
        baked in here.
    """
    from .preprocess import preprocess_mchnl
    from .selection import define_signal_hnl
    from .utils import merge_hdr

    if keys is None:
        keys = ['hdr', rec_key, 'histpotdf']
    load_keys = keys if mevprtl_key in keys else keys + [mevprtl_key]
    if preprocess_fn is _UNSET:
        preprocess_fn = preprocess_mchnl

    # load_dfs defaults to n_max_concat=10 -- silently loads only the first 10 HDF5
    # splits if not told otherwise. Pass the file's actual split count explicitly so
    # a future mchnl production with more than 10 splits doesn't get silently
    # truncated here.
    dfs = load_dfs(file, keys2load=load_keys, n_max_concat=get_n_split(file))
    rec_df = preprocess_fn(dfs[rec_key]) if preprocess_fn is not None else dfs[rec_key]
    df  = correct_cosmic_weight_mevprtl_df(rec_df, dfs[mevprtl_key], dfs['hdr'])
    df  = define_signal_hnl(df)
    df  = merge_hdr(dfs['hdr'], df)

    pot  = dfs['histpotdf'].TotalPOT.sum() if 'histpotdf' in dfs else dfs['hdr'].pot.sum()
    simU = df[('slc', 'prtl', 'C2', '', '', '')].dropna().unique()[0]
    hnlM = df[('slc', 'prtl', 'M', '', '', '')].dropna().unique()[0] * 1000  # GeV -> MeV

    return df, pot, {'simU': simU, 'hnlM': hnlM}
