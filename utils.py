"""Generic DataFrame utilities."""
import pandas as pd
from pyanalib.pandas_helpers import *

from . import config

def ensure_lexsorted(frame, axis):
    """Ensure DataFrame axes are fully lexsorted when using MultiIndex.
    
    This avoids pandas PerformanceWarning about indexing past lexsort depth.
    
    Parameters
    ----------
    frame : pandas.DataFrame
        DataFrame to check and sort if needed.
    axis : int
        Axis to check (0 for index, 1 for columns).
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with sorted index/columns if MultiIndex, otherwise unchanged.
    """
    # axis: 0 -> index, 1 -> columns
    idx = frame.index if axis == 0 else frame.columns
    if isinstance(idx, pd.MultiIndex) and getattr(idx, "lexsort_depth", 0) < idx.nlevels:
        # sort by all levels (returns a new frame)
        return frame.sort_index(axis=axis)
    return frame

def merge_hdr(hdr_df, df):
    """Add header columns (run/subrun/evt) to main DataFrame by index join.

    Performs a left join on the shared index, so hdr_df values are broadcast
    to all matching rows in df (handles the many-slices-per-event case).
    More memory-efficient than the original multicol_merge: no reset_index
    copies, no merge-key hashing on flat columns.

    Parameters
    ----------
    hdr_df : pandas.DataFrame
        Header DataFrame with run, subrun, evt (and optionally file_idx) columns,
        indexed by (__ntuple, entry).
    df : pandas.DataFrame
        Main event DataFrame indexed by (__ntuple, entry), possibly with
        multiple rows per event (slices).

    Returns
    -------
    pandas.DataFrame
        df with run/subrun/evt (and file_idx if present) columns appended.
    """
    add_cols = ['run', 'subrun', 'evt']
    if 'file_idx' in hdr_df.columns:
        add_cols.append('file_idx')

    hdr_subset = hdr_df[add_cols]
    col_depth = df.columns.nlevels
    if col_depth > 1:
        hdr_subset = hdr_subset.copy()
        hdr_subset.columns = pd.MultiIndex.from_tuples(
            [tuple([c] + [''] * (col_depth - 1)) for c in add_cols]
        )

    result = df.join(hdr_subset)
    return ensure_lexsorted(ensure_lexsorted(result, axis=0), axis=1)

def apply_event_mask(df: pd.DataFrame, event_mask: str | None = None) -> pd.DataFrame:
    """ Apply event mask filter to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'signal' column.
    event_mask : str or None
        Event classification filter: 'all', 'signal', or 'background'.
        If None (default), returns all events.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame based on the event mask.
        - 'signal': events where signal == 0
        - 'background': events where signal != 0
        - 'all' or None: all events
        
    Raises
    ------
    ValueError
        If event_mask is not one of the allowed values.
    """
    # Normalize: convert None to "all" and validate
    if event_mask is None:
        event_mask = "all"
    if event_mask not in {"all", "signal", "background"}:
        raise ValueError("event_mask must be one of: 'all', 'signal', 'background', or None")
    
    # Apply: filter based on signal column (0 = signal, nonzero = background)
    if event_mask == "signal":
        return df[df.signal == 0]
    if event_mask == "background":
        return df[df.signal != 0]
    return df