"""Generic DataFrame utilities."""
import pandas as pd
from math import floor, log10
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

# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)
def merge_hdr(hdr_df,df):
    """Merge header DataFrame with main DataFrame on entry and __ntuple.
    
    Parameters
    ----------
    hdr_df : pandas.DataFrame
        DataFrame containing header information with columns including '__ntuple' and 'entry'.
    df : pandas.DataFrame
        Main DataFrame containing event data with columns including '__ntuple' and 'entry'.
    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing all columns from both hdr_df and df, merged on '__ntuple' and 'entry'.
    Notes
    -----
    - The merge is performed on the columns '__ntuple' and 'entry', which are expected to be present in both DataFrames.
    - The function ensures that both DataFrames are lexsorted on the relevant columns before merging to avoid performance issues with MultiIndex.
    """
    nlevels = df.index.nlevels 
    hdr_cols = ['__ntuple','entry','run','subrun','evt']
    return multicol_merge(ensure_lexsorted(hdr_df.reset_index(),axis=1)[hdr_cols],
                          ensure_lexsorted(df.reset_index(),axis=1),
                          on = [tuple(['__ntuple'] + (nlevels-1)*['']),
                                tuple(['entry']    + (nlevels-1)*['']),]
                          )

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