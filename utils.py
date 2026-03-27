"""Generic DataFrame utilities."""
import pandas as pd
from math import floor, log10

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