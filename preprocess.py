"""Preprocessing fixes for MC and data DataFrames.

Each fix is idempotent: calling a fix that has already been applied on the
DataFrame is a no-op (a warning is printed but no error is raised).

Whether a fix has been applied is stored as a boolean column::

    ('_fix_<name>', '', '', ..., '')          # MultiIndex depth matched
    '_fix_<name>'                             # flat Index fallback

The flag column is cheap (one bool per row), survives ``pd.concat``, and
is visible when inspecting the DataFrame.  Use :func:`is_fix_applied` to
check programmatically and :func:`applied_fixes` to list all recorded fixes.

MC-only fixes
-------------
- :func:`fix_flash_pe_scale`  — scale flash PEs by a calibration factor

Data-only fixes
---------------
- :func:`fix_flash_time`      — correct flash time via per-event frame offset

MC + data fixes
---------------
- :func:`add_phi`             — derive shower and track azimuthal angles from direction
"""

import warnings

import numpy as np
import pandas as pd

__all__ = [
    'is_fix_applied',
    'applied_fixes',
    'preprocess_mc',
    'preprocess_data',
    'fix_flash_pe_scale',
    'fix_flash_time',
    'add_phi',
]

_FIX_PREFIX = '_fix_'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag_col(df: pd.DataFrame, name: str):
    """Column key for fix *name*, padded to df's MultiIndex depth."""
    key = f'{_FIX_PREFIX}{name}'
    if not isinstance(df.columns, pd.MultiIndex):
        return key
    depth = df.columns.nlevels
    return tuple([key] + [''] * (depth - 1))


def is_fix_applied(df: pd.DataFrame, name: str) -> bool:
    """Return True if fix *name* has been recorded on *df*."""
    return _flag_col(df, name) in df.columns


def applied_fixes(df: pd.DataFrame) -> list[str]:
    """Return a list of fix names that have been recorded on *df*."""
    if isinstance(df.columns, pd.MultiIndex):
        top = [c[0] for c in df.columns]
    else:
        top = list(df.columns)
    return [c[len(_FIX_PREFIX):] for c in top if c.startswith(_FIX_PREFIX)]


def _mark_applied(df: pd.DataFrame, name: str) -> pd.DataFrame:
    flag_series = pd.Series(True, index=df.index, name=_flag_col(df, name))
    return pd.concat([df, flag_series], axis=1)


def _skip_if_applied(df: pd.DataFrame, name: str) -> bool:
    """Warn and return True if fix already applied; else return False."""
    if is_fix_applied(df, name):
        warnings.warn(
            f"Fix '{name}' is already applied to this DataFrame; skipping.",
            stacklevel=3,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# MC-only fixes
# ---------------------------------------------------------------------------

def fix_flash_pe_scale(df: pd.DataFrame, scale: float = 0.66) -> pd.DataFrame:
    """Scale flash PEs by *scale* (MC only).

    Corrects for the flash PE calibration difference between MC and data.
    Default scale factor is 0.66.

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame containing ``slc.barycenterFM.flashPEs``.
    scale : float
        Multiplicative factor applied to flash PE values.
    """
    name = 'flash_pe_scale'
    if _skip_if_applied(df, name):
        return df
    col = ('slc', 'barycenterFM', 'flashPEs', '', '', '')
    df[col] = df[col] * scale
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Data-only fixes
# ---------------------------------------------------------------------------

def fix_flash_time(df: pd.DataFrame, offset: float = 0.19) -> pd.DataFrame:
    """Apply flash time correction using the per-event frame offset (data only).

    Corrects the recorded flash time as::

        flashTime_corrected = flashTime + frameApplyAtCaf / 1e3 - offset

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame containing ``slc.barycenterFM.flashTime`` and
        ``frameApplyAtCaf``.
    offset : float
        Data–MC timing offset in µs (default 0.19 µs).
    """
    name = 'flash_time'
    if _skip_if_applied(df, name):
        return df
    col = ('slc', 'barycenterFM', 'flashTime', '', '', '')
    df[col] = df.slc.barycenterFM.flashTime + df.frameApplyAtCaf / 1e3 - offset
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Bundled entry points
# ---------------------------------------------------------------------------

def preprocess_mc(df: pd.DataFrame, *, flash_pe_scale: float = 0.66) -> pd.DataFrame:
    """Apply all standard MC preprocessing fixes in order.

    Applies:

    1. :func:`fix_flash_pe_scale` — flash PE calibration correction
    2. :func:`add_phi`            — shower and track azimuthal angles

    All fixes are idempotent; calling this on an already-preprocessed
    DataFrame is safe (each already-applied fix warns and skips).

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame.
    flash_pe_scale : float
        Scale factor forwarded to :func:`fix_flash_pe_scale` (default 0.66).
    """
    df = fix_flash_pe_scale(df, scale=flash_pe_scale)
    df = add_phi(df)
    return df


def preprocess_data(df: pd.DataFrame, *, flash_time_offset: float = 0.19) -> pd.DataFrame:
    """Apply all standard data preprocessing fixes in order.

    Applies:

    1. :func:`fix_flash_time` — flash time frame-offset correction
    2. :func:`add_phi`        — shower and track azimuthal angles

    All fixes are idempotent; calling this on an already-preprocessed
    DataFrame is safe (each already-applied fix warns and skips).

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame (on-beam or off-beam).
    flash_time_offset : float
        Timing offset in µs forwarded to :func:`fix_flash_time` (default 0.19).
    """
    df = fix_flash_time(df, offset=flash_time_offset)
    df = add_phi(df)
    return df


def add_phi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute azimuthal angle φ (degrees) for the primary shower and track.

    Stores results in ``primshw.shw.dir.phi`` and ``primtrk.trk.dir.phi``.
    Track phi will be NaN for events with no reconstructed track.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``primshw.shw.dir.{x,y}`` and
        ``primtrk.trk.dir.{x,y}``.
    """
    name = 'phi'
    if _skip_if_applied(df, name):
        return df
    depth = df.columns.nlevels if isinstance(df.columns, pd.MultiIndex) else 1
    pad = [''] * max(0, depth - 4)
    shw_col = tuple(['primshw', 'shw', 'dir', 'phi'] + pad)
    trk_col = tuple(['primtrk', 'trk', 'dir', 'phi'] + pad)

    # Compute phi values
    shw_phi = np.arctan2(df.primshw.shw.dir.x, df.primshw.shw.dir.y) * 180 / np.pi
    trk_phi = np.arctan2(df.primtrk.trk.dir.x, df.primtrk.trk.dir.y) * 180 / np.pi

    # Batch add columns to avoid fragmentation
    new_cols = pd.DataFrame({shw_col: shw_phi, trk_col: trk_phi})
    df = pd.concat([df, new_cols], axis=1)
    return _mark_applied(df, name)
