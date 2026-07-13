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
- :func:`fix_sec_shw_energy` — scale secondary shower energy from maxplane_energy
- :func:`add_phi`            — derive shower and track azimuthal angles from direction

Timing calibration fixes (opt-in via preprocess_mc/preprocess_data)
---------------------------------------------------------------------
- :func:`fix_bfm_flashtime_mcbnb`  — BFM flashTime reco bugfix, BNB overlay generator
- :func:`fix_bfm_flashtime_mchnl`  — BFM flashTime reco bugfix, MeVPrtl generator
- :func:`fix_timing_calibration`   — adds flashTime_<prefix>/_mod via timing_correction
- :func:`fix_databnb_timing_calibration` — drops bad-period rows, corrects per good period

Pi0 fix (opt-in, call after preprocess_mc / preprocess_data)
-------------------------------------------------------------
- :func:`add_pi0`  — pi0 kinematics: opening angle, invariant mass, momentum

Derived kinematic variables (opt-in, call whenever needed)
------------------------------------------------------------
- :func:`add_variables` — per-shower angle_z/phi, transverse distance to beam, and the
  two-shower transverse-mass proxy m_alt. Topology-agnostic (only touches primshw/secshw
  shower columns, gated on their existence -- unlike the fixes above, works the same on
  nueCC or HNL/pi0 'rec' tables). Was previously its own module (new_variables.py),
  folded in here since it serves the same "derive columns before analysis" purpose.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .utils import ensure_lexsorted
from . import timing_calibration as tc

__all__ = [
    'is_fix_applied',
    'applied_fixes',
    'preprocess_mc',
    'preprocess_mcbnb',
    'preprocess_mchnl',
    'preprocess_data',
    'preprocess_databnb',
    'preprocess_dataoff',
    'fix_flash_pe_scale',
    'fix_flash_time',
    'add_phi',
    'fix_prim_shw_energy',
    'fix_sec_shw_energy',
    'fix_bfm_flashtime_mcbnb',
    'fix_bfm_flashtime_mchnl',
    'fix_timing_calibration',
    'fix_databnb_timing_calibration',
    'add_pi0',
    'add_variables',
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
# Timing calibration fixes (MC + data)
# ---------------------------------------------------------------------------

def fix_bfm_flashtime_mcbnb(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the BFM flashTime reco offset for the BNB overlay generator (MC only).

    Wraps :func:`timing_calibration.bugfix_mcbnb_bfm_flashtime`. Call before
    :func:`fix_timing_calibration`. Applies to any BNB-overlay-generator sample
    (e.g. detvar CV/DV frames), not just mcbnb_df.
    """
    name = 'bfm_flashtime_mcbnb'
    if _skip_if_applied(df, name):
        return df
    df = tc.bugfix_mcbnb_bfm_flashtime(df)
    return _mark_applied(df, name)


def fix_bfm_flashtime_mchnl(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the BFM flashTime reco offset for the MeVPrtl generator (MC only).

    Wraps :func:`timing_calibration.bugfix_mchnl_bfm_flashtime` -- a different
    fixed offset than :func:`fix_bfm_flashtime_mcbnb` (subtraction only, no extra
    beam-period term). Call before :func:`fix_timing_calibration`.
    """
    name = 'bfm_flashtime_mchnl'
    if _skip_if_applied(df, name):
        return df
    df = tc.bugfix_mchnl_bfm_flashtime(df)
    return _mark_applied(df, name)


def fix_timing_calibration(df: pd.DataFrame, *, period: float, t0_offset: float,
                            prefix: str = 'calib', ifData: bool = False) -> pd.DataFrame:
    """Add calibrated flash-time columns via :func:`timing_calibration.timing_correction`.

    Adds ``slc.barycenterFM.flashTime_<prefix>`` and its period-folded
    ``..._<prefix>_mod`` counterpart. Real Data BNB needs
    :func:`timing_calibration.data_filter_bad`/:func:`~timing_calibration.data_correct_good`
    instead (per-run-period constants, not a single fixed value).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``slc.barycenterFM.flashTime`` and ``slc.vertex.z``
        (plus ``frameApplyAtCaf`` when ``ifData=True``).
    period, t0_offset : float
        Calibration constants in ns.
    prefix : str, default 'calib'
        Suffix used for the output column names.
    ifData : bool, default False
        Whether to apply the data-only T0 (``frameApplyAtCaf``) correction term.
    """
    name = f'timing_calibration_{prefix}'
    if _skip_if_applied(df, name):
        return df
    df = tc.timing_correction(df, period=period, t0_offset=t0_offset, prefix=prefix, ifData=ifData)
    return _mark_applied(df, name)


def fix_databnb_timing_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality cut plus per-run-period flash time calibration (Data BNB only).

    Drops slices without RWM timing info (``tdcRwm == 0``), then wraps
    :func:`timing_calibration.data_filter_bad` (drops known-bad run periods) and
    :func:`timing_calibration.data_correct_good` (per-run-period constants:
    ``timing_calibration.bad_period_dict``/``good_period_dict``/``odict``/``pdict``).
    Unlike :func:`fix_timing_calibration`, this drops rows and has no single fixed
    period/offset, since real Data BNB's calibration drifted across changing
    beam/LLRF conditions.
    """
    name = 'databnb_timing_calibration'
    if _skip_if_applied(df, name):
        return df
    print("Removing entries with tdcRwm==0:", len(df[df.tdcRwm == 0]))
    df = df[df.tdcRwm != 0].copy()
    df['rwm_datetime'] = pd.to_datetime(df['tdcRwm'])
    df = tc.data_filter_bad(df, tc.bad_period_dict)
    df = tc.data_correct_good(df, tc.good_period_dict, tc.odict, tc.pdict)
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Bundled entry points
# ---------------------------------------------------------------------------

def preprocess_mc(df: pd.DataFrame) -> pd.DataFrame:
    """Generic MC preprocessing entry point for the HNL/pi0 ``'rec'`` table.

    Currently a no-op: this analysis's samples don't need
    ``fix_flash_pe_scale``/``fix_prim_shw_energy``/``fix_sec_shw_energy``/
    ``add_phi``. Kept as the base :func:`preprocess_mcbnb`/:func:`preprocess_mchnl`
    build on, and for import compatibility with nueCC-topology callers elsewhere
    in this repo.

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame.
    """
    return df


def preprocess_mcbnb(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`preprocess_mc` plus BNB overlay generator timing calibration.

    Adds :func:`fix_bfm_flashtime_mcbnb` and :func:`fix_timing_calibration`
    (``timing_calibration.mcbnb_period_calib``/``mcbnb_offset_calib``), plus
    :func:`add_variables` (derived kinematic columns -- was a separate notebook
    "Add New Variables" step, now applied at load time like nueCC's convention).
    Also the correct call for any BNB-overlay-generator sample (e.g. detvar CV/DV
    frames), not just mcbnb_df.

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame.
    """
    df = preprocess_mc(df)
    df = fix_bfm_flashtime_mcbnb(df)
    df = fix_timing_calibration(df, period=tc.mcbnb_period_calib, t0_offset=tc.mcbnb_offset_calib)
    df = add_variables(df)
    return df


def preprocess_mchnl(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`preprocess_mc` plus MeVPrtl generator timing calibration.

    Adds :func:`fix_bfm_flashtime_mchnl` and :func:`fix_timing_calibration`
    (``timing_calibration.mchnl_period_calib``/``mchnl_offset_calib``), plus
    :func:`add_variables` (derived kinematic columns -- was a separate notebook
    "Add New Variables" step, now applied at load time like nueCC's convention).

    Parameters
    ----------
    df : pd.DataFrame
        MC DataFrame.
    """
    df = preprocess_mc(df)
    df = fix_bfm_flashtime_mchnl(df)
    df = fix_timing_calibration(df, period=tc.mchnl_period_calib, t0_offset=tc.mchnl_offset_calib)
    df = add_variables(df)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generic data preprocessing entry point for the HNL/pi0 ``'rec'`` table.

    Currently a no-op: this analysis's samples don't need
    ``fix_flash_time``/``fix_prim_shw_energy``/``fix_sec_shw_energy``/``add_phi``.
    Kept as the base :func:`preprocess_databnb`/:func:`preprocess_dataoff` build
    on, and for import compatibility with nueCC-topology callers elsewhere in
    this repo.

    Parameters
    ----------
    df : pd.DataFrame
        Data DataFrame (on-beam or off-beam).
    """
    return df


def preprocess_databnb(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`preprocess_data` plus real Data BNB's timing calibration.

    Adds :func:`fix_databnb_timing_calibration` (drops bad-period rows,
    corrects each good period individually), plus :func:`add_variables` (derived
    kinematic columns -- was a separate notebook "Add New Variables" step, now
    applied at load time like nueCC's convention).

    Parameters
    ----------
    df : pd.DataFrame
        Data BNB DataFrame.
    """
    df = preprocess_data(df)
    df = fix_databnb_timing_calibration(df)
    df = add_variables(df)
    return df


def preprocess_dataoff(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`preprocess_data` plus Data Offbeam+Light timing calibration.

    Adds :func:`fix_timing_calibration` (``timing_calibration.offbeam_period_calib``/
    ``offbeam_offset_calib``, ``ifData=True``), plus :func:`add_variables` (derived
    kinematic columns -- was a separate notebook "Add New Variables" step, now
    applied at load time like nueCC's convention).

    Parameters
    ----------
    df : pd.DataFrame
        Data Offbeam+Light DataFrame.
    """
    df = preprocess_data(df)
    df = fix_timing_calibration(df, period=tc.offbeam_period_calib, t0_offset=tc.offbeam_offset_calib,
                                 ifData=True)
    df = add_variables(df)
    return df


def add_phi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute azimuthal angle φ (degrees) for the primary shower and, if present, track.

    Stores results in ``primshw.shw.dir.phi`` and, only when track direction columns
    exist, ``primtrk.trk.dir.phi`` -- topologies with no track-level table at all (e.g.
    HNL/pi0's 'rec' table has no primtrk) get shower phi only, rather than crashing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``primshw.shw.dir.{x,y}`` and, optionally,
        ``primtrk.trk.dir.{x,y}``.
    """
    name = 'phi'
    if _skip_if_applied(df, name):
        return df
    depth = df.columns.nlevels if isinstance(df.columns, pd.MultiIndex) else 1
    pad = [''] * max(0, depth - 4)
    shw_col = tuple(['primshw', 'shw', 'dir', 'phi'] + pad)
    trk_col = tuple(['primtrk', 'trk', 'dir', 'phi'] + pad)
    trk_x_col = tuple(['primtrk', 'trk', 'dir', 'x'] + pad)
    trk_y_col = tuple(['primtrk', 'trk', 'dir', 'y'] + pad)

    # Compute phi values
    new_cols = {shw_col: np.arctan2(df.primshw.shw.dir.x, df.primshw.shw.dir.y) * 180 / np.pi}
    if trk_x_col in df.columns and trk_y_col in df.columns:
        new_cols[trk_col] = np.arctan2(df.primtrk.trk.dir.x, df.primtrk.trk.dir.y) * 180 / np.pi

    # Batch add columns to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Shower energy fixes
# ---------------------------------------------------------------------------

def fix_prim_shw_energy(df: pd.DataFrame, scale: float = 1.17) -> pd.DataFrame:
    """Set primary shower reco_energy from maxplane_energy * scale.

    If ``primshw.shw.reco_energy`` already exists, checks that its ratio to
    ``maxplane_energy`` matches *scale* and warns if not.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``primshw.shw.maxplane_energy``.
    scale : float
        Energy scale factor (default 1.17).
    """
    name = 'prim_shw_energy'
    if _skip_if_applied(df, name):
        return df
    col = ('primshw', 'shw', 'reco_energy', '', '', '')
    if col in df.columns:
        ratio = (df[col] / df.primshw.shw.maxplane_energy).dropna()
        if not np.allclose(ratio, scale, rtol=0.01):
            warnings.warn(
                f"primshw.shw.reco_energy already exists but ratio to maxplane_energy "
                f"differs from {scale} (mean ratio: {ratio.mean():.3f}). "
                "Overwriting with maxplane_energy * scale.",
                stacklevel=2,
            )
    df[col] = df.primshw.shw.maxplane_energy * scale
    return _mark_applied(df, name)


def fix_sec_shw_energy(df: pd.DataFrame, scale: float = 1.17) -> pd.DataFrame:
    """Set secondary shower reco_energy from maxplane_energy * scale.

    Mirrors what :func:`~nueana.selection.select` does for the primary shower.
    Must be called before :func:`add_pi0` so the pi0 invariant mass uses the
    scaled energy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``secshw.shw.maxplane_energy``.
    scale : float
        Energy scale factor (default 1.17, matching the primary shower default).
    """
    name = 'sec_shw_energy'
    if _skip_if_applied(df, name):
        return df
    col = ('secshw', 'shw', 'reco_energy', '', '', '')
    df[col] = df.secshw.shw.maxplane_energy * scale
    return _mark_applied(df, name)


def add_pi0(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pi0 kinematics from the primary and secondary shower.

    Assumes ``primshw.shw.reco_energy`` and ``secshw.shw.reco_energy`` have
    already been set (call :func:`fix_sec_shw_energy` first if needed).

    Derived columns
    ---------------
    pi0.cos2angle             — cos of opening angle between the two showers
    pi0.openangle             — opening angle in degrees
    primshw.shw.p.{x,y,z}    — primary shower momentum vector
    secshw.shw.p.{x,y,z}     — secondary shower momentum vector
    pi0.mass                  — pi0 invariant mass [GeV]
    pi0.p.{x,y,z}             — pi0 momentum vector
    pi0.p.totp                — pi0 momentum magnitude
    pi0.dir.{x,y,z}           — pi0 unit direction

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing primary and secondary shower direction and energy
        columns.
    """
    name = 'pi0'
    if _skip_if_applied(df, name):
        return df

    def _valid(s, sentinel=-999):
        return s.where(s != sentinel)

    prim_E  = df.primshw.shw.reco_energy
    sec_E   = df.secshw.shw.reco_energy
    prim_dx = _valid(df.primshw.shw.dir.x)
    prim_dy = _valid(df.primshw.shw.dir.y)
    prim_dz = _valid(df.primshw.shw.dir.z)
    sec_dx  = _valid(df.secshw.shw.dir.x)
    sec_dy  = _valid(df.secshw.shw.dir.y)
    sec_dz  = _valid(df.secshw.shw.dir.z)

    cos2angle  = prim_dx*sec_dx + prim_dy*sec_dy + prim_dz*sec_dz
    open_angle = np.degrees(np.arccos(cos2angle.clip(-1, 1)))

    prim_px = prim_E * prim_dx
    prim_py = prim_E * prim_dy
    prim_pz = prim_E * prim_dz
    sec_px  = sec_E  * sec_dx
    sec_py  = sec_E  * sec_dy
    sec_pz  = sec_E  * sec_dz

    pi0_px  = prim_px + sec_px
    pi0_py  = prim_py + sec_py
    pi0_pz  = prim_pz + sec_pz
    pi0_mag = np.sqrt(pi0_px**2 + pi0_py**2 + pi0_pz**2)

    pi0_mag_safe = pi0_mag.where(pi0_mag > 0, other=np.nan)
    pi0_dx = pi0_px / pi0_mag_safe
    pi0_dy = pi0_py / pi0_mag_safe
    pi0_dz = pi0_pz / pi0_mag_safe

    pi0_mass = np.sqrt((2 * prim_E * sec_E * (1 - cos2angle)).clip(0))
    
    alpha = (prim_E - sec_E) / (prim_E + sec_E)
    _denom = (1 - alpha**2) * (1 - cos2angle)
    _denom_safe = np.where(_denom > 0, _denom, np.nan)
    _arg = (2 / _denom_safe) - 1
    pi0_mag_alt = 0.135 * np.sqrt(np.where(_arg >= 0, _arg, np.nan))

    new_cols = pd.DataFrame({
        ('pi0',    'cos2angle', '',    '', '', ''): cos2angle,
        ('pi0',    'openangle', '',    '', '', ''): open_angle,
        ('primshw', 'shw',     'p', 'x', '', ''): prim_px,
        ('primshw', 'shw',     'p', 'y', '', ''): prim_py,
        ('primshw', 'shw',     'p', 'z', '', ''): prim_pz,
        ('secshw',  'shw',     'p', 'x', '', ''): sec_px,
        ('secshw',  'shw',     'p', 'y', '', ''): sec_py,
        ('secshw',  'shw',     'p', 'z', '', ''): sec_pz,
        ('pi0',    'mass',      '',    '', '', ''): pi0_mass,
        ('pi0',    'p',        'x',    '', '', ''): pi0_px,
        ('pi0',    'p',        'y',    '', '', ''): pi0_py,
        ('pi0',    'p',        'z',    '', '', ''): pi0_pz,
        ('pi0',    'p',        'totp', '', '', ''): pi0_mag,
        ('pi0',    'p',        'totp_alt', '', '', ''): pi0_mag_alt,
        ('pi0',    'alpha',     '',    '', '', ''): alpha,
        ('pi0',    'dir',      'x',    '', '', ''): pi0_dx,
        ('pi0',    'dir',      'y',    '', '', ''): pi0_dy,
        ('pi0',    'dir',      'z',    '', '', ''): pi0_dz,
    }, index=df.index)

    df = pd.concat([df, new_cols], axis=1)
    return _mark_applied(df, name)


# ---------------------------------------------------------------------------
# Derived kinematic variables (topology-agnostic)
# ---------------------------------------------------------------------------

def add_variables(df: pd.DataFrame, beam_x: float = -74.0, beam_y: float = 0.0) -> pd.DataFrame:
    """Add derived kinematic columns to a slice-level DataFrame from make_topo_df.

    Unlike the fixes above, this only ever touches primshw/secshw shower columns
    (each gated on existence), so it works unchanged on nueCC or HNL/pi0 'rec' tables.

    Columns added:

    Per shower (primshw / secshw if present):
      (shw, 'shw', 'phi', '', '', '')              -- azimuthal angle [deg]

    ``angle_z``/``transverse_distance_beam_2``/``m_alt`` are NOT computed here for the
    HNL/pi0 'rec' table -- cafpyana's own maker (analysis_village/hnl_nuee_nupi0/
    makedf/make_hnldf.py's own add_variables()) already computes identical values for
    these at df-production time, confirmed present on the raw loaded DataFrame.
    Recomputing them here would be a harmless but wasted duplicate (same formula, same
    inputs). See the commented-out code below if a future 'rec' source ever lacks them.

    Parameters
    ----------
    df : pd.DataFrame
        Slice-level DataFrame produced by topology.make_topo_df.
    beam_x, beam_y : float
        Beam centre x and y position [cm]. Default: (-74, 0). Unused now that the
        transverse_distance_beam_2 block below is commented out; kept in the
        signature so call sites don't need to change if it's reinstated.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new columns appended.
    """
    name = 'variables'
    if _skip_if_applied(df, name):
        return df

    df = ensure_lexsorted(df, axis=1)
    # Ensures primshw.shw.dir.phi exists (and primtrk.trk.dir.phi too, if that table is
    # present) -- idempotent, so a no-op if add_phi already ran. Reused below instead of
    # recomputing the same arctan2 a second time under a different column path.
    df = add_phi(df)

    # def _angle_z(shw):
    #     dx = df[(shw, 'shw', 'dir', 'x', '', '')].values
    #     dy = df[(shw, 'shw', 'dir', 'y', '', '')].values
    #     dz = df[(shw, 'shw', 'dir', 'z', '', '')].values
    #     n  = np.sqrt(dx**2 + dy**2 + dz**2)
    #     with np.errstate(invalid='ignore'):
    #         return np.degrees(np.arccos(np.clip(dz / np.where(n > 0, n, np.nan), -1, 1)))
    #
    # for shw in ('primshw', 'secshw'):
    #     if (shw, 'shw', 'dir', 'z', '', '') in df.columns:
    #         df[(shw, 'shw', 'angle_z', '', '', '')] = _angle_z(shw)

    if ('primshw', 'shw', 'dir', 'phi', '', '') in df.columns:
        df[('primshw', 'shw', 'phi', '', '', '')] = df.primshw.shw.dir.phi
    if ((('secshw', 'shw', 'dir', 'x', '', '') in df.columns)
            and (('secshw', 'shw', 'dir', 'y', '', '') in df.columns)):
        dx = df[('secshw', 'shw', 'dir', 'x', '', '')].values
        dy = df[('secshw', 'shw', 'dir', 'y', '', '')].values
        df[('secshw', 'shw', 'phi', '', '', '')] = np.arctan2(dx, dy) * 180 / np.pi

    # vtx_x_col = ('slc', 'vertex', 'x', '', '', '')
    # vtx_y_col = ('slc', 'vertex', 'y', '', '', '')
    # if vtx_x_col in df.columns and vtx_y_col in df.columns:
    #     df[('slc', 'vertex', 'transverse_distance_beam_2', '', '', '')] = (
    #         (df[vtx_x_col].values - beam_x) ** 2 +
    #         (df[vtx_y_col].values - beam_y) ** 2
    #     )
    #
    # has_prim = ('primshw', 'shw', 'bestplane_energy', '', '', '') in df.columns
    # has_sec  = ('secshw',  'shw', 'bestplane_energy', '', '', '') in df.columns
    #
    # if has_prim and has_sec:
    #     E1 = df[('primshw', 'shw', 'bestplane_energy', '', '', '')].values
    #     E2 = df[('secshw',  'shw', 'bestplane_energy', '', '', '')].values
    #
    #     def _unit(shw):
    #         dx = df[(shw, 'shw', 'dir', 'x', '', '')].values
    #         dy = df[(shw, 'shw', 'dir', 'y', '', '')].values
    #         dz = df[(shw, 'shw', 'dir', 'z', '', '')].values
    #         n  = np.sqrt(dx**2 + dy**2 + dz**2)
    #         n  = np.where(n > 0, n, np.nan)
    #         return dx/n, dy/n, dz/n
    #
    #     ux1, uy1, uz1 = _unit('primshw')
    #     ux2, uy2, uz2 = _unit('secshw')
    #
    #     with np.errstate(invalid='ignore'):
    #         ET1 = E1 * np.sqrt(np.clip(1 - uz1**2, 0, None))
    #         ET2 = E2 * np.sqrt(np.clip(1 - uz2**2, 0, None))
    #         cos_theta = np.clip(ux1*ux2 + uy1*uy2 + uz1*uz2, -1, 1)
    #         df[('slc', 'm_alt', '', '', '', '')] = (
    #             np.sqrt(2 * ET1 * ET2 * (1 - cos_theta)) * 1000  # GeV -> MeV
    #         )

    return _mark_applied(df, name)
