"""
Software-based recombination detector variations for shower energy calibration.

Produces modified nuecc DataFrames (suitable for ``DetVarFile.slc_df``) by
recomputing shower dEdx and/or rescaling energy columns under each systematic:

  Variation   Description                                          Modified columns
  ----------  ---------------------------------------------------  ------------------------------------
  calo_Ccal   ±2 % overall charge-to-energy calibration scale      bestplane_dEdx, max/bestplane_energy, reco_energy, density
  calo_alpha  ±0.008 on recombination parameter A (Birks/Box)      bestplane_dEdx
  calo_beta90 ±0.008 on recombination parameter B_90               bestplane_dEdx
  calo_R      ±0.02 on recombination ellipticity R                 bestplane_dEdx
  calo_phi    angle-dependent recombination (unisim)               bestplane_dEdx
  calo_yz     spatial YZ calibration map correction (unisim)       bestplane_dEdx, max/bestplane_energy, reco_energy, density
  calo_Ecorr  ±3 %/1.17 direct energy scale (no recombination)     max/bestplane_energy, reco_energy, density

The ±1σ values for Ccal, alpha, beta90, and R are read directly from
``makedf.chi2pid.CALO_VARIATIONS`` (cafpyana), so updating those entries
there automatically propagates here.

For each multisim variation the list is [+1σ, −1σ]; unisim variations return a
single-element list.  All shower prefixes in ``showers`` (default primshw and
secshw) are modified; columns that are absent or fully-NaN are left untouched.

Usage
-----
    from nueana.detvar.store  import prepare_detvar_df, write_detvar_store
    from nueana.detvar.recomb import make_recomb_detvars

    cv = prepare_detvar_df('detvar_cv.df')
    dv_dfs = make_recomb_detvars(cv.slc_df)

    # wrap in DetVarFile (lite_df is unchanged – 100 % event overlap)
    dv_files = {name: [cv._replace(slc_df=df) for df in dfs]
                for name, dfs in dv_dfs.items()}
    write_detvar_store('recomb_detvars.h5',
        cv_dict = {'cv': cv},
        dv_dict = dv_files,
        cv_map  = {name: 'cv' for name in dv_files},
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ['make_recomb_detvars']

_DEFAULT_SHOWERS = ['primshw', 'secshw']

# Maps pairs of CALO_VARIATIONS keys to DetVar output names.
# Each tuple is (output_key, plus_key, minus_key).
_CALO_PAIRS = [
    ('calo_Ccal',   'ccal_p',  'ccal_m'),
    ('calo_alpha',  'alpha_p', 'alpha_m'),
    ('calo_beta90', 'beta_p',  'beta_m'),
    ('calo_R',      'R_p',     'R_m'),
]
# Plane index used to read c_cal_frac from a CALO_VARIATIONS entry.
# Collection plane (2) is the default bestplane for showers.
_CALO_PLANE = 2

# Defer CVMFS-backed calibration imports to first call.
_chi2pid = None
_calo    = None


def _import_deps():
    global _chi2pid, _calo
    if _chi2pid is None:
        from makedf import chi2pid as _c
        _chi2pid = _c
    if _calo is None:
        from makedf import calo as _ca
        _calo = _ca


# ---------------------------------------------------------------------------
# DataFrame column helpers
# ---------------------------------------------------------------------------

def _pad(df, *keys):
    """Column tuple padded with empty strings to the DataFrame's column nlevels."""
    return keys + ('',) * (df.columns.nlevels - len(keys))


def _get(df, *keys):
    """Return a column Series, or None if absent."""
    key = _pad(df, *keys)
    return df[key] if key in df.columns else None


def _set(df, *keys, value, sentinel=-999):
    """Return a copy of df with one column replaced; no-op if the column is absent.
    Entries equal to *sentinel* in the original column are left unchanged."""
    key = _pad(df, *keys)
    if key not in df.columns:
        return df
    out = df.copy()
    orig = out[key]
    out[key] = value
    out.loc[orig == sentinel, key] = sentinel
    return out


def _has_shower(df, shower):
    return shower in df.columns.get_level_values(0)


# ---------------------------------------------------------------------------
# Recombination helpers
# ---------------------------------------------------------------------------

def _to_dqdx(df, shower):
    """Back-compute dQdx from stored bestplane_dEdx at phi=90, SBND defaults."""
    _import_deps()
    dedx = _get(df, shower, 'shw', 'bestplane_dEdx')
    if dedx is None:
        return None
    with np.errstate(invalid='ignore'):
        return _calo.recombination_sbnd(dedx, 90)


def _to_dedx(dqdx, phi=90, alpha=None, beta90=None, R=None):
    """Convert dQdx → dEdx with SBND parameters, optionally varied."""
    _import_deps()
    cv = _chi2pid.SBND_CALO_PARAMS
    A   = alpha  if alpha  is not None else cv['alpha_emb'][0]
    B90 = beta90 if beta90 is not None else cv['beta_90'][0]
    Rv  = R      if R      is not None else cv['R_emb'][0]
    return _calo.recombination_cor(
        dqdx, phi,
        E=_calo.Efield_sbnd,
        rho=_calo.LAr_density_gmL_sbnd,
        A=A, B90=B90, R=Rv,
    )


# ---------------------------------------------------------------------------
# Per-shower modification helpers
# ---------------------------------------------------------------------------

def _apply_recomb(df, shower, *, dqdx_scale=1.0, phi=90,
                  alpha=None, beta90=None, R=None):
    """Recompute bestplane_dEdx for one shower with modified recombination.

    Sequence: dQdx = inv_recomb(dEdx, phi=90) → dQdx *= dqdx_scale →
              new_dEdx = recomb(dQdx, phi, alpha, beta90, R).
    ``dqdx_scale`` and ``phi`` may be scalars or aligned Series.
    Returns df unchanged when the shower column is absent.
    """
    if not _has_shower(df, shower):
        return df
    dqdx = _to_dqdx(df, shower)
    if dqdx is None:
        return df
    new_dedx = _to_dedx(dqdx * dqdx_scale, phi=phi, alpha=alpha, beta90=beta90, R=R)
    return _set(df, shower, 'shw', 'bestplane_dEdx', value=new_dedx)


def _apply_energy_scale(df, shower, scale):
    """Scale shower energy columns and density for one shower.

    Scales ``maxplane_energy`` and ``bestplane_energy`` so that the variation
    survives when ``select()`` creates ``reco_energy`` from them afterwards.
    Also scales ``reco_energy`` and ``density`` directly if present, keeping
    all energy columns consistent in the post-selection case.
    ``scale`` may be a scalar or an aligned Series (e.g. from the YZ map).
    Columns that are absent are silently skipped.
    """
    if not _has_shower(df, shower):
        return df
    for col in ('maxplane_energy', 'bestplane_energy', 'reco_energy', 'density'):
        v = _get(df, shower, 'shw', col)
        if v is not None:
            df = _set(df, shower, 'shw', col, value=v * scale)
    return df


# ---------------------------------------------------------------------------
# Phi angle and YZ scale
# ---------------------------------------------------------------------------

def _shower_phi(df, shower):
    """Compute shower angle w.r.t. drift (degrees) from the stored direction vector.

    Returns None when the direction column is absent.
    """
    dx = _get(df, shower, 'shw', 'dir', 'x')
    if dx is None:
        return None
    return np.degrees(np.arccos(dx.clip(-1.0, 1.0)))


def _yz_scale(df, shower):
    """Per-event YZ spatial calibration scale from the SBND MC map.

    Returns a pd.Series of ones when position/plane columns are absent.
    """
    _import_deps()
    y     = _get(df, shower, 'shw', 'start', 'y')
    z     = _get(df, shower, 'shw', 'start', 'z')
    x     = _get(df, shower, 'shw', 'start', 'x')
    plane = _get(df, shower, 'shw', 'bestplane')
    if any(v is None for v in (y, z, x, plane)):
        return pd.Series(1.0, index=df.index)

    ybin = _chi2pid._yz_ybin(y,     _chi2pid.yz_ybin_sbnd_mc)
    zbin = _chi2pid._yz_zbin(z,     _chi2pid.yz_zbin_sbnd_mc)
    itpc = pd.Series(np.where(x < 0, 0, 1), index=df.index)
    iov  = pd.Series(0, index=df.index)
    # bestplane may be NaN for missing secondary showers; default to 0
    plane_int = plane.fillna(0).astype(int)

    lookup = pd.DataFrame({
        'ybin': ybin, 'zbin': zbin,
        'itpc': itpc, 'plane': plane_int,
        'iov':  iov,
    })
    scale = (
        lookup
        .merge(_chi2pid.SBND_yz_cal_mc_df,
               on=['iov', 'itpc', 'plane', 'ybin', 'zbin'], how='left')
        .scale
    )
    # replace missing / zero entries with unity
    scale = scale.where(scale.fillna(0) > 1e-6, 1.0).fillna(1.0)
    scale.index = df.index
    return scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_recomb_detvars(
    slc_df: pd.DataFrame,
    showers: list | None = None,
) -> dict:
    """Produce recombination-based detector variation DataFrames.

    For each variation a modified copy of ``slc_df`` is returned.  Multisim
    variations return a two-element list [+1σ, −1σ]; unisim variations return
    a single-element list.  All shower prefixes listed in ``showers`` are
    modified; columns absent from ``slc_df`` are silently skipped so that
    missing secondary-shower entries are handled automatically.

    Parameters
    ----------
    slc_df : pd.DataFrame
        The nuecc analysis DataFrame.
    showers : list of str, optional
        Shower prefixes to modify.  Default: ``['primshw', 'secshw']``.

    Returns
    -------
    dict[str, list[pd.DataFrame]]
        Keys: ``calo_Ccal``, ``calo_alpha``, ``calo_beta90``, ``calo_R``,
              ``calo_phi``, ``calo_yz``, ``calo_Ecorr``.
    """
    _import_deps()
    if showers is None:
        showers = _DEFAULT_SHOWERS

    out: dict = {}
    calo_vars = _chi2pid.CALO_VARIATIONS

    # ----- CALO_VARIATIONS multisim pairs (Ccal, alpha, beta90, R) -----
    for detvar_key, key_p, key_m in _CALO_PAIRS:
        for calo_key in (key_p, key_m):
            var   = calo_vars[calo_key]
            ccal  = var['c_cal_frac'][_CALO_PLANE]
            df    = slc_df
            for sh in showers:
                df = _apply_recomb(df, sh,
                                   dqdx_scale=1.0 / ccal,
                                   alpha=var['alpha_emb'][0],
                                   beta90=var['beta_90'][0],
                                   R=var['R_emb'][0])
                df = _apply_energy_scale(df, sh, scale=1.0 / ccal)
            out.setdefault(detvar_key, []).append(df)

    # ----- phi: use actual shower angle w.r.t. drift (unisim) -----
    df = slc_df
    for sh in showers:
        phi_col = _shower_phi(df, sh)
        if phi_col is not None:
            df = _apply_recomb(df, sh, phi=phi_col)
    out['calo_phi'] = [df]

    # ----- YZ spatial calibration (unisim) -----
    df = slc_df
    for sh in showers:
        yz = _yz_scale(df, sh)
        df = _apply_recomb(df, sh, dqdx_scale=yz)
        df = _apply_energy_scale(df, sh, scale=yz)
    out['calo_yz'] = [df]

    # ----- Ecorr ±3%/1.17: energy scale only, no recombination change -----
    err = 0.03 / 1.17
    for scale in (1.0 + err, 1.0 - err):
        df = slc_df
        for sh in showers:
            df = _apply_energy_scale(df, sh, scale=scale)
        out.setdefault('calo_Ecorr', []).append(df)

    return out
