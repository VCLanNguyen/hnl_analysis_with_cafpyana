"""Kinematic variable computation for slice-level DataFrames."""

import numpy as np

from .utils import ensure_lexsorted


def add_variables(df, beam_x: float = -74.0, beam_y: float = 0.0):
    """Add derived kinematic columns to a slice-level DataFrame from make_topo_df.

    Columns added:

    Per shower (primshw / secshw if present):
      (shw, 'shw', 'angle_z', '', '', '')          -- angle w.r.t. beam axis z [deg]

    Slice-level:
      ('slc', 'vertex', 'transverse_distance_beam_2', '', '', '')
                                                    -- d_T^2 = (vtx_x-beam_x)^2 + (vtx_y-beam_y)^2 [cm^2]


    Two-shower only (when secshw columns are present):
      ('slc', 'm_alt', '', '', '', '')              -- transverse mass of the shower pair [MeV]
                                                       m_alt = sqrt(2*ET1*ET2*(1-cos_theta))*1000

    Parameters
    ----------
    df : pd.DataFrame
        Slice-level DataFrame produced by topology.make_topo_df.
    beam_x, beam_y : float
        Beam centre x and y position [cm]. Default: (-74, 0).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new columns appended.
    """
    df = ensure_lexsorted(df, axis=1)

    def _angle_z(shw):
        dx = df[(shw, 'shw', 'dir', 'x', '', '')].values
        dy = df[(shw, 'shw', 'dir', 'y', '', '')].values
        dz = df[(shw, 'shw', 'dir', 'z', '', '')].values
        n  = np.sqrt(dx**2 + dy**2 + dz**2)
        with np.errstate(invalid='ignore'):
            return np.degrees(np.arccos(np.clip(dz / np.where(n > 0, n, np.nan), -1, 1)))

    for shw in ('primshw', 'secshw'):
        if (shw, 'shw', 'dir', 'z', '', '') in df.columns:
            df[(shw, 'shw', 'angle_z', '', '', '')] = _angle_z(shw)

    vtx_x_col = ('slc', 'vertex', 'x', '', '', '')
    vtx_y_col = ('slc', 'vertex', 'y', '', '', '')
    vtx_z_col = ('slc', 'vertex', 'z', '', '', '')
    if vtx_x_col in df.columns and vtx_y_col in df.columns:
        df[('slc', 'vertex', 'transverse_distance_beam_2', '', '', '')] = (
            (df[vtx_x_col].values - beam_x) ** 2 +
            (df[vtx_y_col].values - beam_y) ** 2
        )

    has_prim = ('primshw', 'shw', 'bestplane_energy', '', '', '') in df.columns
    has_sec  = ('secshw',  'shw', 'bestplane_energy', '', '', '') in df.columns

    if has_prim and has_sec:
        E1 = df[('primshw', 'shw', 'bestplane_energy', '', '', '')].values
        E2 = df[('secshw',  'shw', 'bestplane_energy', '', '', '')].values

        def _unit(shw):
            dx = df[(shw, 'shw', 'dir', 'x', '', '')].values
            dy = df[(shw, 'shw', 'dir', 'y', '', '')].values
            dz = df[(shw, 'shw', 'dir', 'z', '', '')].values
            n  = np.sqrt(dx**2 + dy**2 + dz**2)
            n  = np.where(n > 0, n, np.nan)
            return dx/n, dy/n, dz/n

        ux1, uy1, uz1 = _unit('primshw')
        ux2, uy2, uz2 = _unit('secshw')

        with np.errstate(invalid='ignore'):
            ET1 = E1 * np.sqrt(np.clip(1 - uz1**2, 0, None))
            ET2 = E2 * np.sqrt(np.clip(1 - uz2**2, 0, None))
            cos_theta = np.clip(ux1*ux2 + uy1*uy2 + uz1*uz2, -1, 1)
            df[('slc', 'm_alt', '', '', '', '')] = (
                np.sqrt(2 * ET1 * ET2 * (1 - cos_theta)) * 1000  # GeV -> MeV
            )

    return df
