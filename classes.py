from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# class VariableConfig:
#     """
#     A configurable class for setting up unfolding variable configurations.
#     Choose a configuration using one of the provided class methods,
#     or instantiate directly with custom parameters.
#     """

#     def __init__(
#         self,
#         var_save_name: str,
#         var_plot_name: str,
#         var_unit: str,
#         bins: np.ndarray,
#         reco_var_reco: str | tuple,
#         reco_var_true: str | tuple,
#         true_var_true: str | tuple,
#     ):
#         self.var_save_name = var_save_name
#         self.var_plot_name = var_plot_name
#         self.var_unit = var_unit
#         unit_suffix = f"~[{var_unit}]" if len(var_unit) > 0 else ""
#         self.var_labels = [
#             r"$\mathrm{" + var_plot_name + unit_suffix + "}$",
#             r"$\mathrm{" + var_plot_name + "^{reco.}" + unit_suffix + "}$",
#             r"$\mathrm{" + var_plot_name + "^{true}" + unit_suffix + "}$",
#         ]
#         self.bins = bins
#         self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
#         self.reco_var_reco = reco_var_reco
#         self.reco_var_true = reco_var_true
#         self.true_var_true = true_var_true

#     @classmethod
#     def electron_energy(cls) -> "VariableConfig":
#         return cls(
#             var_save_name="electron-E",
#             var_plot_name=r"E_e",
#             var_unit="GeV",
#             bins=np.linspace(0.15, 1.2, 11),
#             reco_var_reco=('primshw','shw','reco_energy','','',''),
#             reco_var_true=('slc','truth','e','genE','',''),
#             true_var_true=("e", "genE", ""),
#         )

#     @classmethod
#     def from_dict(cls, d: dict) -> "VariableConfig":
#         """Instantiate from a plain dictionary, e.g. loaded from a config file."""
#         return cls(
#             var_save_name=d["save_name"],
#             var_plot_name=d["plot_name"],
#             var_unit=d["unit"],
#             bins=np.array(d["bins"]),
#             reco_var_reco=d["reco_col"],
#             reco_var_true=d["truth_col"],
#             true_var_true=d["nu_col"],
#         )

@dataclass(frozen=True)
class XSecInputs:
    """
    Run-level inputs for cross-section unfolding.
    Column references live on VariableConfig; only truth-signal
    information that is independent of the choice of variable belongs here.
    """

    true_signal_df: pd.DataFrame
    true_signal_scale: float
    reco_var_true: str | tuple
    true_var_true: str | tuple


@dataclass(frozen=True)
class SystematicsOutput:
    """
    Results of a systematics evaluation for a single variable.
    xsec_* fields are optional; check .has_xsec before accessing them.
    """

    hist_cv: np.ndarray
    rate_cov: np.ndarray
    rate_syst_df: pd.DataFrame
    rate_syst_dict: dict
    xsec_cov: np.ndarray | None = None
    xsec_syst_df: pd.DataFrame | None = None
    xsec_syst_dict: dict | None = None

    @property
    def has_xsec(self) -> bool:
        """True if cross-section covariance was computed."""
        return self.xsec_cov is not None


# class UnfoldingResult:
#     """
#     Ties a VariableConfig, XSecInputs, and SystematicsOutput together
#     so that variable metadata (bins, labels, column refs) is always
#     in scope alongside the numerical results.
#     """

#     def __init__(
#         self,
#         variable: VariableConfig,
#         inputs: XSecInputs,
#         systematics: SystematicsOutput,
#     ):
#         self.variable = variable
#         self.inputs = inputs
#         self.systematics = systematics

#     def covariance(
#         self, mode: Literal["rate", "xsec"] = "rate"
#     ) -> np.ndarray:
#         """
#         Return the covariance matrix for the requested mode.

#         Parameters
#         ----------
#         mode : {"rate", "xsec"}
#             "rate" returns the event-rate covariance (always available).
#             "xsec" returns the cross-section covariance; raises if not computed.
#         """
#         if mode == "xsec":
#             if not self.systematics.has_xsec:
#                 raise ValueError(
#                     f"Cross-section covariance not computed for "
#                     f"'{self.variable.var_save_name}'. "
#                     "Re-run systematics with xsec=True."
#                 )
#             return self.systematics.xsec_cov
#         return self.systematics.rate_cov

#     def syst_breakdown(
#         self, mode: Literal["rate", "xsec"] = "rate"
#     ) -> pd.DataFrame:
#         """
#         Return the per-systematic breakdown DataFrame for the requested mode.

#         Parameters
#         ----------
#         mode : {"rate", "xsec"}
#         """
#         if mode == "xsec":
#             if not self.systematics.has_xsec:
#                 raise ValueError(
#                     f"Cross-section systematics not available for "
#                     f"'{self.variable.var_save_name}'."
#                 )
#             return self.systematics.xsec_syst_df
#         return self.systematics.rate_syst_df