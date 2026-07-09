"""Global configuration and paths for nueana module."""

import sys
from pathlib import Path

# ========================
# Directory Paths
# ========================

# Root directories
NUEANA_DIR    = "/exp/sbnd/data/users/lnguyen/xsection/cafpyana/hnl_analysis_with_cafpyana"
CAFPYANA_PATH = "/exp/sbnd/data/users/lnguyen/cafpyana_pi0/cafpyana"

# Setup cafpyana - append to sys.path so we can import cafpyana modules
if CAFPYANA_PATH not in sys.path:
    sys.path.append(CAFPYANA_PATH)
    sys.path.append(CAFPYANA_PATH+"/analysis_village")

# ========================
# Data and File Paths
# ========================

# Flux file path
FLUX_FILE = "/exp/sbnd/data/users/lynnt/xsection/flux/sbnd_original_flux.root"

# In-time cosmic sample file path
INTIME_FILE = "/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/dfs_nu26/mc_intime.df"

# Detector variation (detvar) dictionaries path
# List of pickle files to load and combine for detector variations
# NOTE: this is Lynn's own nueCC-topology store (built with her DEFAULT_CUTS/
# SIDEBAND_CUTS). Not usable as-is for HNL selection -- see HNL_DETVAR_* below.
DETVAR_DICT_DIR = "/exp/sbnd/data/users/lynnt/xsection/samples/MCP2025B_v10_06_00_09/dfs_nu26/detvars"
DETVAR_DICT_FILES = [DETVAR_DICT_DIR + "/detvars.h5",]
DETVAR_DICT_SIGNAL = DETVAR_DICT_DIR + "/detvars_signal.h5"
DETVAR_DICT_CONTROL = DETVAR_DICT_DIR + "/detvars_sideband.h5"

# HNL-topology detvar store (produced via hnl_mcnu_detvar.py + process_detvars.py
# --slc-key rec -s preprocess), built from the concatenated detvar_cv_0/
# detvar_0p94xly_0/detvar_1p19xly_0 samples in dataframes/July2026/. Since HNL has
# no fixed DEFAULT_CUTS-style cut sequence (cuts are built per-mode via
# select_by_mode() in the notebook), only the preprocess-only store is used;
# apply the real HNL cuts at analysis time via
# get_total_cov(..., cuts=<hnl_cuts>, select_region="all").
HNL_DETVAR_DIR = "/exp/sbnd/data/users/lnguyen/cafpyana_pi0/dataframes/July2026/detvar"
HNL_DETVAR_DICT_FILES = [HNL_DETVAR_DIR + "/detvars.h5",]

# ========================
# Path Verification (Optional)
# ========================

def _verify_path(path, name):
    """Verify that a path exists and is accessible."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found at: {path}")
    return path

# Set to True for debugging to verify all critical paths exist
VERIFY_PATHS = False

if VERIFY_PATHS:
    _verify_path(CAFPYANA_PATH, "CAFPYANA_PATH")
    _verify_path(FLUX_FILE, "FLUX_FILE")
    _verify_path(INTIME_FILE, "INTIME_FILE")
    for i, detvar_file in enumerate(DETVAR_DICT_FILES):
        _verify_path(detvar_file, f"DETVAR_DICT_FILES[{i}]")
