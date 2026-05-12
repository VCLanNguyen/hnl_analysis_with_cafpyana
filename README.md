# nueana

Utilities for SBND CCnue cross-section analysis.

This package is designed for notebook and script workflows where CAF-derived dataframes
have already been produced (typically via `cafpyana`) and you want to run selection,
plotting, and uncertainty studies.

## What this package provides

- Selection helpers for signal and sideband studies.
- Signal-category definitions for MC truth labeling.
- Histogram utilities with overflow handling.
- Plotting helpers for stacked MC, data overlays, and data/MC ratios.
- Systematic uncertainty tools (covariance, correlation, universe handling, detector-variation helpers).
- I/O helpers for split HDF5 dataframe files (output of cafpyana).
- Common constants and geometry utilities.

## Package layout

- `config.py`: Global paths and environment setup (cafpyana path, flux file, detvar files).
- `constants.py`: Signal/background category dicts, physics constants, flux values.
- `classes.py`: Core dataclasses for analysis variables, cross-section inputs, systematics results, and plot configuration.
- `variables.py`: Pre-built analysis variable definitions (bins, column names, labels).
- `selection.py`: Event selection pipeline and MC truth signal labeling.
- `preprocess.py`: Idempotent preprocessing fixes for MC and data (flash PE scaling, column renames, derived columns).
- `plotting.py`: Stacked MC, data overlay, data/MC ratio, systematics breakdown, and DetVar comparison plots.
- `funcs.py`: High-level systematics driver — total covariance calculation and custom uncertainty helpers.
- `syst.py`: Low-level systematics — universe histograms, covariance matrices, detector variations.
- `detvar_store.py`: HDF5-based DetVar store — write, load, inspect, and apply selection to DetVar dictionaries.
- `detvar_recomb.py`: Software-based calorimetry detector variations derived from recombination parameter shifts.
- `histogram.py`: Histogram wrappers with overflow handling.
- `utils.py`: DataFrame helpers for MultiIndex sorting, header merging, and event masking.
- `io.py`: Split-HDF5 dataframe loading.
- `geometry.py`: Detector geometry checks.

## Quickstart for a new analysis

There are four things to update when adapting this package to a different signal or
selection: the paths in `config.py`, the signal categories in `constants.py`, the
selection cuts in `selection.py`, and the analysis variables in `variables.py`.

### 1. Update paths — `config.py`

Set the paths for your environment before importing anything else:

```python
import nueana.config as config

config.CAFPYANA_PATH  = "/path/to/your/cafpyana"
config.FLUX_FILE      = "/path/to/your/flux.root"
config.INTIME_FILE    = "/path/to/your/intime.df"
config.DETVAR_DICT_SIGNAL  = "/path/to/your/detvar_signal.h5"
config.DETVAR_DICT_CONTROL = "/path/to/your/detvar_control.h5"
```

> **Note:** `INTIME_FILE`, `DETVAR_DICT_SIGNAL`, and `DETVAR_DICT_CONTROL` are only
> used by the systematic uncertainty functions (`get_total_cov`, `get_intime_cov`,
> `get_detvar_systs`). If you are not yet running systematics, these paths can be
> left as-is. `FLUX_FILE` is the exception — it is read at import time by
> `constants.py`, so it must point to a valid file before `import nueana` is called.
>
> **DetVar files must be in HDF5 format (`.h5`).** Pickle (`.pkl`) files are no longer
> supported. Use `write_detvar_store()` from `detvar_store.py` to create `.h5` files.

### 2. Define your signal categories — `constants.py`

`signal_categories` drives the integer labels written by `define_signal()` and the
colors/labels used by `plot_var()`. Each entry needs a `"value"` (integer ID),
`"label"` (legend text), and `"color"`. By convention, the signal topology has ID `0`.

```python
# In constants.py — replace or extend signal_categories with your own channels
signal_categories = {
    "mySignal":   {"value": 0,  "label": r"My Signal",  "color": "steelblue"},
    "background1":{"value": 1,  "label": "Background 1","color": "tomato"},
    "nonFV":      {"value": 10, "label": "Non-FV",      "color": "gray"},
    "dirt":       {"value": 11, "label": "Dirt",        "color": "peru"},
    "cosmic":     {"value": 12, "label": "Cosmic",      "color": "orchid"},
    "offbeam":    {"value": 13, "label": "Off-beam",    "color": "silver"},
}
signal_dict = {k: v["value"] for k, v in signal_categories.items()}
```

Then update `define_signal()` in `selection.py` to assign your integer IDs.

### 3. Adjust selection cuts — `selection.py`

Pass cut overrides directly to `select()` without touching the source, or skip cuts
entirely:

```python
import nueana as nue
from nueana.selection import DEFAULT_CUTS, modify_cut, drop_cuts, CutSpec

# Tighten or loosen a cut threshold
tight_cuts = modify_cut(DEFAULT_CUTS, "shower_energy", min=0.3)
df_sel = nue.select(df, cuts=tight_cuts)

# Skip a cut that doesn't apply to your topology
no_muon_cuts = drop_cuts(DEFAULT_CUTS, "muon_rejection")
df_sel = nue.select(df, cuts=no_muon_cuts)

# Add a custom cut on top of the standard pipeline
custom_cuts = DEFAULT_CUTS + [CutSpec("my_cut", fn=lambda df: df.primshw.shw.open_angle < 0.1)]
df_sel = nue.select(df, cuts=custom_cuts)
```

`select()` can return all intermediate stages for cut-flow studies:

```python
stages = nue.select(df, savedict=True)
# Keys: 'flash_pe', 'nu_score', 'clear_cosmic', 'fiducial_volume',
#       'flash_time', 'flash_score', 'shower_energy', 'muon_rejection',
#       'conversion_gap', 'dedx', 'opening_angle', 'shower_length'

# Or stop at a specific stage
df_preflash = nue.select(df, stage="flash_pe")
```

### 4. Define your analysis variables — `variables.py`

Add a factory function returning a `VariableConfig` for each variable you want to unfold:

```python
from nueana.classes import VariableConfig
import numpy as np

def my_variable() -> VariableConfig:
    return VariableConfig(
        var_save_name  = "my_var",
        var_plot_name  = r"$p_T$",
        var_unit       = "GeV",
        bins           = np.array([0.0, 0.2, 0.5, 1.0, 2.0]),
        bin_labels     = np.array([0.0, 0.2, 0.5, 1.0, 2.0]),
        var_evt_reco_col   = ("primshw", "shw", "my_reco_col", "", "", ""),
        var_evt_truth_col  = ("slc", "truth", "e", "my_truth_col"),
        var_nu_col         = ("e", "my_truth_col"),
    )
```

### Minimal working example

```python
import numpy as np
import nueana as nue
from nueana.classes import PlottingConfig

# Load CAF-derived dataframes
mc_dfs   = nue.load_dfs("/path/to/mc.df",   ["mcnu", "hdr", "nuecc"])
data_dfs = nue.load_dfs("/path/to/data.df", ["hdr",  "nuecc"])

# Preprocess raw DataFrames (column fixes, corrections) — do this once before any selection
mc_raw   = nue.preprocess_mc(mc_dfs["nuecc"])
data_raw = nue.preprocess_data(data_dfs["nuecc"])

# Run selection and label signal categories
mc_df   = nue.define_signal(nue.select(mc_raw,   savedict=False))
data_df = nue.select(data_raw, savedict=False)

# Make a stacked MC + data plot
cfg = PlottingConfig(xlabel="Reco shower energy [GeV]", plot_err=True)
fig, ax_main, ax_sub, mc_dict = nue.plot_mc_data(
    mc_df=mc_df, data_df=data_df,
    var=nue.electron_energy().var_evt_reco_col,
    bins=nue.electron_energy().bins,
    config=cfg,
)
```

## Preprocessing raw DataFrames

Call `preprocess_mc` / `preprocess_data` on the raw CAF-derived DataFrame **before**
any call to `select()` or the systematic functions. They create derived columns (e.g.
`reco_energy`) and apply any necessary corrections exactly once, preventing silent
inconsistencies when the same raw DataFrame is reused across multiple selection paths.

```python
mc_raw   = nue.preprocess_mc(mc_dfs["nuecc"], pot=mc_pot)
data_raw = nue.preprocess_data(data_dfs["nuecc"])

# All downstream calls operate on the preprocessed frames
sel_df         = nue.select(mc_raw)
detvar_output  = nue.get_total_cov(mc_raw, ..., uncertainty_keys=["detv"])
```

## Working with detector variations

### Loading from an HDF5 store

DetVar files are HDF5 (`.h5`). Load them once per session and reuse across variables.
`load_detvar_dict` automatically applies `preprocess_mc` to every loaded DataFrame
so that DV/CV samples receive identical treatment to the main MC:

```python
detvar_dict = nue.load_detvar_dicts()   # reads paths from config.py

# Or load a specific file
from nueana.detvar_store import load_detvar_dict
detvar_dict = load_detvar_dict("/path/to/mydetvars.h5")

# Inspect what's inside
from nueana.detvar_store import detvar_store_info
detvar_store_info("/path/to/mydetvars.h5")
```

To skip preprocessing (e.g. the DataFrames were already preprocessed before writing):

```python
detvar_dict = load_detvar_dict("/path/to/mydetvars.h5", preprocess_fn=lambda df: df)
```

Apply a custom selection to every DV/CV DataFrame with `apply_selection`. Pass a
modified cut list via `cuts` rather than keyword arguments, since `select()` takes
cut sequences rather than per-parameter overrides:

```python
from nueana.detvar_store import apply_selection
from nueana.selection import modify_cut, DEFAULT_CUTS

tight_cuts = modify_cut(DEFAULT_CUTS, "shower_length", min=15)
detvar_dict = apply_selection(detvar_dict, nue.select, cuts=tight_cuts)
```

### Building a new HDF5 store

```python
from nueana.detvar_store import write_detvar_store

write_detvar_store(
    "mydetvars.h5",
    cv_dict={"cv": cv_file},
    dv_dict={"pmtgain": [dv_lo, dv_hi], "lyatt": [dv_ly]},
    cv_map={"pmtgain": "cv", "lyatt": "cv"},
)
```

### Software calorimetry variations

`make_recomb_detvars` derives shower-energy systematics from the raw `slc_df` without
separate MC samples. Pass the result to `write_detvar_store` to persist them:

```python
from nueana.detvar_recomb import make_recomb_detvars

dv_dfs = make_recomb_detvars(slc_df)
# dv_dfs is dict[str, list[pd.DataFrame]]
# Keys: calo_Ccal, calo_alpha, calo_beta90, calo_R, calo_phi, calo_yz, calo_Ecorr

dv_files = {name: [cv._replace(slc_df=df) for df in dfs]
            for name, dfs in dv_dfs.items()}
write_detvar_store("recomb_detvars.h5",
    cv_dict={"cv": cv},
    dv_dict=dv_files,
    cv_map={name: "cv" for name in dv_files},
)
```

### Comparing DV and CV histograms

```python
fig, ax_main, ax_ratio = nue.plot_detvar(
    detvar_dict,
    key="calo_Ccal",
    var="primshw.shw.bestplane_dEdx",
    bins=np.linspace(0, 5, 26),
    xlabel="Best-plane dE/dx [MeV/cm]",
)
```

## Notes and caveats

- `constants.py` reads the flux ROOT file at import time. If the file is unavailable,
  importing `nueana` will fail.
- Several utilities assume pandas MultiIndex columns with CAF-style naming.
- Many routines expect a `signal` column to already be present — call `define_signal()`
  or `define_generic()` before plotting or applying event masks.
- Overflow is enabled by default (`overflow=True`) and folds out-of-range values into
  the first/last bin.
