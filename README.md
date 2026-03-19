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
- Systematic uncertainty tools (covariance, correlation, universe handling,
	detector-variation helpers).
- I/O helpers for split HDF5 dataframe files (output of cafpyana).
- Common constants and geometry utilities.

## Package layout

- `selection.py`: Event selection and signal labeling.
	- `select`, `select_sideband`
	- `define_signal`, `define_generic`
- `plotting.py`: MC/data plotting utilities.
	- `plot_var`, `plot_var_pdg`
	- `data_plot_overlay`, `plot_mc_data`
- `syst.py`: Systematic uncertainty utilities.
	- `get_syst_hists`, `get_syst`
	- `calc_matrices`, `get_detvar_systs`, `get_syst_df`
	- `mcstat`
- `histogram.py`: Histogram wrappers with overflow behavior.
	- `get_hist1d`, `get_hist2d`
- `io.py`: Split-HDF5 dataframe loading and exposure summaries.
	- `get_n_split`, `print_keys`, `load_dfs`
- `geometry.py`: Detector geometry checks.
	- `whereTPC`
- `constants.py`: Analysis dictionaries and default plotting metadata.
- `utils.py`: DataFrame helpers.
	- `ensure_lexsorted`

## constants.py reference

The objects in `constants.py` are used by selection/plotting/systematics code to keep
category definitions and display metadata consistent. All signal colors, labels, and names can be user-specific. `signal_dict` will be called by the signal/background definition function in `selection.py`

- `signal_dict`
	- Maps detailed analysis channels to integer IDs used in the `signal` column.
	- Convention: ID `0` (`nueCC`) is the signal topology.
	- Includes backgrounds such as `numuCCpi0`, `NCpi0`, `nonFV`, `dirt`, `cosmic`, and `offbeam`.

- `signal_labels`
	- Human-readable LaTeX labels corresponding to `signal_dict` order.
	- Used in stacked MC legends (for example by `plot_var`).

- `signal_colors`
	- Default color palette corresponding to `signal_dict`/`signal_labels` order.
	- Keeps category colors stable across plots.

- `generic_dict`
	- Coarser category mapping (`CCnu`, `NCnu`, `nonFV`, `dirt`, `cosmic`).
	- Used when plotting/labeling with the generic categorization mode.

- `generic_labels`, `generic_colors`
	- Display labels and default colors for the generic categories.

Important behavior:

- Because flux is read at import time in `constants.py`, importing `nueana` requires
	that `fluxfile` exists and is readable in the current environment.

## Environment setup

Most users should add `cafpyana` to their path as well. In a notebook: 

```
import nueana
import sys
sys.path.append("<path/to>/cafpyana")
```

## Quick start

```python
import numpy as np
import nueana as nue

file = "/path/to/cafpyana/df.df"
keys = ['mcnu','hdr','nuecc']
mc_dfs= = nue.load_dfs(file,keys)
# df_mc should be a CAF-style pandas DataFrame with expected columns.
df_final = nue.select(mc_dfs['nuecc'], savedict=False)

# Define signal labels for MC.
df_labeled = nue.define_signal(df_final)

# Make a basic stacked plot.
bins = np.linspace(0.0, 2.0, 21)
fig, ax_main, ax_sub, _ = nue.plot_mc_data(
		mc_df=df_labeled,
		data_df=df_data,
		var=("primshw", "shw", "reco_energy", "", "", ""),
		bins=bins,
		xlabel="Reco shower energy [GeV]",
		title="nue candidate selection",
)
```

## Selection stages

`select(...)` can return either all intermediate stages as a dictionary(`savedict=True`) or only the
final dataframe (`savedict=False`). Default stage names for the nueCC selection:

- `preselection`
- `flash matching`
- `shower energy`
- `muon rejection`
- `conversion gap`
- `dEdx`
- `opening angle`
- `shower length`
- `shower density`

Use `stage="..."` to stop at an intermediate point.

## Notes and caveats

- `constants.py` reads the flux ROOT file at import time. If the file is unavailable,
	importing `nueana.constants` (or full `nueana`) will fail.
- Several utilities assume pandas MultiIndex columns with CAF-style naming.
- Many routines expect a `signal` column to already be defined (use
	`define_signal` or `define_generic` where appropriate).
- Overflow behavior in histograms is enabled by default (`overflow=True`) and folds
	values into the last bin.