# nueana/detvar

Detector variation (DetVar) utilities for SBND nue CC analysis.

## Modules

| File | Purpose |
|---|---|
| `store.py` | Read/write DetVar HDF5 stores (`prepare_detvar_df`, `write_detvar_store`, `load_detvar_dict`) |
| `recomb.py` | Software recombination variations (`make_recomb_detvars`) |
| `process_detvars.py` | Script to build stores from a directory of raw `.df` files |

---

## Input directory structure

`process_detvars.py` scans an input directory for files matching
`detvar_<name>_<idx>.df`. The `<name>` field determines the role:

- **`cv`** — Central Value sample (e.g. `detvar_cv_0.df`, `detvar_cv_1.df`)
- **anything else** — Detector variation sample (e.g. `detvar_pmtgain_0.df`)

The `<idx>` suffix identifies which CV a DV pairs with. Each DV is automatically 
matched to the CV sharing the same `<idx>` (e.g. `detvar_pmtgain_1.df` → `detvar_cv_1.df`). 
Different DVs can pair with different CVs by using the appropriate suffix.

### Example layout

```
/path/to/dfs/
  detvar_cv_0.df
  detvar_cv_1.df
  detvar_pmtgain_0.df       # paried with cv_0
  detvar_wiremodx_1.df      # paired with cv_1
  detvar_lyattenuation_0.df
```

Recombination variations (`calo_Ccal`, `calo_alpha`, `calo_beta90`,
`calo_R`, `calo_phi`, `calo_yz`, `calo_Ecorr`) are computed automatically
from the reference CV and do not need separate input files.

---

## Running process_detvars.py

The `signal` and `sideband` selections use `DEFAULT_CUTS` and `SIDEBAND_CUTS`
from `nueana/analysis.py`. To change which cuts are applied, update those
sequences there before running the script.

```bash
# Signal + sideband stores (default)
python process_detvars.py -i /path/to/dfs/ -o /path/to/output/

# All four selection stages
python process_detvars.py -i /path/to/dfs/ -o /path/to/output/ -s all

# Preprocessing only (no cuts)
python process_detvars.py -i /path/to/dfs/ -o /path/to/output/ -s preprocess

# Use a specific CV as the reference for DV pairing and recomb variations
python process_detvars.py -i /path/to/dfs/ -o /path/to/output/ --cv-key cv_1

# Recompute only specific groups and patch them into an existing store
python process_detvars.py -i /path/to/dfs/ -o /path/to/output/ -g recomb_lo recomb_hi wiremodx_0
```

### Output files

| Selection flag | Output file |
|---|---|
| `preprocess` | `detvars.h5` |
| `signal` | `detvars_signal.h5` |
| `sideband` | `detvars_sideband.h5` |
| `preselect` | `detvars_preselect.h5` |

---

## Loading a store at analysis time

```python
import nueana as nue

# Load all groups
detvar_dict = nue.load_detvar_dict("detvars_signal.h5")

# Load a subset
detvar_dict = nue.load_detvar_dict("detvars_signal.h5", groups=["pmtgain", "wiremodx"])

# Each entry: {'dv_df': DataFrame (or list), 'cv_df': DataFrame, 'pot': float}
entry = detvar_dict["pmtgain"]
cv_df = entry["cv_df"]
dv_df = entry["dv_df"]
pot   = entry["pot"]

# Inspect store metadata
nue.detvar_store_info("detvars_signal.h5")
```

## Store format

The HDF5 store has the following key layout:

```
/meta              DataFrame: group, cv_key, n_dv, pot
/cv/<cv_key>       Full CV nuecc DataFrame (stored once per unique CV)
/cv_iloc/<group>   int64 array of iloc positions into /cv/<cv_key>
/dv/<group>/v0     DV nuecc DataFrame (matched to CV intersection)
/dv/<group>/v1     Second DV universe for multisim groups
```
