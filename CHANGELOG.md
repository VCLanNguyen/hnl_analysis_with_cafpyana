# Changelog

A running record of breaking changes and new features. Add a new dated section at the
top each time changes are merged that other users should know about.

---

## 2026-05-11 — HDF5 DetVar store, calorimetry detector variations, integral syst percentages

### Bug fixes

**`syst.py` / `plotting.py` — systematic percentage now bin-count independent**
The fractional uncertainty shown next to each systematic source and in legend labels was
previously computed as the mean of `sqrt(diag(C)) / cv` over bins. This value changes
when the binning changes. It is now computed as `sqrt(sum_ij C_ij) / N_total`, which
equals the fractional uncertainty on the total integral and is independent of how many
bins are used. Category totals are combined in quadrature across sources.

**`syst.py` — DetVar key classification fixed for HDF5 group names**
When loading from an HDF5 store the group names are bare (e.g. `pmtgain`, `calo_Ccal`).
Previously these were not recognised by the `DetVar` category classifier and produced
`Warning: category not found` for every DetVar key. The output keys are now prefixed
with `DetVar_` inside `get_detvar_systs`, matching the expected format.

---

### Breaking changes

**DetVar files are now HDF5 (`.h5`) — pickle (`.pkl`) files are no longer read**
`load_detvar_dicts()` and `get_total_cov()` now use `load_detvar_dict` from the new
`detvar_store` module. The paths in `config.py` have been updated to point at `.h5`
files. If you have local overrides pointing at `.pkl` files, update them:

```python
# Before
nue.load_detvar_dicts(detvar_files=["mydetvars.pkl"])

# After — pass .h5 paths produced by write_detvar_store()
nue.load_detvar_dicts(detvar_files=["mydetvars.h5"])
```

---

### New features

**HDF5-based DetVar store (`detvar_store`)**

Write and read detector-variation DataFrames in a compact HDF5 format.
`load_detvar_dict` automatically applies `preprocess_mc` to every loaded DataFrame
so DV/CV samples receive the same flash PE scaling and derived columns as the main MC:

```python
from nueana.detvar_store import write_detvar_store, load_detvar_dict, detvar_store_info

# Write
write_detvar_store(
    "mydetvars.h5",
    cv_dict={"cv": cv_file},
    dv_dict={"pmtgain": [dv_lo, dv_hi], "lyatt": [dv_ly]},
    cv_map={"pmtgain": "cv", "lyatt": "cv"},
)

# Read back — preprocess_mc applied automatically
detvar_dict = load_detvar_dict("mydetvars.h5")

# Skip preprocessing (e.g. DataFrames were preprocessed before writing)
detvar_dict = load_detvar_dict("mydetvars.h5", preprocess_fn=lambda df: df)

# Inspect
detvar_store_info("mydetvars.h5")
```

`apply_selection` applies a `select()`-style function to every DV/CV DataFrame inside
a loaded dict without rewriting the loading loop:

```python
from nueana.detvar_store import apply_selection
detvar_dict = apply_selection(detvar_dict, nue.select, min_shower_length=10)
```

---

**Software calorimetry detector variations (`detvar_recomb`)**

`make_recomb_detvars` produces modified `slc_df` DataFrames for seven shower-energy
systematics without requiring separate MC samples:

| Variation | Description |
|-----------|-------------|
| `calo_Ccal` | ±2 % overall charge-to-energy calibration scale |
| `calo_alpha` | ±0.008 on recombination parameter A |
| `calo_beta90` | ±0.008 on recombination parameter B₉₀ |
| `calo_R` | ±0.02 on recombination ellipticity R |
| `calo_phi` | Angle-dependent recombination (unisim) |
| `calo_yz` | Spatial YZ calibration map correction (unisim) |
| `calo_Ecorr` | ±3 %/1.17 direct energy scale |

```python
from nueana.detvar_recomb import make_recomb_detvars

dv_dfs = make_recomb_detvars(slc_df)  # dict[str, list[pd.DataFrame]]
```

Multisim variations return `[+1σ df, −1σ df]`; unisim variations return a
single-element list.

---

**`plot_detvar` — overlay DV and CV histograms**

```python
fig, ax_main, ax_ratio = nue.plot_detvar(
    detvar_dict,
    key="calo_Ccal",
    var="primshw.shw.bestplane_dEdx",
    bins=np.linspace(0, 5, 26),
    xlabel="Best-plane dE/dx [MeV/cm]",
)
```

Upper panel shows CV (black) and each DV variation (dashed). Lower panel shows DV/CV
ratio with a reference line at 1.

---

## 2026-05-11 — CutSpec-based selection refactor; preprocess functions

### Bug fixes

**`select()` — non-νe background dropping fixed**
A logic error caused some non-νe MC events to survive the PDG filter. The condition is
now correct, and a guard ensures all PDG stack categories are included in downstream
plots even when a category has zero selected events.

---

### New features

**`preprocess_mc` / `preprocess_data` — apply fixes once before any selection**

Run mandatory column creation and corrections on the raw DataFrame exactly once,
before calling `select()` or any systematic function:

```python
mc_df   = nue.preprocess_mc(mc_df,   pot=mc_pot)
data_df = nue.preprocess_data(data_df)

sel_df  = nue.select(mc_df)
```

Calling these avoids silent inconsistencies when the same raw DataFrame is passed
through multiple selection paths.

---

**`CutSpec` — structured cut management inside `select()`**
Internal refactor; no API change required. `select()` and the systematic functions now
build their cut lists as `CutSpec` objects. The public signature of `select()` is
unchanged; use `modify_cut` / `drop_cuts` to customise the cut sequence.

---

## 2026-05-01 — Cross-section systematics framework, selection improvements, plotting overhaul

### Bug fixes

**`plot_var()` — MC stat uncertainty now detected from the covariance dictionary**
Previously the plotting function always computed MC stat as a separate uncertainty band.
It now scans the passed `systs` dict for an `"MCstat"` key first: if found, MC stat is
already folded into the covariance and a single combined stat+syst band is drawn; if not
found, a separate MC stat band is drawn alongside the systematic band. This prevents
double-counting when `get_total_cov` has already included MC stat.

**`syst.py` — GENIE/xsec knob identification no longer relies on string matching**
Previously, whether a systematic knob followed the cross-section (response-matrix) path
was decided by checking whether the column name contained the string `"GENIE"`. This
misclassified knobs whose names don't include that string (e.g. `SBNNuSyst`, `SuSAv2`).
The check now uses the authoritative lists `regen_systematics` and
`ar23p_genie_systematics` imported directly from `cafpyana`'s `geniesyst.py`, so
classification is exact regardless of naming convention.

---

### Breaking changes

**`select()` — `min_shower_length` default changed from `0.1` → `10` cm**
If your analysis relied on the old default, pass it explicitly:

```python
df_sel = nue.select(df, min_shower_length=0.1)
```

**Category dicts are now nested**
`signal_categories`, `pdg_categories`, `mode_categories`, and `generic_categories`
are now nested dicts of the form `{name: {"value": int, "label": str, "color": str}}`.
The flat `{name: value}` dicts (`signal_dict`, `pdg_dict`, `mode_dict`, `generic_dict`)
are still exported unchanged — use those if you only need integer IDs.

```python
# Before (no longer works for label/color)
color = signal_colors[i]
label = signal_labels[i]

# After
color = nue.signal_categories["nueCC"]["color"]
label = nue.signal_categories["nueCC"]["label"]
value = nue.signal_dict["nueCC"]   # still works as before
```

**Wildcard imports no longer export everything**
All modules now define `__all__`. Replace `import *` with explicit imports:

```python
# Before
from nueana.funcs import *

# After
from nueana.funcs import get_total_cov, load_detvar_dicts, add_flat_norm_uncertainty
```

**`plot_var()` and `plot_mc_data()` return an extra value**
Both functions now return a 4-tuple. Update any unpacking:

```python
# plot_var — was 3-tuple, now 4-tuple
bins, steps, total_err, syst_dict = nue.plot_var(df, var, bins, ax=ax)

# plot_mc_data — was 3-tuple, now 4-tuple
fig, ax_main, ax_sub, mc_dict = nue.plot_mc_data(mc_df, data_df, var, bins)
```

**`SystematicsOutput` and `XSecInputs` are frozen**
These dataclasses are now `frozen=True` and cannot be modified after construction.
Use `dataclasses.replace()` or the `add_*` helpers (see below) to build modified copies.

---

### New features

**Fine-grained selection control with `modify_cut`, `drop_cuts`, and `CutSpec`**

Adjust, remove, or extend cuts without rewriting the full pipeline:

```python
from nueana.selection import DEFAULT_CUTS, modify_cut, drop_cuts, CutSpec

# Skip the shower-length cut entirely
no_len_cuts = drop_cuts(DEFAULT_CUTS, "shower_length")
df_no_len = nue.select(df, cuts=no_len_cuts)

# Add a custom cut on top of the standard selection
custom_cuts = DEFAULT_CUTS + [CutSpec("high_energy", fn=lambda df: df.primshw.shw.reco_energy > 0.8)]
df_custom = nue.select(df, cuts=custom_cuts)

# Tighten an existing cut threshold
tight_cuts = modify_cut(DEFAULT_CUTS, "dedx", min=1.5, max=3.0)
df_tight = nue.select(df, cuts=tight_cuts)
```

Available cut names: `flash_pe`, `nu_score`, `clear_cosmic`, `fiducial_volume`,
`flash_time`, `flash_score`, `shower_energy`, `muon_rejection`, `conversion_gap`,
`dedx`, `opening_angle`, `shower_length`.

---

**Streamlined plot config with `PlottingConfig`**

Bundle display options into a reusable dataclass instead of spelling them out every call:

```python
from nueana.classes import PlottingConfig

signal_cfg = PlottingConfig(
    xlabel=r"$\cos\theta_e$",
    ylabel="Events / bin",
    plot_err=True,
    overflow=True,
)

# Pass as config=; individual kwargs still override
bins, steps, err, systs = nue.plot_var(df, var, bins, ax=ax, config=signal_cfg)

# Override one field for a specific plot without modifying the config
bins, steps, err, systs = nue.plot_var(df, var, bins, ax=ax, config=signal_cfg, normalize=True)
```

---

**Cross-section covariance with `XSecInputs`**

Pass signal truth information to get separate event-rate and cross-section covariance matrices:

```python
from nueana.classes import XSecInputs

xsec_inputs = XSecInputs(
    true_signal_df=mcsig_df,
    true_signal_scale=1 / (nue.integrated_flux * (mc_pot / 1e6)),
    reco_var_true=nue.electron_energy().var_evt_truth_col,
    true_var_true=nue.electron_energy().var_nu_col,
)

output = nue.get_total_cov(
    reco_df=mc_df,
    reco_var=nue.electron_energy().var_evt_reco_col,
    bins=nue.electron_energy().bins,
    mcbnb_pot=mc_pot,
    select_region="signal",
    uncertainty_keys=["xsec", "detv", "norm"],
    xsec_inputs=xsec_inputs,
)

if output.has_xsec:
    print("xsec CV:  ", output.xsec_hist_cv)
    print("xsec cov:\n", output.xsec_cov)

print("rate CV:  ", output.rate_hist_cv)
print("rate cov:\n", output.rate_cov)
```

---

**Selective uncertainty inclusion with `uncertainty_keys`**

Compute only the systematic blocks you need. Allowed keys: `"rate"`, `"xsec"`, `"detv"`,
`"norm"`, `"cosmic"`. Default (when `None`): `{"rate", "detv", "norm", "cosmic"}`, plus
`"xsec"` automatically when `xsec_inputs` is provided.

```python
# Rate systematics + detector variations only
output = nue.get_total_cov(reco_df=mc_df, reco_var=var, bins=bins,
                           mcbnb_pot=mc_pot, uncertainty_keys=["rate", "detv"])

# Norm uncertainties only (fast cross-check)
output = nue.get_total_cov(reco_df=mc_df, reco_var=var, bins=bins,
                           mcbnb_pot=mc_pot, uncertainty_keys=["norm"])
```

---

**Pre-load detector variations to avoid repeated disk reads**

`load_detvar_dicts()` is slow. Load once per session and pass the result to every
`get_total_cov` call:

```python
detvar_dict = nue.load_detvar_dicts()

output_angle = nue.get_total_cov(..., detvar_dict=detvar_dict)
output_energy = nue.get_total_cov(..., detvar_dict=detvar_dict)  # no extra disk read
```

---

**Adding custom uncertainties to `SystematicsOutput`**

```python
from nueana.funcs import add_flat_norm_uncertainty, add_fractional_uncertainty, add_uncertainty

# 2% fully-correlated normalization uncertainty
output = add_flat_norm_uncertainty(output, frac_unc=0.02, key="MyNorm", category="BeamExposure")

# Per-bin fractional uncertainties (uncorrelated bin-to-bin)
frac_unc = np.array([0.05, 0.10, 0.10, 0.08])
output = add_fractional_uncertainty(output, frac_unc=frac_unc, key="MyBinUnc", correlation="diagonal")

# Fully custom covariance matrix
my_cov = np.diag([0.1, 0.2, 0.2, 0.1]) ** 2
output = add_uncertainty(output, cov=my_cov, key="MyCustom", category="MyCategory", target="rate")
```

`target` controls which covariance matrix the source is added to: `"rate"`, `"xsec"`, or `"both"`.

---

**Region-aware detector variations**

```python
output_sig  = nue.get_total_cov(..., select_region="signal")   # default
output_ctrl = nue.get_total_cov(..., select_region="control")
```

---

**Event masking**

Filter a selected dataframe to signal-only or background-only events
(requires `define_signal` to have been called first):

```python
from nueana.utils import apply_event_mask

df_signal_only     = apply_event_mask(df, "signal")      # signal == 0
df_background_only = apply_event_mask(df, "background")  # signal != 0
df_all             = apply_event_mask(df, "all")          # no filter
```

---

**Data/MC ratio panel and chi-squared annotation in `plot_mc_data()`**

`plot_mc_data()` now annotates the main axis with the integrated Data/MC ratio and a chi-squared
goodness-of-fit test (using `scipy.stats.chi2` when available). Both can be suppressed:

```python
fig, ax_main, ax_sub, mc_dict = nue.plot_mc_data(
    mc_df, data_df, var, bins,
    annot=False,          # suppress Data/MC and chi-sq text
    ratio_min=0.5,        # customize ratio panel y-limits
    ratio_max=1.5,
)

# The returned mc_dict includes the chi-sq value for downstream use
chi2  = mc_dict["chi2"]
p_val = mc_dict["p_value"]
ratio = mc_dict["ratio"]      # integrated Data/MC
```

---

**Systematics breakdown plots**

```python
fig, axes, angle_summary, energy_summary = nue.plot_syst_category_breakdown(
    angle_syst_df=output_angle.rate_syst_df,
    energy_syst_df=output_energy.rate_syst_df,
    category_dict=nue.category_dict_signal,
    angle_var=r"$\cos\theta_e$",
    energy_var=r"$E_e$ [GeV]",
    region_label="Signal Region",
)

# Drill into a single category
fig, axes = nue.plot_syst_breakdown(
    angle_syst_df=output_angle.rate_syst_df,
    energy_syst_df=output_energy.rate_syst_df,
    category="GENIE",
    category_dict=nue.category_dict_signal,
)
```
