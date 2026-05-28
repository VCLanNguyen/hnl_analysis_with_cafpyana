"""
Numerical equivalence check: current get_xsec_hists vs proposed vectorized version.
Self-contained — no nueana import needed.

Run with:
    /exp/sbnd/data/users/lynnt/xsection/.venv/bin/python \
        nueana/tests/test_xsec_hists_vectorized.py
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inline helpers (mirrors nueana.utils)
# ---------------------------------------------------------------------------
def get_hist1d(weights=None, data=None, bins=None, overflow=True):
    if weights is None:
        weights = np.ones(len(data))
    if overflow:
        cleaned = np.nan_to_num(data, nan=bins[-1]-1e-10, posinf=bins[-1]-1e-10, neginf=bins[0])
        clipped = np.clip(cleaned, bins[0], bins[-1]-1e-10)
        return np.histogram(clipped, bins=bins, weights=weights)[0]
    return np.histogram(data, bins=bins, weights=weights)[0]

def get_hist2d(weights=None, x=None, y=None, bins=None, overflow=True):
    x_bins = y_bins = bins
    if weights is None:
        weights = np.ones(len(x))
    if overflow:
        cx = np.clip(np.nan_to_num(x, nan=x_bins[-1]-1e-10, posinf=x_bins[-1]-1e-10, neginf=x_bins[0]), x_bins[0], x_bins[-1]-1e-10)
        cy = np.clip(np.nan_to_num(y, nan=y_bins[-1]-1e-10, posinf=y_bins[-1]-1e-10, neginf=y_bins[0]), y_bins[0], y_bins[-1]-1e-10)
        return np.histogram2d(cx, cy, bins=[x_bins, y_bins], weights=weights)[0]
    return np.histogram2d(x, y, bins=[x_bins, y_bins], weights=weights)[0]

# ---------------------------------------------------------------------------
# Current implementation (mirrors nueana.syst.get_xsec_hists)
# ---------------------------------------------------------------------------
def get_xsec_hists_current(reco_df, true_signal_df, true_signal_scale,
                            reco_var_reco, reco_var_true, true_var_true,
                            reco_weights, true_signal_weights, bins):
    true_signal_df = true_signal_df[true_signal_df['signal'] == 0]
    signal_mask = reco_df['signal'] == 0

    smearing = np.apply_along_axis(
        get_hist2d, 0,
        reco_weights[signal_mask],
        reco_df[signal_mask][reco_var_reco].to_numpy(),
        reco_df[signal_mask][reco_var_true].to_numpy(),
        bins)

    sig_hist_univ = np.apply_along_axis(
        get_hist1d, 0,
        true_signal_weights,
        true_signal_df[true_var_true].to_numpy(), bins)

    sig_hist_cv = get_hist1d(
        np.ones(len(true_signal_df)) * true_signal_scale,
        true_signal_df[true_var_true].to_numpy(), bins)

    response = np.divide(smearing, sig_hist_univ,
                         out=np.zeros_like(smearing), where=sig_hist_univ > 0)
    sig_reco_hists = np.einsum('ijk,j->ik', response, sig_hist_cv)

    bkg_reco_hists = np.apply_along_axis(
        get_hist1d, 0,
        reco_weights[~signal_mask],
        reco_df[~signal_mask][reco_var_reco].to_numpy(), bins)

    return sig_reco_hists + bkg_reco_hists, response

# ---------------------------------------------------------------------------
# Proposed vectorized implementation
# ---------------------------------------------------------------------------
def get_xsec_hists_vectorized(reco_df, true_signal_df, true_signal_scale,
                               reco_var_reco, reco_var_true, true_var_true,
                               reco_weights, true_signal_weights, bins):
    n_bins = len(bins) - 1
    true_signal_df = true_signal_df[true_signal_df['signal'] == 0]
    signal_mask = reco_df['signal'] == 0

    def _clip(arr):
        arr = np.asarray(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=bins[-1]-1e-10, posinf=bins[-1]-1e-10, neginf=bins[0])
        return np.clip(arr, bins[0], bins[-1]-1e-10)

    # smearing: (n_bins_reco, n_bins_true, n_univs)
    reco_idx = np.clip(np.searchsorted(bins, _clip(reco_df[signal_mask][reco_var_reco]), side='right') - 1, 0, n_bins-1)
    true_idx = np.clip(np.searchsorted(bins, _clip(reco_df[signal_mask][reco_var_true]), side='right') - 1, 0, n_bins-1)
    reco_oh  = np.eye(n_bins)[reco_idx]
    true_oh  = np.eye(n_bins)[true_idx]
    smearing = np.einsum('ki,kj,ku->iju', reco_oh, true_oh, reco_weights[signal_mask])

    # signal truth histograms: (n_bins, n_univs)
    tidx = np.clip(np.searchsorted(bins, _clip(true_signal_df[true_var_true]), side='right') - 1, 0, n_bins-1)
    t_oh = np.eye(n_bins)[tidx]
    sig_hist_univ = t_oh.T @ true_signal_weights

    # CV truth histogram
    sig_hist_cv = get_hist1d(np.ones(len(true_signal_df)) * true_signal_scale,
                             true_signal_df[true_var_true].to_numpy(), bins)

    response = np.divide(smearing, sig_hist_univ,
                         out=np.zeros_like(smearing), where=sig_hist_univ > 0)
    sig_reco_hists = np.einsum('ijk,j->ik', response, sig_hist_cv)

    # background histograms: (n_bins, n_univs)
    bkg_idx = np.clip(np.searchsorted(bins, _clip(reco_df[~signal_mask][reco_var_reco]), side='right') - 1, 0, n_bins-1)
    b_oh = np.eye(n_bins)[bkg_idx]
    bkg_reco_hists = b_oh.T @ reco_weights[~signal_mask]

    return sig_reco_hists + bkg_reco_hists, response

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N_RECO  = 60
N_TRUE  = 40
N_UNIVS = 8
N_BINS  = 5
bins = np.linspace(0.0, 1.0, N_BINS + 1)

signal_col = np.array([0]*30 + [1]*30, dtype=float)
reco_df = pd.DataFrame({
    'signal':  signal_col,
    'reco_e':  rng.uniform(0, 1, N_RECO),
    'true_e':  rng.uniform(0, 1, N_RECO),
})
true_signal_df = pd.DataFrame({
    'signal': np.zeros(N_TRUE),
    'true_e': rng.uniform(0, 1, N_TRUE),
})

reco_weights        = rng.uniform(0.5, 1.5, (N_RECO,  N_UNIVS))
true_signal_weights = rng.uniform(0.5, 1.5, (N_TRUE,  N_UNIVS))
true_signal_scale   = 0.8

kwargs = dict(
    reco_df=reco_df, true_signal_df=true_signal_df, true_signal_scale=true_signal_scale,
    reco_var_reco='reco_e', reco_var_true='true_e', true_var_true='true_e',
    reco_weights=reco_weights, true_signal_weights=true_signal_weights, bins=bins,
)

ref_hists,  ref_response  = get_xsec_hists_current(**kwargs)
vec_hists,  vec_response  = get_xsec_hists_vectorized(**kwargs)

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
hists_match    = np.allclose(ref_hists,    vec_hists,    atol=1e-10)
response_match = np.allclose(ref_response, vec_response, atol=1e-10)

print(f"hists match:    {hists_match}")
print(f"response match: {response_match}")

if not hists_match:
    diff = np.abs(ref_hists - vec_hists)
    print(f"  max hists diff:    {diff.max():.3e}")
if not response_match:
    diff = np.abs(ref_response - vec_response)
    print(f"  max response diff: {diff.max():.3e}")

assert hists_match and response_match, "FAIL: vectorized output does not match reference"
print("PASS")

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
import timeit

n_repeat = 50
t_current = timeit.timeit(lambda: get_xsec_hists_current(**kwargs),    number=n_repeat)
t_vector  = timeit.timeit(lambda: get_xsec_hists_vectorized(**kwargs), number=n_repeat)

print(f"\nTiming ({n_repeat} runs, N_RECO={N_RECO}, N_TRUE={N_TRUE}, N_UNIVS={N_UNIVS}, N_BINS={N_BINS}):")
print(f"  current:    {t_current/n_repeat*1000:.2f} ms/call")
print(f"  vectorized: {t_vector/n_repeat*1000:.2f} ms/call")
print(f"  speedup:    {t_current/t_vector:.1f}x")
