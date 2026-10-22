"""Comparative data: loaders for obs_data/ (ALMA-C11, CEERS, LEGA-C, literature), relations
(K98, main sequence, depletion time), observed overlays and analogue matching. Filled in from the
notebooks in the next step of the restructure."""
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════

# ── reference relations (copied from pilot_specphot/scripts/plot_ks.py) ──────────────────────────
# log Sigma_SFR [Msun/yr/kpc^2] = A + N log Sigma_x [Msun/pc^2]; sig = published 1-sigma scatter [dex]
RELATIONS = {
    "K98": dict(A=float(np.log10(2.5e-4 * 0.63)), N=1.4, sig=0.30, color="#d95f02", ls="-",
                label="Kennicutt 98 (total gas, Chabrier)", scat_label="0.30 dex rms"),
    "B08": dict(A=-2.06 - 0.96 * 1.0, N=0.96, sig=0.20, color="#1b9e77", ls="--",
                label=r"Bigiel+08 ($\Sigma_{\rm H_2}$, 750 pc)", scat_label="0.20 dex rms"),
    "RK19": dict(A=-3.84, N=1.41, sig=0.28, color="#7570b3", ls="-.",
                 label="de los Reyes & Kennicutt 19 (spirals)",
                 scat_label=r"0.28 dex $\sigma_{\rm int}$"),
}

def relation_y(key, log_sigma_h2):
    """log Sigma_SFR predicted by relation `key` at log Sigma_H2 [Msun/pc^2]."""
    rel = RELATIONS[key]
    return rel["A"] + rel["N"] * np.asarray(log_sigma_h2, float)

def tdep_ms_gyr(z):
    """Tacconi+18 main-sequence depletion time: log t_dep [Gyr] = 0.09 - 0.62 log10(1+z)."""
    return 10.0 ** (0.09 - 0.62 * np.log10(1.0 + np.asarray(z, float)))
