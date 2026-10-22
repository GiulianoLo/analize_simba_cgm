"""Comparative data: reference relations, observed-point overlays, readers of the observed tables, analogue matching.

- ``RELATIONS`` (K98, B08, RK19) and ``RELATIONS_EXTRA`` (VERTICO): the Kennicutt-Schmidt relations, ``relation_y``,
  ``IMF_TO_CHABRIER``, ``tdep_ms_gyr`` (Tacconi+18)
- drawing: ``draw_relation``, ``draw_ks_relations``, ``overlay_ks_observations`` (the ALMA-C11 points on
  the K-S plane), ``draw_observed_points`` (observed points with limits on any plane)
- readers: ``read_cigale_best_model``, ``read_observed_photometry``
- analogue matching: ``mahalanobis_cloud``, ``mahalanobis_d2``
"""
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

def relation_y(key, log_sigma_h2, shift_dex=0.0):
    """log Sigma_SFR predicted by relation `key` at log Sigma_H2 [Msun/pc^2] (+ ``shift_dex``, e.g. an IMF conversion)."""
    rel = relation(key)
    return rel["A"] + rel["N"] * np.asarray(log_sigma_h2, float) + shift_dex


def relation(key):
    """The parameters of relation ``key`` (from ``RELATIONS`` or ``RELATIONS_EXTRA``)."""
    return RELATIONS[key] if key in RELATIONS else RELATIONS_EXTRA[key]

def tdep_ms_gyr(z):
    """Tacconi+18 main-sequence depletion time: log t_dep [Gyr] = 0.09 - 0.62 log10(1+z)."""
    return 10.0 ** (0.09 - 0.62 * np.log10(1.0 + np.asarray(z, float)))


# VERTICO (Jiménez-Donaire+23): log Σ_SFR = N (log Σ_mol - log piv) + log(piv 1e6 / t_dep), t_dep = 1.86 Gyr at 10 Msun/pc^2
# Kept out of RELATIONS on purpose: ks_tracks computes an offset for every entry of RELATIONS.
RELATIONS_EXTRA = {}
RELATIONS_EXTRA["VERTICO"] = dict(A=float(np.log10(10.0 * 1e6 / (1.86 * 1e9)) - 0.97 * np.log10(10.0)), N=0.97, sig=0.42,
                            color="#7b3294", ls="-", label="Jiménez-Donaire+23", scat_label="0.42 dex")

IMF_TO_CHABRIER = {"chabrier": 1.0, "kroupa": 0.94, "salpeter": 0.63}   # multiply an SFR by this to put it on Chabrier


def imf_shift_dex(imf):
    """log10 of the factor that puts an SFR measured with ``imf`` on a Chabrier IMF."""
    return float(np.log10(IMF_TO_CHABRIER[imf]))


def draw_relation(ax, key, xs, band=True, lw=1.4, alpha=1.0, zorder=2, shift_dex=0.0):
    """One K-S relation (with its scatter band) on ``ax`` over ``xs`` -> a legend item ``(handle, label)``."""
    from matplotlib.lines import Line2D
    rel = relation(key)
    y = relation_y(key, xs) + shift_dex
    if band:
        ax.fill_between(xs, y - rel["sig"], y + rel["sig"], color=rel["color"], alpha=0.10, lw=0, zorder=0)
    ax.plot(xs, y, color=rel["color"], lw=lw, ls=rel["ls"], alpha=alpha, zorder=zorder)
    return (Line2D([], [], color=rel["color"], lw=lw, ls=rel["ls"], alpha=alpha),
            rf"{rel['label']}" + (rf" $\pm$ {rel['scat_label']}" if band else ""))


def draw_ks_relations(ax, xlim, reference=("B08",), faint=("K98",), tdep_lines_gyr=(0.1, 1.0, 10.0)):
    """The reference relation(s) with their scatter band, the faint ones thin, and lines of constant
    depletion time on a K-S panel spanning ``xlim`` -> legend items."""
    from matplotlib.lines import Line2D
    xs = np.linspace(xlim[0] - 0.5, xlim[1] + 0.5, 80)
    items = []
    for t, ls in zip(tdep_lines_gyr, (":", "-", "--")):
        ax.plot(xs, xs + 6 - np.log10(t * 1e9), color="0.65", ls=ls, lw=0.9, zorder=1)
        ax.text(xlim[1] - 0.05, xlim[1] - 0.05 + 6 - np.log10(t * 1e9) - 0.25, rf"$t_{{\rm dep}}$={t:g} Gyr",
                fontsize=7.5, color="0.45", ha="right", rotation=42, rotation_mode="anchor")
    for key in reference:
        rel = relation(key)
        y = relation_y(key, xs)
        ax.fill_between(xs, y - rel["sig"], y + rel["sig"], color=rel["color"], alpha=0.10, lw=0, zorder=0)
        ax.plot(xs, y, color=rel["color"], lw=1.4, ls=rel["ls"], zorder=2)
        items.append((Line2D([], [], color=rel["color"], lw=1.4, ls=rel["ls"]), rf"{rel['label']} $\pm$ {rel['scat_label']}"))
    for key in faint:
        rel = relation(key)
        ax.plot(xs, relation_y(key, xs), color=rel["color"], lw=0.9, ls=rel["ls"], alpha=0.55, zorder=1)
        items.append((Line2D([], [], color=rel["color"], lw=0.9, ls=rel["ls"], alpha=0.55), rf"{rel['label']} (for reference)"))
    return items


def overlay_ks_observations(ax, table, ylim, color="#f4a261", edge="#3b0f2a", annotate=False, alpha=1.0,
                            label=r"ALMA-C11 QGs, $z\approx0.4$"):
    """Observed galaxies on a K-S panel: filled with asymmetric errors when the size is measured, open with a
    slope-1 arrow when unresolved (both Σ are lower limits). Unbounded lower errors run to ``ylim[0]``.

    ``table`` (a DataFrame) needs ``logSigmaH2``, ``logSigmaSFR``, ``sigma_is_ll``, ``logSigmaSFR_elo`` / ``_ehi``,
    ``logSigmaH2_err`` (and ``id`` for ``annotate``). Returns legend items.
    """
    from matplotlib.lines import Line2D
    for _, r in table.iterrows():
        x, y = float(r["logSigmaH2"]), float(r["logSigmaSFR"])
        if bool(r["sigma_is_ll"]):
            ax.annotate("", xy=(x + 0.3, y + 0.3), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.2, mutation_scale=10, alpha=alpha), zorder=8)
            ax.scatter(x, y, s=110, marker="o", facecolors="white", edgecolors=edge, linewidths=1.4, alpha=alpha, zorder=9)
            ax.scatter(x, y, s=40, marker="o", c=color, edgecolors="none", alpha=alpha, zorder=10)
        else:
            elo = r["logSigmaSFR_elo"] if np.isfinite(r["logSigmaSFR_elo"]) else 0.0
            ehi = r["logSigmaSFR_ehi"] if np.isfinite(r["logSigmaSFR_ehi"]) else 0.0
            elo = min(elo, y - ylim[0])
            xe = r["logSigmaH2_err"] if np.isfinite(r["logSigmaH2_err"]) else 0.0
            ax.errorbar(x, y, xerr=xe, yerr=[[elo], [ehi]], fmt="none", ecolor=edge, elinewidth=1.0, alpha=alpha, zorder=8)
            ax.scatter(x, y, s=110, marker="o", c=color, edgecolors=edge, linewidths=1.4, alpha=alpha, zorder=9)
        if annotate:
            ax.annotate(str(r["id"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color="0.25", zorder=11)
    return [(Line2D([], [], marker="o", ls="", ms=9, mfc=color, mec=edge), rf"{label} (N={len(table)}; size measured)"),
            (Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec=edge), r"unresolved: $\Sigma$ lower limits")]


def draw_observed_points(ax, table, xcol, ycol, to_x=None, to_y=None, color_of=None, xcens=None, ycens=None):
    """Observed points on a plane, one marker per row (``size``, ``marker``, ``edge`` columns), with arrows for upper limits.

    ``to_x`` / ``to_y`` map the table values to data units (e.g. from log space); ``color_of(row)`` gives the
    face colour; ``xcens`` / ``ycens`` name columns that hold ``"ul"`` for an upper limit.
    """
    to_x = to_x or (lambda v: v); to_y = to_y or (lambda v: v)
    for _, r in table.iterrows():
        if not (np.isfinite(r[xcol]) and np.isfinite(r[ycol])):
            continue
        x, y = float(to_x(r[xcol])), float(to_y(r[ycol]))
        ax.scatter(x, y, s=r["size"], marker=r["marker"], c=[color_of(r) if color_of else "k"], edgecolors=r["edge"],
                   linewidths=1.4, zorder=6 + (r["marker"] == "*"))
        if ycens and r[ycens] == "ul":
            ax.annotate("", (x, y), xytext=(0, -15), textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=r["edge"], lw=1.2, mutation_scale=9), zorder=8)
        if xcens and r[xcens] == "ul":
            ax.annotate("", (x, y), xytext=(-15, 0), textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=r["edge"], lw=1.2, mutation_scale=9), zorder=8)


def read_cigale_best_model(path, agn_columns=("agn.SKIRTOR2016_torus", "agn.SKIRTOR2016_polar_dust", "agn.SKIRTOR2016_disk")):
    """A CIGALE best-model FITS -> ``(wavelength [µm, observed], F_nu [mJy], F_nu of the galaxy without the AGN)``, sorted."""
    from astropy.io import fits
    with fits.open(path) as f:
        d = f[1].data
        wl_um = np.asarray(d["wavelength"], float) * 1e-3
        fnu = np.asarray(d["Fnu"], float)
        ltot = np.asarray(d["L_lambda_total"], float)
        lagn = np.zeros_like(ltot)
        for c in agn_columns:
            if c in d.columns.names:
                lagn += np.asarray(d[c], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fgal = fnu * np.clip(1.0 - lagn / ltot, 0.0, 1.0)
    s = np.argsort(wl_um)
    return wl_um[s], fnu[s], fgal[s]


def read_observed_photometry(path, obs_id, bands):
    """The fitted bands of one galaxy in a CIGALE observations FITS -> ``(pivot [µm], flux [mJy], error [mJy])``.

    ``bands`` maps band names to pivot wavelengths; absent, non-finite and non-positive fluxes are dropped.
    """
    from astropy.io import fits
    with fits.open(path) as f:
        d = f[1].data
        row = d[np.asarray([str(i).strip() for i in d["id"]]) == obs_id][0]
        names = d.columns.names
    wl, fl, er = [], [], []
    for band, piv in bands.items():
        if band not in names:
            continue
        v = float(row[band]); e = float(row[band + "_err"])
        if np.isfinite(v) and v > 0:
            wl.append(piv); fl.append(v); er.append(e)
    s = np.argsort(wl)
    return np.asarray(wl)[s], np.asarray(fl)[s], np.asarray(er)[s]


def mahalanobis_cloud(X, ridge=1e-6):
    """Complete rows of a feature matrix -> ``(rows, mean, inverse covariance)``, with a small ridge for stability."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(axis=1)]
    k = X.shape[1]
    C = np.cov(X.T) if k > 1 else np.atleast_2d(np.var(X[:, 0], ddof=1))
    C = C + np.eye(k) * ridge * np.trace(C) / k
    return X, X.mean(axis=0), np.linalg.inv(C)


def mahalanobis_d2(X, mean, inv_cov):
    """Squared Mahalanobis distance of every row of ``X`` to a cloud (:func:`mahalanobis_cloud`)."""
    d = np.asarray(X, float) - mean
    return np.einsum("ij,jk,ik->i", d, inv_cov, d)
