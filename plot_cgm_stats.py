#!/usr/bin/env python
"""Quick sanity stats + diagnostic plots for the CGM-temperature product.

Reads cgm_temperature_allcrit.hdf5 (schema: (nrec, nst) arrays T_mw, T_med, M_cgm,
f_hot, f_warm, f_cold, Ngas; plus gid and a `stages` attr) and:
  1. prints a per-stage summary table + a global "anomalies" report (the bit that
     usually explains 'something weird' — fraction sums, out-of-range values, NaNs,
     low particle counts, T_mw vs T_med skew);
  2. saves a multi-panel figure (T, phase fractions, M_cgm, Ngas, frac-sum, T-vs-M).

Usage:
    python plot_cgm_stats.py                          # default cis100 path
    python plot_cgm_stats.py --file path/to/prod.hdf5 # explicit product
    python plot_cgm_stats.py --sim cis100 --ngas-min 50
"""
import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")            # headless / cluster-safe
import matplotlib.pyplot as plt

KEYS = ["T_mw", "T_med", "M_cgm", "f_hot", "f_warm", "f_cold", "Ngas"]


def load(path):
    with h5py.File(path, "r") as f:
        stages = list(f.attrs["stages"].split(","))
        D = {k: f[k][:] for k in KEYS if k in f}
        gid = f["gid"][:] if "gid" in f else None
    missing = [k for k in KEYS if k not in D]
    if missing:
        print(f"  [warn] product missing datasets: {missing}")
    return D, stages, gid


def fmt(x):
    return "   nan" if not np.isfinite(x) else f"{x:7.3g}"


def summary_table(D, stages):
    nrec, nst = D["T_mw"].shape
    print(f"\n{'='*96}\nPRODUCT: {nrec} records x {nst} stages\n{'='*96}")
    hdr = f"{'stage':>13} {'Nfin':>5} {'medT_mw':>9} {'medT_med':>9} {'medMcgm':>9} {'medNgas':>8} {'fhot':>6} {'fwarm':>6} {'fcold':>6} {'sum~1':>6}"
    print(hdr); print("-" * len(hdr))
    for j, st in enumerate(stages):
        Tm = D["T_mw"][:, j]
        fin = np.isfinite(Tm)
        n = int(fin.sum())
        if n == 0:
            print(f"{st:>13} {0:>5}   (no finite entries)")
            continue
        fs = (D["f_hot"][:, j] + D["f_warm"][:, j] + D["f_cold"][:, j])
        print(f"{st:>13} {n:>5} "
              f"{fmt(np.nanmedian(Tm))} {fmt(np.nanmedian(D['T_med'][:, j]))} "
              f"{fmt(np.nanmedian(D['M_cgm'][:, j]))} {fmt(np.nanmedian(D['Ngas'][:, j]))} "
              f"{fmt(np.nanmedian(D['f_hot'][:, j]))} {fmt(np.nanmedian(D['f_warm'][:, j]))} "
              f"{fmt(np.nanmedian(D['f_cold'][:, j]))} {fmt(np.nanmedian(fs))}")


def anomalies(D, stages, ngas_min, sum_tol):
    print(f"\n{'='*96}\nANOMALY REPORT\n{'='*96}")
    fh, fw, fc = D["f_hot"], D["f_warm"], D["f_cold"]
    fsum = fh + fw + fc
    finite = np.isfinite(fsum)

    def report(name, mask):
        n = int(np.sum(mask & finite))
        tot = int(finite.sum())
        flag = "  <-- LOOK" if n else ""
        print(f"  {name:<46} {n:>6} / {tot}{flag}")

    report("fraction sum deviates from 1 (|sum-1|>%.0e)" % sum_tol,
           np.abs(fsum - 1.0) > sum_tol)
    report("any phase fraction < 0", (fh < 0) | (fw < 0) | (fc < 0))
    report("any phase fraction > 1", (fh > 1) | (fw > 1) | (fc > 1))

    Tm, Tmed, Ng = D["T_mw"], D["T_med"], D["Ngas"]
    have = np.isfinite(Ng) & (Ng > 0)
    report("T_mw non-finite where Ngas>0", have & ~np.isfinite(Tm))
    report("T_mw <= 0 (unphysical)", np.isfinite(Tm) & (Tm <= 0))
    report("T_mw > 1e9 K (unphysical)", np.isfinite(Tm) & (Tm > 1e9))
    report(f"Ngas < {ngas_min} (unreliable T)", have & (Ng < ngas_min))
    with np.errstate(all="ignore"):
        skew = np.isfinite(Tm) & np.isfinite(Tmed) & (Tmed > 0) & (Tm / Tmed > 10)
    report("T_mw/T_med > 10 (strong skew)", skew)

    # worst fraction-sum offenders (most diagnostic of a normalization bug)
    bad = finite & (np.abs(fsum - 1.0) > sum_tol)
    if bad.any():
        idx = np.argwhere(bad)
        dev = np.abs(fsum - 1.0)[bad]
        order = np.argsort(dev)[::-1][:8]
        print("\n  worst fraction-sum offenders (record, stage -> sum):")
        for o in order:
            ri, sj = idx[o]
            print(f"    rec {ri:4d}  {stages[sj]:>12}  sum={fsum[ri, sj]:.4f} "
                  f"(fh={fh[ri,sj]:.3f} fw={fw[ri,sj]:.3f} fc={fc[ri,sj]:.3f})")


def make_figure(D, stages, out_png):
    nst = len(stages)
    pos = np.arange(nst)
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))

    def violin(ax, arr2d, logy, title, ylabel):
        data = []
        for j in range(nst):
            v = arr2d[:, j]
            v = v[np.isfinite(v)]
            if logy:
                v = v[v > 0]
                v = np.log10(v)
            data.append(v if v.size else np.array([np.nan]))
        parts = ax.violinplot([d[np.isfinite(d)] if np.isfinite(d).any() else [0] for d in data],
                              positions=pos, showmedians=True, widths=0.8)
        ax.set_xticks(pos); ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)

    violin(axs[0, 0], D["T_mw"],  True,  r"$T_{\rm CGM}$ (mass-weighted)", r"$\log_{10} T_{\rm mw}$ [K]")
    violin(axs[0, 2], D["M_cgm"], True,  r"$M_{\rm CGM}$", r"$\log_{10} M_{\rm CGM}$ [M$_\odot$]")
    violin(axs[1, 0], D["Ngas"],  True,  "CGM particle count", r"$\log_{10} N_{\rm gas}$")

    # phase fractions: median +/- IQR per stage, three lines
    ax = axs[0, 1]
    for key, c, lab in [("f_hot", "C3", "hot"), ("f_warm", "C1", "warm"), ("f_cold", "C0", "cold")]:
        med = np.nanmedian(D[key], axis=0)
        q1 = np.nanpercentile(D[key], 25, axis=0)
        q3 = np.nanpercentile(D[key], 75, axis=0)
        ax.plot(pos, med, "-o", color=c, label=lab)
        ax.fill_between(pos, q1, q3, color=c, alpha=0.2)
    ax.set_xticks(pos); ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mass fraction"); ax.set_title("CGM phase fractions (median, IQR)")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)

    # fraction-sum sanity histogram (should spike at 1.0)
    ax = axs[1, 1]
    fsum = (D["f_hot"] + D["f_warm"] + D["f_cold"]).ravel()
    fsum = fsum[np.isfinite(fsum)]
    ax.hist(fsum, bins=60, color="0.4")
    ax.axvline(1.0, color="r", ls="--", lw=1.5, label="expected = 1")
    ax.set_xlabel(r"$f_{\rm hot}+f_{\rm warm}+f_{\rm cold}$")
    ax.set_ylabel("entries"); ax.set_title("phase-fraction sum (normalization check)")
    ax.legend(); ax.grid(alpha=0.3)

    # T vs M_cgm (all stages pooled), colored by Ngas
    ax = axs[1, 2]
    T = D["T_mw"].ravel(); M = D["M_cgm"].ravel(); Ng = D["Ngas"].ravel()
    m = np.isfinite(T) & (T > 0) & np.isfinite(M) & (M > 0)
    sc = ax.scatter(np.log10(M[m]), np.log10(T[m]),
                    c=np.log10(np.clip(Ng[m], 1, None)), s=8, cmap="viridis")
    ax.set_xlabel(r"$\log_{10} M_{\rm CGM}$ [M$_\odot$]")
    ax.set_ylabel(r"$\log_{10} T_{\rm mw}$ [K]")
    ax.set_title("T vs M_CGM (all stages)"); ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10} N_{\rm gas}$")

    fig.suptitle("CGM-temperature product — basic stats", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nsaved figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="cis100")
    ap.add_argument("--file", default=None, help="explicit path to cgm_temperature_allcrit.hdf5")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--ngas-min", type=int, default=50, help="flag CGM with fewer particles")
    ap.add_argument("--sum-tol", type=float, default=1e-3, help="tolerance on phase-fraction sum")
    args = ap.parse_args()

    path = args.file or os.path.join("output", args.sim, "caesar_sfh", "cgm_temperature_allcrit.hdf5")
    if not os.path.exists(path):
        raise SystemExit(f"product not found: {path}\n  pass --file or run the §7g build first.")
    print(f"loading {path}")

    D, stages, gid = load(path)
    summary_table(D, stages)
    anomalies(D, stages, args.ngas_min, args.sum_tol)
    out_png = args.out or os.path.join(os.path.dirname(path), "cgm_stats.png")
    make_figure(D, stages, out_png)


if __name__ == "__main__":
    main()
