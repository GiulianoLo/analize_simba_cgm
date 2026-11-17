"""OMEGA feasibility: resolution, overlap, merger-tree links, compact vs disky tests, figures.

Light (catalogue products only, seconds). Reads what build_catalogue.py wrote under
output/<sim>/omega_feasibility/ and writes to output/omega_feasibility/:
    feasibility_numbers.json   every number FEASIBILITY.md quotes
    summary_tables.md          the same as readable tables
    figures/omega_{a_tracks,b_overlap,c_young}.{png,pdf}

    python -m omega_feasibility.analyse
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omega_feasibility import config as C  # noqa: E402
from simbanator.stats import wilson, kde_levels  # noqa: E402
from simbanator.plotting.figures import PAPER_RC, save_figure  # noqa: E402

COL = {"cis25": "#2a78d6", "cis100": "#eb6834"}          # boxes (figs b, c)
MRK = {"cis25": "o", "cis100": "s"}
CLS = {"compact": "#2a78d6", "disky": "#eb6834"}          # classes (fig a)
LS = {"cis25": "-", "cis100": "--"}
INK = "#1f1f1e"; MUTED = "#9a998f"
YOUNG = f"n_young{int(C.T_YOUNG_MYR)}"
PLANES = [("log_mstar", "log_sfr10", "logMstar", "logSFR"),
          ("log_mstar", "oh12_sfr", "logMstar", "OH12"),
          ("oh12_sfr", "logCO_sfr", "OH12", "logCO"),
          ("oh12_sfr", "logNO_sfr", "OH12", "logNO")]
LABEL = {"log_mstar": r"$\log M_\star\ [M_\odot]$", "log_sfr10": r"$\log {\rm SFR}\ [M_\odot\,{\rm yr^{-1}}]$",
         "oh12_sfr": r"$12+\log({\rm O/H})$", "logCO_sfr": r"$\log({\rm C/O})$", "logNO_sfr": r"$\log({\rm N/O})$"}


# ─────────────────────────────────────────────────────────────────────────────
# loading
# ─────────────────────────────────────────────────────────────────────────────

def _read(path_noext):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        if os.path.exists(path_noext + ext):
            return fn(path_noext + ext)
    return None


def load(sim):
    d = C.out_dir(sim)
    out = {k: _read(os.path.join(d, f"omega_{k}")) for k in ("highz", "lowz", "progenitors")}
    tp = os.path.join(d, "omega_tracks.npz")
    out["tracks"] = dict(np.load(tp)) if os.path.exists(tp) else None
    h = out["highz"]
    if h is not None:
        with np.errstate(all="ignore"):
            h["log_sfr10"] = np.log10(h["sfr10"].where(h["sfr10"] > 0))
            h["sfr10_100"] = h["sfr10"] / h["sfr100"].where(h["sfr100"] > 0)
        h["sf_central"] = h["central"].astype(bool) & (h["sfr"] > 0)
    return out


def load_obs():
    if not os.path.exists(C.OBS_CSV):
        warnings.warn(f"no observed table at {C.OBS_CSV}: overlap is reported as 'no observed data'")
        return pd.DataFrame(columns=list(C.OBS_COLUMNS) + ["zt"])
    raw = pd.read_csv(C.OBS_CSV)
    o = pd.DataFrame({k: raw[v] for k, v in C.OBS_COLUMNS.items()})
    for k in ("z", "logMstar", "logSFR", "OH12", "logCO", "logNO"):
        o[k] = pd.to_numeric(o[k], errors="coerce")
        for side in ("errp", "errm"):
            src = f"{C.OBS_COLUMNS[k]}_{side}"
            o[f"{k}_{side}"] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
    o["logNO_upper"] = o["logNO_limit"].astype(str).str.strip().eq("<")
    zts = np.array(list(C.HIGHZ_SNAPS))
    dz = np.abs(o["z"].to_numpy()[:, None] - zts[None, :])
    o["zt"] = np.where(dz.min(1) < C.OBS_DZ_MATCH, zts[dz.argmin(1)], -1)
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 1. resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolution_table(h, sim):
    rows = []
    edges = np.arange(6.5, 12.0 + 1e-9, C.MASS_BIN_DEX)
    for zt in C.HIGHZ_SNAPS:
        g = h[(h.z_target == zt) & h.sf_central]
        lm = g.log_mstar.to_numpy()
        b = np.digitize(lm, edges) - 1
        res = {}
        for tag, ok in (("star", g.resolved_star), ("star_young", g.resolved)):
            ok = ok.to_numpy()
            frac = np.array([ok[b == i].mean() if (b == i).sum() >= 3 else np.nan for i in range(len(edges) - 1)])
            filled = np.flatnonzero(np.isfinite(frac))
            thr = np.nan
            for i in filled:  # lowest edge above which every populated bin passes
                if np.all(frac[filled[filled >= i]] >= C.RES_FRACTION):
                    thr = edges[i]; break
            res[tag] = thr
        r = g[g.resolved]
        # one young particle = this much SFR10 (the SFR quantum of the 10 Myr window)
        with np.errstate(all="ignore"):
            q = np.nanmedian((g.sfr10 / g[YOUNG]).where(g[YOUNG] > 0))
        rows.append(dict(sim=sim, run=C.SIMS[sim], zt=zt, snap=C.HIGHZ_SNAPS[zt], z=float(g.z.iloc[0]) if len(g) else np.nan,
                         n_sf_centrals=len(g), n_resolved_star=int(g.resolved_star.sum()), n_resolved=len(r),
                         logM_thr_star=res["star"], logM_thr_star_young=res["star_young"],
                         logM_resolved_min=float(r.log_mstar.min()) if len(r) else np.nan,
                         logM_resolved_max=float(r.log_mstar.max()) if len(r) else np.nan,
                         sfr10_quantum=float(q),
                         sfr10_at_Nyoung_min=float(q * C.N_YOUNG_MIN),
                         sfr10_poisson_err_at_Nyoung_min=1 / np.sqrt(C.N_YOUNG_MIN),
                         median_Nyoung_resolved=float(r[YOUNG].median()) if len(r) else np.nan))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. overlap
# ─────────────────────────────────────────────────────────────────────────────

def _kde_inside(x, y, px, py, upper=False):
    """Is (px, py) inside the KDE_FRACTION contour of (x, y)? For an upper limit: any y' <= py inside."""
    from scipy.stats import gaussian_kde
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < C.KDE_NMIN or not (np.isfinite(px) and np.isfinite(py)):
        return np.nan
    pad = lambda v: (v.min() - 1.0, v.max() + 1.0)
    X, Y, Z, lev = kde_levels(x, y, pad(np.r_[x, px]), pad(np.r_[y, py]), [C.KDE_FRACTION], n=200)
    k = gaussian_kde(np.vstack([x, y]))
    if not upper:
        return bool(k([[px], [py]])[0] >= lev[0])
    yy = np.linspace(min(y.min(), py) - 1.0, py, 200)
    return bool(np.any(k(np.vstack([np.full_like(yy, px), yy])) >= lev[0]))


def _offset(x, y, px, py, half=0.25, nmin=5):
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x - px) <= half)
    return float(py - np.median(y[ok])) if ok.sum() >= nmin and np.isfinite(py) else np.nan


def overlap(H, obs):
    rows = []
    for sim, h in H.items():
        for _, p in obs[obs.zt > 0].iterrows():
            g = h[(h.z_target == p.zt) & h.sf_central & h.resolved]
            for xs, ys, xo, yo in PLANES:
                upper = (yo == "logNO") and bool(p.logNO_upper)
                x, y = g[xs].to_numpy(float), g[ys].to_numpy(float)
                rows.append(dict(sim=sim, id=p.id, zt=p.zt, plane=f"{yo}-{xo}", upper=upper,
                                 n_analogues=int((np.isfinite(x) & np.isfinite(y)).sum()),
                                 obs_x=p[xo], obs_y=p[yo],
                                 inside=_kde_inside(x, y, p[xo], p[yo], upper),
                                 offset_y=np.nan if upper else _offset(x, y, p[xo], p[yo]),
                                 above_mass_threshold=np.nan))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. merger trees
# ─────────────────────────────────────────────────────────────────────────────

def backward(sim, D):
    h, prog = D["highz"], D["progenitors"]
    q = prog[prog.quiescent.astype(bool)]
    out, rows = {}, []
    hz = h.set_index(["z_target", "galaxy_id"])
    for zt in C.HIGHZ_SNAPS:
        pid = q[f"prog_id_z{zt}"].to_numpy()
        has = pid >= 0
        sub = hz.loc[zt].reindex(pid[has]).reset_index()
        sub["qclass"] = q.qclass.to_numpy()[has]; sub["galaxy_id_lowz"] = q.galaxy_id_lowz.to_numpy()[has]
        sub["zt"] = zt
        out[zt] = sub
        for cl in ("compact", "disky"):
            s = sub[sub.qclass == cl]; n_all = int((q.qclass == cl).sum())
            rows.append(dict(sim=sim, zt=zt, qclass=cl, n_quiescent=n_all, n_progenitor=len(s),
                             n_prog_central=int(s.central.astype(bool).sum()),
                             n_prog_resolved_star=int(s.resolved_star.sum()), n_prog_resolved=int(s.resolved.sum()),
                             med_log_mstar=float(s.log_mstar.median()), med_oh12=float(s.oh12_sfr.median()),
                             med_log_sfr10=float(s.log_sfr10.median()), med_logNO=float(s.logNO_sfr.median())))
    return out, rows


def forward(sim, D):
    h, prog = D["highz"], D["progenitors"]
    rows = []
    for zt in C.HIGHZ_SNAPS:
        g = h[(h.z_target == zt) & h.sf_central & h.resolved]
        link = prog[prog[f"prog_id_z{zt}"] >= 0].drop_duplicates(f"prog_id_z{zt}").set_index(f"prog_id_z{zt}")
        m = link.reindex(g.galaxy_id.to_numpy())
        n = len(g); main = int(m.galaxy_id_lowz.notna().sum())
        qq = m.quiescent.fillna(False).astype(bool)
        row = dict(sim=sim, zt=zt, n_resolved=n, n_main_branch=main)
        for tag, k in (("quiescent", int(qq.sum())), ("compact", int((m.qclass == "compact").sum())),
                       ("disky", int((m.qclass == "disky").sum()))):
            p, lo, hi = wilson(k, n)
            row.update({f"n_{tag}": k, f"f_{tag}": p, f"f_{tag}_lo": lo, f"f_{tag}_hi": hi})
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 4. compact vs disky at z~6-8
# ─────────────────────────────────────────────────────────────────────────────

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


def compare(a, b, rng):
    from scipy.stats import ks_2samp, anderson_ksamp
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 5:
        return dict(n_compact=len(a), n_disky=len(b))
    ks = ks_2samp(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad = anderson_ksamp([a, b])
    bd, bm = [], []
    for _ in range(C.N_BOOT):
        ra, rb = rng.choice(a, len(a)), rng.choice(b, len(b))
        bd.append(cliffs_delta(ra, rb)); bm.append(np.median(ra) - np.median(rb))
    return dict(n_compact=len(a), n_disky=len(b), ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
                ad_stat=float(ad.statistic), ad_p_capped=float(ad.significance_level),
                cliffs_delta=cliffs_delta(a, b), cliffs_lo=float(np.percentile(bd, 16)), cliffs_hi=float(np.percentile(bd, 84)),
                dmedian=float(np.median(a) - np.median(b)), dmedian_lo=float(np.percentile(bm, 16)),
                dmedian_hi=float(np.percentile(bm, 84)))


def class_tests(sim, back):
    rng = np.random.default_rng(C.SEED)
    rows = []
    for zt, s in back.items():
        s = s[s.resolved_star]  # >= N_STAR_MIN star particles
        for v in ("oh12_sfr", "logCO_sfr", "logNO_sfr", "sfr10_100", "f_gas", "log_mstar"):
            r = compare(s.loc[s.qclass == "compact", v].to_numpy(float), s.loc[s.qclass == "disky", v].to_numpy(float), rng)
            rows.append(dict(sim=sim, zt=zt, var=v, **r))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────

def _obs_points(ax, obs, xo, yo, zts=None):
    o = obs if zts is None else obs[obs.zt.isin(zts)]
    for _, p in o.iterrows():
        x, y = p[xo], p[yo]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xe = [[p.get(f"{xo}_errm", np.nan)], [p.get(f"{xo}_errp", np.nan)]]
        if yo == "logNO" and p.logNO_upper:
            ax.errorbar(x, y, xerr=np.nan_to_num(xe), yerr=0.15, uplims=True, fmt="D", ms=9, color=INK,
                        mfc="white", mew=1.6, zorder=6, capsize=0)
        else:
            ye = [[p.get(f"{yo}_errm", np.nan)], [p.get(f"{yo}_errp", np.nan)]]
            ax.errorbar(x, y, xerr=np.nan_to_num(xe), yerr=np.nan_to_num(ye), fmt="D", ms=9, color=INK,
                        mec="white", mew=1.0, elinewidth=1.4, zorder=6, capsize=0)


def fig_tracks(H, D, back, res, obs, outdir):
    import matplotlib.pyplot as plt
    from simbanator.histories.properties import gas_oh12
    sims = [s for s in H if D[s]["tracks"] is not None]
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, len(sims), figsize=(7.2 * len(sims), 6.4), sharey=True, squeeze=False)
        for ax, sim in zip(axs[0], sims):
            T = D[sim]["tracks"]
            lm = np.log10(np.where(T["mstar"] > 0, T["mstar"], np.nan)); oh = gas_oh12(T["z_sfr"])
            for cl in ("compact", "disky"):
                m = T["qclass"] == cl
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mx, my = np.nanmedian(lm[:, m], 1), np.nanmedian(oh[:, m], 1)
                ax.plot(mx, my, color=CLS[cl], lw=2.4, label=cl)
                ax.plot(mx[0], my[0], marker="*", ms=16, color=CLS[cl], mec="white", zorder=5)
                for zt in C.HIGHZ_SNAPS:  # true O/H of the progenitors from the particles
                    s = back[sim][zt]; s = s[s.qclass == cl]
                    ax.plot(np.nanmedian(s.log_mstar), np.nanmedian(s.oh12_sfr), marker="o" if cl == "compact" else "s",
                            ms=10, color=CLS[cl], mec="white", mew=1.2, zorder=5)
            for r in res:
                if r["sim"] == sim and np.isfinite(r["logM_thr_star_young"]):
                    ax.axvline(r["logM_thr_star_young"], color=MUTED, lw=1.2, ls=":")
                    ax.text(r["logM_thr_star_young"], 0.03, f" z={r['zt']}", transform=ax.get_xaxis_transform(),
                            color=MUTED, fontsize=13, rotation=90, va="bottom")
            _obs_points(ax, obs, "logMstar", "OH12")
            ax.set_xlabel(LABEL["log_mstar"])
            ax.text(0.03, 0.95, C.SIMS[sim], transform=ax.transAxes, va="top", fontsize=17)
        axs[0, 0].set_ylabel(LABEL["oh12_sfr"])
        axs[0, 0].legend(loc="lower right", frameon=False)
        return save_figure(fig, "omega_a_tracks", os.path.join(outdir, "figures"))


def fig_overlap(H, obs, outdir):
    import matplotlib.pyplot as plt
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(2, 2, figsize=(13, 11.5))
        for ax, (xs, ys, xo, yo) in zip(axs.ravel(), PLANES):
            for sim, h in H.items():
                g = h[h.sf_central]
                r, u = g[g.resolved], g[~g.resolved]
                ax.scatter(u[xs], u[ys], s=5, color=MUTED, alpha=0.25, lw=0, zorder=1)
                ax.scatter(r[xs], r[ys], s=14, color=COL[sim], marker=MRK[sim], alpha=0.55, lw=0, zorder=2,
                           label=C.SIMS[sim])
                x, y = r[xs].to_numpy(float), r[ys].to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= C.KDE_NMIN:
                    pad = lambda v: (np.nanmin(v) - 0.5, np.nanmax(v) + 0.5)
                    X, Y, Z, lev = kde_levels(x[ok], y[ok], pad(x[ok]), pad(y[ok]), [C.KDE_FRACTION])
                    ax.contour(X, Y, Z, lev, colors=COL[sim], linewidths=1.8, zorder=3)
            _obs_points(ax, obs, xo, yo)
            ax.set_xlabel(LABEL[xs]); ax.set_ylabel(LABEL[ys])
        axs[0, 0].legend(loc="upper left", frameon=False, markerscale=2.5)
        fig.tight_layout()
        return save_figure(fig, "omega_b_overlap", os.path.join(outdir, "figures"))


def fig_young(H, res, outdir):
    import matplotlib.pyplot as plt
    zts = sorted(C.HIGHZ_SNAPS, reverse=True)
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, len(zts), figsize=(6.2 * len(zts), 5.6), sharey=True)
        for ax, zt in zip(axs, zts):
            for sim, h in H.items():
                g = h[(h.z_target == zt) & h.sf_central]
                ax.scatter(g.log_mstar, g[YOUNG], s=12, color=COL[sim], marker=MRK[sim], alpha=0.5, lw=0, label=C.SIMS[sim])
                thr = [r["logM_thr_star_young"] for r in res if r["sim"] == sim and r["zt"] == zt]
                if thr and np.isfinite(thr[0]):
                    ax.axvline(thr[0], color=COL[sim], lw=1.6, ls="--")
            ax.axhline(C.N_YOUNG_MIN, color=INK, lw=1.2, ls=":")
            ax.set_yscale("symlog", linthresh=1); ax.set_xlabel(LABEL["log_mstar"])
            ax.text(0.04, 0.95, f"z = {zt}", transform=ax.transAxes, va="top", fontsize=17)
        axs[0].set_ylabel(rf"$N_\star(<{C.T_YOUNG_MYR:.0f}\,{{\rm Myr}})$")
        axs[0].legend(loc="lower right", frameon=False, markerscale=2.5)
        fig.tight_layout()
        return save_figure(fig, "omega_c_young", os.path.join(outdir, "figures"))


# ─────────────────────────────────────────────────────────────────────────────

def _md(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.3g}")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    out = C.combined_dir()
    D = {s: load(s) for s in C.SIMS}
    D = {s: d for s, d in D.items() if d["highz"] is not None}
    H = {s: d["highz"] for s, d in D.items()}
    obs = load_obs()
    print(f"observed points: {len(obs)} ({(obs.zt > 0).sum()} matched to a snapshot) from {C.OBS_CSV}")

    res = sum((resolution_table(h, s) for s, h in H.items()), [])
    ov = overlap(H, obs) if len(obs) else pd.DataFrame()
    if len(ov):
        thr = {(r["sim"], r["zt"]): r["logM_thr_star_young"] for r in res}
        mo = ov.plane == "OH12-logMstar"
        ov.loc[mo, "above_mass_threshold"] = [x >= thr[(s, z)] if np.isfinite(thr[(s, z)]) else np.nan
                                              for s, z, x in zip(ov.sim[mo], ov.zt[mo], ov.obs_x[mo])]
    tree_ok = {s: d for s, d in D.items() if d["progenitors"] is not None}
    back, brows, frows, trows = {}, [], [], []
    for s, d in tree_ok.items():
        back[s], b = backward(s, d); brows += b
        frows += forward(s, d)
        trows += class_tests(s, back[s])

    summ = {}
    if len(ov):
        g = ov.groupby(["sim", "plane"])
        summ = pd.DataFrame({"n_points": g.inside.apply(lambda v: int(v.notna().sum())),
                             "n_inside": g.inside.apply(lambda v: int((v == True).sum())),  # noqa: E712
                             "median_offset_obs_minus_simba": g.offset_y.median()}).reset_index()
        summ["f_inside"] = summ.n_inside / summ.n_points.where(summ.n_points > 0)
    numbers = dict(obs_csv=C.OBS_CSV, n_obs=len(obs), n_obs_matched=int((obs.zt > 0).sum()),
                   obs_logMstar_range=[float(obs.logMstar.min()), float(obs.logMstar.max())] if len(obs) else None,
                   resolution=res, overlap_summary=summ.to_dict("records") if len(ov) else [],
                   overlap_points=ov.to_dict("records") if len(ov) else [],
                   backward=brows, forward=frows, class_tests=trows,
                   lowz_C_split={s: float(np.nanmedian(d["lowz"].loc[d["lowz"].quiescent.astype(bool), "C_r80_r20"]))
                                 for s, d in tree_ok.items()},
                   definitions=dict(solar=C.SOLAR, n_star_min=C.N_STAR_MIN, n_young_min=C.N_YOUNG_MIN,
                                    t_young_myr=C.T_YOUNG_MYR, res_fraction=C.RES_FRACTION, kde_fraction=C.KDE_FRACTION))
    with open(os.path.join(out, "feasibility_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=lambda v: None if (isinstance(v, float) and not np.isfinite(v)) else
                  (v.item() if hasattr(v, "item") else str(v)))
    md = ["# OMEGA feasibility: tables", "", "## Resolution", _md(pd.DataFrame(res))]
    if len(ov):
        md += ["", "## Overlap (resolved analogues, 95 % KDE)", _md(summ), "", "### per point", _md(ov)]
    if brows:
        md += ["", "## Backward: progenitors of z~0.4 quiescent", _md(pd.DataFrame(brows)),
               "", "## Forward: resolved z~6-8 analogues -> z~0.4", _md(pd.DataFrame(frows)),
               "", "## Compact vs disky progenitors (N* >= %d)" % C.N_STAR_MIN, _md(pd.DataFrame(trows))]
    with open(os.path.join(out, "summary_tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    fig_overlap(H, obs, out)
    fig_young(H, res, out)
    if back:
        fig_tracks({s: H[s] for s in back}, D, back, res, obs, out)
    print("->", out)


if __name__ == "__main__":
    main()
