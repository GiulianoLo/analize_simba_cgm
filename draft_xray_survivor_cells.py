# Draft cells for paper_xray_classes_m25.ipynb — the survivor test of the old dusty tail (Section 5c follow-up).
#
# CELL 1 (Section 5d) -> paste into paper_xray_classes_m25.ipynb right after the Section 5c cell. Pure reads of the
#   existing products (QM / XT of Part 0 + B's ism_prediction_histories_<box>.hdf5); runs as-is today. Once the
#   histories carry mdust (CELL 2), the same cell adds the survivor-vs-regrower split of the dust tracks and a
#   third figure panel — no edit needed.
# CELL 2 -> a one-line patch of paper_ism_prediction_boxes.ipynb Part 6 (cell 13) + the rebuild instructions.

# %% ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ── Section 5d — the survivors at fixed dose: HOW the old dusty tail lived through the X-ray phase, and WHY ──
# Section 5c left the puzzle: the age >= 2 Gyr half of the dusty tail is FULLY DOSED (cis100 med E_x 1.56 Gyr,
# strong- and before-enriched) yet dusty. Four candidate mechanisms, each with its own signature in the histories;
# every signature is tested tail-vs-control INSIDE the old fully-dosed sample, so the dose and the clock are fixed:
#   (1) regrowth after shut-off      — the channel stopped firing long ago and the dust re-formed:
#                                        tail HIGHER t_since_x / gap_max, LOWER f_x_late
#   (2) re-accretion                 — the gas came back after the post-SFT minimum and brought / regrew the dust:
#                                        tail regas > 0, higher t_regas / kappa_gas (the noxray-analogue direction)
#   (3) shielding / gas-poor first   — the Section 5c before x strong route: the dust never left because the ISM
#                                        was already compact and gas-poor when the channel opened:
#                                        tail gate_at_sft = 1, higher w_pre / E_x_pre, NO clock or gas signal
#   (4) inheritance artifact         — satellite chains inherit the central's BH history (E_x an upper limit;
#                                        noxray Part 3): every contrast must survive the centrals-only repeat
# With mdust tracks (CELL 2 rebuild) the question is answered directly instead of by proxy: net_lfd = the dust lost
# since SFT and rebound_lfd = the recovery above the post-onset minimum classify every tail galaxy as a SURVIVOR
# (never lost it), a REGROWER (lost it, grew it back) or leave it with the destroyed control.
import h5py
from scipy.stats import mannwhitneyu, spearmanr

P5D_AGE, P5D_DOSE_MIN = P5C_AGE, 1.0     # old (the 5c split) x fully dosed (Part 9: the D/G response saturates by ~1 Gyr)
P5D_LATE_WIN = 1.0                       # the recent-dose window before the anchor [Gyr]
P5D_REGROW   = 0.5                       # a dust change beyond this [dex] separates survived / regrown / destroyed
_G5D = dict(LOGMBH=7.5, FEDD=0.2, FGAS=0.2, THR=0.5)   # the B Part 6a/9 gate, verbatim (JET_LOGMBH, JET_FEDD, XRAY_FGAS_MAX, ONSET_THR)


def _anatomy5d(box):
    """Duty-cycle + gas (+ dust-track when present) anatomy of every tracked galaxy of `box` from B's histories
    -> DataFrame keyed (box, snap, gal_id). x_coup is rebuilt exactly as B Part 6a/9 builds it."""
    path = os.path.join(BOXDIR, f"ism_prediction_histories_{box}.hdf5")
    if not os.path.exists(path):
        return pd.DataFrame()
    aux = XT.loc[XT["box"] == box, KEY + [c for c in ("t_sft", "t_anchor", "t_onset") if c in XT.columns]].set_index(["snap", "gal_id"])
    rows = []
    with h5py.File(path, "r") as f:
        for grp in f:
            g = f[grp]; anchor = int(grp.replace("snap", ""))
            t = g["t_yr"][:] / 1e9; order = np.argsort(t); t = t[order]
            H = {k: g[k][:][order] for k in ("mstar", "mgas", "mbh", "fedd")}
            md_all = g["mdust"][:][order] if "mdust" in g else None
            fgas = np.where(H["mstar"] > 0, H["mgas"] / H["mstar"], np.nan)
            bh_ok = np.isfinite(H["mbh"]) & np.isfinite(H["fedd"])
            wjet = np.where(bh_ok, np.where(H["mbh"] > 10 ** _G5D["LOGMBH"],
                                            np.clip(np.log10(_G5D["FEDD"] / np.clip(H["fedd"], 1e-12, None)), 0.0, 1.0), 0.0), np.nan)
            xc = np.where(np.isfinite(wjet) & np.isfinite(fgas), wjet * (fgas < _G5D["FGAS"]).astype(float), np.nan)
            mfloor = max(float(BOXES.get(box, {}).get("m_gas") or 1e6), 1.0)          # one gas particle: the re-accretion log's floor
            for j, gid in enumerate(g["gal_ids"][:]):
                try:
                    a = aux.loc[(anchor, int(gid))]
                except KeyError:
                    continue
                t_sft, t_anchor = float(a.get("t_sft", np.nan)), float(a.get("t_anchor", np.nan))
                r = dict(box=box, snap=anchor, gal_id=int(gid), t_since_x=np.nan, f_x_late=np.nan, n_epi=0, gap_max=np.nan,
                         regas=np.nan, t_regas=np.nan, net_lfd=np.nan, rebound_lfd=np.nan)
                x = xc[:, j]; fin = np.isfinite(x)
                if fin.sum() >= 3 and np.isfinite(t_anchor):
                    tt, xx = t[fin], x[fin]
                    on = xx >= _G5D["THR"]
                    if on.any():
                        r["t_since_x"] = t_anchor - float(tt[on][-1])                 # time since the channel last fired
                        e = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0])); st, en = e[::2], e[1::2] - 1
                        r["n_epi"] = len(st)
                        r["gap_max"] = float(max((tt[st[k + 1]] - tt[en[k]] for k in range(len(st) - 1)), default=0.0))
                    late = tt >= t_anchor - P5D_LATE_WIN
                    if late.any():
                        r["f_x_late"] = float(np.mean(xx[late]))                      # the recent dose rate
                mg = H["mgas"][:, j]
                fing = np.isfinite(mg) & (t >= t_sft) if np.isfinite(t_sft) else np.zeros(len(t), bool)
                if fing.sum() >= 3:
                    tg, mgg = t[fing], np.clip(mg[fing], mfloor, None)
                    k = int(np.argmin(mgg))
                    r["regas"] = float(np.log10(mgg[-1] / mgg[k]))                    # > 0: the gas came back after its minimum
                    r["t_regas"] = float(tg[-1] - tg[k])
                if md_all is not None and np.isfinite(t_sft):                         # the direct answer, once CELL 2 ran
                    md, ms = md_all[:, j], H["mstar"][:, j]
                    fd = np.isfinite(md) & np.isfinite(ms) & (ms > 0)
                    if fd.sum() >= 3:
                        td = t[fd]; lfd_t = np.where(md[fd] > 0, np.log10(np.clip(md[fd], 1e-3, None) / ms[fd]), P9_FLOOR)
                        r["net_lfd"] = float(lfd_t[-1] - np.interp(t_sft, td, lfd_t))     # dust change SFT -> anchor [dex]
                        t_on = float(a.get("t_onset", np.nan))
                        w = td >= (t_on if np.isfinite(t_on) else t_sft)
                        if w.any():
                            r["rebound_lfd"] = float(lfd_t[-1] - np.min(lfd_t[w]))        # recovery above the post-onset minimum
                rows.append(r)
    return pd.DataFrame(rows)


A5D = pd.concat([_anatomy5d(b) for b in P9_BOXES], ignore_index=True) if P9_BOXES else pd.DataFrame()
HAS_MD5D = bool(len(A5D)) and A5D["net_lfd"].notna().any()
_extra5d = [c for c in ("gate_at_sft", "fgas_sft", "lag_gate", "e_x_pre") if c in XT.columns and c not in QM.columns]
QD = QM.merge(A5D, on=KEY, how="left").merge(XT[KEY + _extra5d], on=KEY, how="left") if len(A5D) else QM.copy()

P5D_VARS = [("t_since_x", "time since the last X-ray episode [Gyr]", "regrowth"), ("f_x_late", f"mean x_coup, last {P5D_LATE_WIN:g} Gyr", "regrowth"),
            ("gap_max", "longest duty gap [Gyr]", "regrowth"), ("n_epi", "N X-ray episodes", "regrowth"),
            ("regas", "log M_gas(anchor) / min M_gas after SFT", "re-accretion"), ("t_regas", "time since the gas minimum [Gyr]", "re-accretion"),
            ("kappa_gas", "kappa_rot gas at the anchor", "re-accretion"),
            ("gate_at_sft", "gas-poor at SFT (the sequence)", "shielding"), ("fgas_sft", "f_gas at SFT", "shielding"),
            ("w_pre", "w_pre (pre-SFT coupling)", "shielding"), ("e_x_pre", "E_x over [SFT-1, SFT] [Gyr]", "shielding"),
            ("lag_gate", "t_gate - t_SFT [Gyr]", "shielding"),
            ("log_mstar", "log M*", "context"), ("central", "central", "context")]

print(f"the old fully-dosed sample (age >= {P5D_AGE:g} Gyr, E_x >= {P5D_DOSE_MIN:g} Gyr): the dusty tail (log M_dust/M* >= {P5C_TAIL:g}) vs the control at the same dose and age"
      + ("" if HAS_MD5D else " — mdust tracks NOT in the histories yet: run CELL 2's rebuild for the survivor / regrower split"))
S5D_ROWS, _fig_boxes = [], []
for box in P9_BOXES:
    g = QD[(QD["box"] == box) & QD["xray_class"].isin(XCLASSES) & np.isfinite(QD["e_x"])].copy()
    g["dusty"] = g["lfd"] >= P5C_TAIL
    s = g[(g["age"] >= P5D_AGE) & (g["e_x"] >= P5D_DOSE_MIN)].copy()
    if len(s) < 4 * P9_MIN or s["dusty"].sum() < P9_MIN:
        print(f"\n{box}: only {int(s['dusty'].sum()) if len(s) else 0} old fully-dosed tail galaxies — skipped")
        continue
    _fig_boxes.append(box)
    d, c = s[s["dusty"]], s[~s["dusty"]]
    print(f"\n{box}: {len(s)} old fully-dosed galaxies, {len(d)} in the tail ({100 * len(d) / len(s):.1f} %); med E_x {d['e_x'].median():.2f} vs {c['e_x'].median():.2f} Gyr, age {d['age'].median():.2f} vs {c['age'].median():.2f} Gyr")
    print(f"  {'quantity':42s} {'mechanism':12s} {'med tail':>9s} {'med ctrl':>9s} {'MW p':>8s} {'AUC':>6s} {'AUC cen':>8s}")
    for v, lab, mech in P5D_VARS:
        if v not in s.columns:
            continue
        auc, p = auc_of(s[v], s["dusty"])
        cen = s[s["central"] > 0] if "central" in s.columns else s.iloc[0:0]
        auc_c, _ = auc_of(cen[v], cen["dusty"]) if len(cen) >= 4 * P9_MIN and cen["dusty"].sum() >= 3 else (np.nan, np.nan)
        S5D_ROWS.append(dict(box=box, var=v, mechanism=mech, med_tail=d[v].median(), med_ctrl=c[v].median(), mw_p=p, auc=auc, auc_cen=auc_c,
                             n=len(s), n_tail=len(d)))
        print(f"  {lab:42s} {mech:12s} {d[v].median():9.2f} {c[v].median():9.2f} {p:8.1g} {auc:6.2f} {auc_c:8.2f}")
    top = sorted((r for r in S5D_ROWS if r["box"] == box and np.isfinite(r["auc"]) and r["mw_p"] < 0.05),
                 key=lambda r: -abs(r["auc"] - 0.5))[:3]
    print("  strongest tail signals: " + ("; ".join(f"{r['var']} (AUC {r['auc']:.2f}, p {r['mw_p']:.1g})" for r in top) if top else "none at p < 0.05"))
    if HAS_MD5D and d["net_lfd"].notna().sum() >= 3:
        surv = (d["net_lfd"] > -P5D_REGROW) & (d["rebound_lfd"] < P5D_REGROW)
        regr = d["rebound_lfd"] >= P5D_REGROW
        print(f"  DUST TRACKS — the tail: {int(surv.sum())} SURVIVORS (lost < {P5D_REGROW:g} dex since SFT, no rebound), {int(regr.sum())} REGROWERS "
              f"(rebound >= {P5D_REGROW:g} dex above the post-onset minimum), {int((~surv & ~regr).sum())} other; "
              f"net dust change since SFT: control {c['net_lfd'].median():.2f} dex vs tail {d['net_lfd'].median():.2f} dex")
        for lab, m in (("survivors", surv), ("regrowers", regr)):
            h = d[m]
            if len(h) < 3:
                continue
            extra = f", gas-poor at SFT {100 * h['gate_at_sft'].mean():.0f} %" if "gate_at_sft" in h.columns else ""
            print(f"    {lab:10s}: med t_since_x {h['t_since_x'].median():.2f} Gyr, regas {h['regas'].median():.2f}, w_pre {h['w_pre'].median():.2f}{extra}")

if S5D_ROWS:
    pd.DataFrame(S5D_ROWS).to_csv(os.path.join(PAPERDIR, "paper_xray_survivor_anatomy.csv"), index=False)
    print(f"\n-> {os.path.join(PAPERDIR, 'paper_xray_survivor_anatomy.csv')}")

# ── the figure: one row per box; (a) D/G against the time since the last episode, (b) against the re-accretion,
#    (c, once the tracks exist) the net-vs-rebound plane that names every galaxy's route ──
if _fig_boxes:
    ncol = 3 if HAS_MD5D else 2
    fig, axs = plt.subplots(len(_fig_boxes), ncol, figsize=(4.9 * ncol, 3.7 * len(_fig_boxes)), squeeze=False)
    for i, box in enumerate(_fig_boxes):
        g = QD[(QD["box"] == box) & QD["xray_class"].isin(XCLASSES) & np.isfinite(QD["e_x"])].copy()
        g["dusty"] = g["lfd"] >= P5C_TAIL
        s = g[(g["age"] >= P5D_AGE) & (g["e_x"] >= P5D_DOSE_MIN)]
        d, c = s[s["dusty"]], s[~s["dusty"]]
        for k, (xv, xlab, vline) in enumerate([("t_since_x", "time since the last X-ray episode [Gyr]", None),
                                               ("regas", "log $M_{\\rm gas}$(anchor) / min $M_{\\rm gas}$ after SFT", 0.0)]):
            ax = axs[i][k]
            ax.scatter(c[xv], c["ldg"], s=7, color="0.75", lw=0, zorder=1, label=f"control (N = {len(c)})")
            for oc in XCLASSES:
                for cc in P5C_COUP:
                    m = (d["xray_class"] == oc) & (d["agn_class"] == cc)
                    if not m.any():
                        continue
                    ax.scatter(d.loc[m, xv], d.loc[m, "ldg"], s=26, marker="o", facecolors="none" if cc == "weak" else XCOL[oc],
                               edgecolors=XCOL[oc], lw=1.2, zorder=3, label=f"tail, {XABBR[oc]} x {cc} ({int(m.sum())})")
            ok = np.isfinite(s[xv]) & np.isfinite(s["ldg"])
            if ok.sum() >= 4 * P9_MIN:
                rr = spearmanr(s.loc[ok, xv], s.loc[ok, "ldg"])
                ax.text(0.03, 0.03, f"$\\rho$ = {rr.statistic:.2f} (p = {rr.pvalue:.1g})", transform=ax.transAxes,
                        fontsize=FONT["note"], color="0.15", path_effects=[withStroke(linewidth=2.6, foreground="white")])
            if vline is not None:
                ax.axvline(vline, color="0.5", lw=1.0, ls=":", zorder=1)
            ax.set_xlabel(xlab); ax.set_ylabel("log $M_{\\rm dust}/M_{\\rm gas}$" if k == 0 else "")
            ax.tick_params(direction="in", top=True, right=True, labelsize=FONT["tick"]); ax.grid(False)
            ax.set_title(f"({'abcdef'[i * ncol + k]}) {BOXES[box]['label'].split(' (')[0]}", loc="left", fontsize=FONT["tag"] - 0.8)
            if i == 0 and k == 0:
                ax.legend(loc="upper right", frameon=False, fontsize=FONT["legend"] - 1.8, handlelength=1.2)
        if HAS_MD5D:
            ax = axs[i][2]
            ax.scatter(c["net_lfd"], c["rebound_lfd"], s=7, color="0.75", lw=0, zorder=1)
            ax.scatter(d["net_lfd"], d["rebound_lfd"], s=26, facecolors="none", edgecolors="#b2182b", lw=1.2, zorder=3)
            ax.axvline(-P5D_REGROW, color="0.5", lw=1.0, ls=":"); ax.axhline(P5D_REGROW, color="0.5", lw=1.0, ls=":")
            ax.set_xlabel("net $\\Delta$log $M_{\\rm dust}/M_\\star$, SFT $\\to$ anchor [dex]"); ax.set_ylabel("rebound above the post-onset minimum [dex]")
            ax.text(0.97, 0.95, "REGROWERS", transform=ax.transAxes, ha="right", va="top", fontsize=FONT["note"], color="0.3")
            ax.text(0.97, 0.05, "SURVIVORS", transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT["note"], color="0.3")
            ax.text(0.03, 0.05, "DESTROYED", transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT["note"], color="0.3")
            ax.tick_params(direction="in", top=True, right=True, labelsize=FONT["tick"]); ax.grid(False)
            ax.set_title(f"({'abcdef'[i * ncol + 2]}) the dust routes (tail in red)", loc="left", fontsize=FONT["tag"] - 0.8)
    fig.suptitle(wrap_caption(f"the survivors at fixed dose: the old (age >= {P5D_AGE:g} Gyr), fully-dosed (E_x >= {P5D_DOSE_MIN:g} Gyr) quenched galaxies, "
                              f"the dusty tail (log M_dust/M* >= {P5C_TAIL:g}) against the control at the same age and dose — "
                              f"(a) dust survives where the channel still fires or long after it stopped? (b) did the gas come back after its post-SFT minimum? "
                              + ("(c) each galaxy's dust route since quenching (net loss vs recovery). " if HAS_MD5D else "")
                              + f"quenched galaxies, {SEL_TXT}; classes = {XTITLE}", width=int(13 * fig.get_figwidth())), fontsize=FONT["note"] + 1, y=0.995)
    fig.subplots_adjust(top=0.90 if len(_fig_boxes) > 1 else 0.82, bottom=0.07, hspace=0.32, wspace=0.24)
    paper_save(fig, "survivor_anatomy")
    plt.show()

# %% ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CELL 2 — the dust tracks: a one-line patch of paper_ism_prediction_boxes.ipynb Part 6 (cell 13) + the rebuild.
#
# 1. In paper_ism_prediction_boxes.ipynb cell 13, extend HIST_COLS with the two catalogue fields (both verified
#    present in the caesar catalogues, e.g. m100n1024_105.hdf5):
#
#    HIST_COLS = {"mstar": "galaxy_data/dicts/masses.stellar", "sfr": "galaxy_data/sfr",
#                 "mgas": "galaxy_data/dicts/masses.gas",      "mbh": "galaxy_data/dicts/masses.bh",
#                 "fedd": "galaxy_data/bh_fedd",
#                 "mdust": "galaxy_data/dicts/masses.dust",                        # <- new: the dust track
#                 "zgas": "galaxy_data/dicts/metallicities.mass_weighted"}         # <- new: D/G's metallicity leg
#
# 2. Delete the cached histories so the builder re-runs (it reads each catalogue of the ladder once, the same
#    pass as the 2026-08-30 build; everything downstream of the histories is unchanged — Part 6a/9's outputs
#    only gain the two arrays):
#
#    rm output/box_resolution/ism_prediction/ism_prediction_histories_*.hdf5
#
# 3. Re-run B Part 6 (and Part 9 if its CSV should refresh, though nothing it writes changes), then re-run
#    Section 5d above: HAS_MD5D flips on, the survivor / regrower split prints and panel (c) appears.
