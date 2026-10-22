# ── Section 5c — the AGN-coupling class (weak / strong of B Part 6a) crossed with the X-ray class on the dusty tail: the cells of the combined division, the
#    stratified tests (does the coupling add separation at fixed onset timing, and the reverse?), the dose behind the tail, and the tail split by stellar age ──
from scipy.stats import fisher_exact, chi2_contingency
from scipy.stats import hypergeom as _hg
from matplotlib.patches import Patch

P5C_TAIL = -4.0                     # the dusty tail on log M_dust/M* (the P5 threshold the observed detections sit above)
P5C_COUP = ["weak", "strong"]       # the two-class coupling division (B Part 6a: w_pre < 0.3 -> weak, >= 0.3 -> strong; no_AGN dropped here)
P5C_AGE  = 2.0                      # the tail split into young / old at this mass-weighted age [Gyr]
P5C_DOSE = [("e_x", "$E_x$"), ("f_x", "$f_x$"), ("t_gaspoor", "$T_{\\rm gas-poor}$"), ("dt_onset_sft", "$t_{\\rm onset} - t_{\\rm SFT}$"), ("w_pre", "$w_{\\rm pre}$"), ("age", "age")]
P5C_HATCH = {"weak": "///", "strong": None}


def wilson(k, n, z=1.0):
    """Binomial fraction k/n with its Wilson interval at z sigma -> (p, lo, hi)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def auc_of(x, pos):
    """AUC of x as a score for `pos` against the rest (Mann-Whitney U / n1 n0; > 0.5: a higher x favours `pos`) -> (auc, p)."""
    x, s = np.asarray(x, float), np.asarray(pos, bool); ok = np.isfinite(x)
    a, b = x[ok & s], x[ok & ~s]
    if min(len(a), len(b)) < 2:
        return np.nan, np.nan
    r = mannwhitneyu(a, b)
    return float(r.statistic / (len(a) * len(b))), float(r.pvalue)


def _g5c(box):
    g = QM[(QM["box"] == box) & QM["xray_class"].isin(XCLASSES) & QM["agn_class"].isin(P5C_COUP)].copy()
    g["dusty"] = g["lfd"] >= P5C_TAIL
    return g


P5C_ROWS = []
print(f"the coupling class x the X-ray class ({XSCHEME}) on the dusty tail (log M_dust/M* >= {P5C_TAIL:g}); log M* > {MASS_MIN:g}, classified galaxies with a finite w_pre")
for box in P9_BOXES:
    g = _g5c(box)
    _nall = int(((QM["box"] == box) & QM["xray_class"].isin(XCLASSES)).sum())
    nt = int(g["dusty"].sum())
    tab = pd.crosstab(g["agn_class"], g["xray_class"]).reindex(index=P5C_COUP, columns=XCLASSES, fill_value=0)
    t = tab.to_numpy(); t = t[t.sum(1) > 0][:, t.sum(0) > 0]
    chi2p, v = np.nan, np.nan
    if min(t.shape) >= 2:
        _chi = chi2_contingency(t); chi2p = float(_chi[1]); v = float(np.sqrt(_chi[0] / (t.sum() * (min(t.shape) - 1))))
    print(f"\n{box}: {len(g)} galaxies with both classes ({_nall - len(g)} classified without a coupling class: no BH history), {nt} in the tail ({100 * nt / len(g):.1f} %); "
          f"coupling x onset chi-square p = {chi2p:.2g}, Cramér's V = {v:.2f}")
    print("  " + tab.to_string().replace("\n", "\n  "))
    print(f"  {'cell':>18s} {'N':>5s} {'in tail':>7s} {'% dusty [68 %]':>21s} {'med lfd':>8s} {'med ldg':>8s} {'med age':>8s} {'med E_x':>8s} {'tail vs box share':>19s} {'P':>8s}")
    for oc in XCLASSES:
        for cc in P5C_COUP:
            h = g[(g["xray_class"] == oc) & (g["agn_class"] == cc)]
            if not len(h):
                continue
            k = int(h["dusty"].sum()); p_, lo, hi = wilson(k, len(h))
            sa, st = len(h) / len(g), (k / nt if nt else np.nan)
            over = bool(nt) and st >= sa
            ph = (float(_hg.sf(k - 1, len(g), len(h), nt)) if over else float(_hg.cdf(k, len(g), len(h), nt))) if nt else np.nan
            P5C_ROWS.append(dict(box=box, onset=oc, coupling=cc, n=len(h), n_tail=k, frac=p_, frac_lo=lo, frac_hi=hi, lfd=h["lfd"].median(), ldg=h["ldg"].median(),
                                 age=h["age"].median(), e_x=h["e_x"].median(), share_tail=st, share_all=sa, side="over" if over else "under", p_hyper=ph))
            print(f"  {XABBR[oc] + ' x ' + cc:>18s} {len(h):5d} {k:7d} {100 * p_:7.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}] {h['lfd'].median():8.2f} {h['ldg'].median():8.2f} "
                  f"{h['age'].median():8.2f} {h['e_x'].median():8.2f} {100 * st:7.1f} vs {100 * sa:4.1f} % {'over' if over else 'under':>5s} {ph:8.1g}")
    for oc in XCLASSES:                                   # the coupling at fixed onset timing
        h = g[g["xray_class"] == oc]
        a, b = (h.loc[h["agn_class"] == c, "lfd"].dropna() for c in P5C_COUP)
        if min(len(a), len(b)) < 3:
            continue
        t2 = np.array([[int((h["dusty"] & (h["agn_class"] == c)).sum()), int((~h["dusty"] & (h["agn_class"] == c)).sum())] for c in P5C_COUP])
        odds, pf = fisher_exact(t2)
        print(f"  inside {XABBR[oc]:>7s}: weak vs strong — MW on lfd p = {mannwhitneyu(a, b).pvalue:.2g}; Fisher on the tail p = {pf:.2g}, odds(weak) = {odds:.2f}")
    for cc in P5C_COUP:                                   # the onset timing at fixed coupling
        h = g[g["agn_class"] == cc]
        a, b = h.loc[h["xray_class"] == XREF, "lfd"].dropna(), h.loc[h["xray_class"] != XREF, "lfd"].dropna()
        if min(len(a), len(b)) < 3:
            continue
        print(f"  inside {cc:>7s}: {XABBR[XREF]} ({100 * h.loc[h['xray_class'] == XREF, 'dusty'].mean():.0f} % dusty, N = {len(a)}) vs the rest "
              f"({100 * h.loc[h['xray_class'] != XREF, 'dusty'].mean():.0f} %, N = {len(b)}) — MW on lfd p = {mannwhitneyu(a, b).pvalue:.2g}")
    parts = []
    for c, lab in P5C_DOSE:
        auc, p = auc_of(g[c], g["dusty"])
        parts.append(f"{lab} {auc:.2f} ({p:.1g})")
    print("  AUC of each quantity for the tail (> 0.5: higher favours the tail): " + "; ".join(parts))
    d = g[g["dusty"]]
    for lab, m in ((f"age >= {P5C_AGE:g}", d["age"] >= P5C_AGE), (f"age < {P5C_AGE:g}", d["age"] < P5C_AGE)):
        h = d[m]
        if len(h) < 3:
            continue
        print(f"  the tail, {lab:10s}: N = {len(h):4d} — onset " + str({XABBR[c]: int((h["xray_class"] == c).sum()) for c in XCLASSES})
              + ", coupling " + str({c: int((h["agn_class"] == c).sum()) for c in P5C_COUP})
              + f"; med E_x {h['e_x'].median():.2f}, f_x {h['f_x'].median():.2f}, lfd {h['lfd'].median():+.2f}, z {h['z'].median():.2f}")
pd.DataFrame(P5C_ROWS).to_csv(os.path.join(PAPERDIR, "paper_xray_coupling_onset.csv"), index=False)
print(f"\n-> {os.path.join(PAPERDIR, 'paper_xray_coupling_onset.csv')}")

# ── the figure: (top) the tail fraction of every coupling x onset cell; (bottom) the dose-age plane — the dusty galaxies against the dose typical of their age ──
if len(P9_BOXES):
    _nb = len(P9_BOXES)
    fig, axes = plt.subplots(2, _nb, figsize=(6.4 * _nb + 0.8, 10.4), squeeze=False, gridspec_kw=dict(hspace=0.30, wspace=0.20))
    for i, box in enumerate(P9_BOXES):
        g = _g5c(box)
        ax = axes[0][i]
        for jo, oc in enumerate(XCLASSES):
            for jc, cc in enumerate(P5C_COUP):
                h = g[(g["xray_class"] == oc) & (g["agn_class"] == cc)]
                if not len(h):
                    continue
                p_, lo, hi = wilson(int(h["dusty"].sum()), len(h))
                xx = jo + (jc - 0.5) * 0.38
                ax.bar(xx, 100 * p_, width=0.34, color=XCOL[oc], alpha=0.45 if cc == "weak" else 1.0, hatch=P5C_HATCH[cc], edgecolor="k", lw=0.7,
                       yerr=[[100 * (p_ - lo)], [100 * (hi - p_)]], capsize=2.5, error_kw=dict(ecolor="0.15", lw=1.0))
                ax.text(xx, 1.5, f"N = {len(h)}", ha="center", va="bottom", rotation=90, fontsize=FONT["note"] - 0.5, color="k", zorder=6)
        ax.axhline(100 * g["dusty"].mean(), color="0.2", lw=1.5, ls=(0, (1.2, 1.4)), zorder=1)
        hnd = [Patch(facecolor="0.65", alpha=0.45, hatch=P5C_HATCH["weak"], edgecolor="k", label=r"weak ($w_{\rm pre} < 0.3$)"),
               Patch(facecolor="0.65", edgecolor="k", label=r"strong ($w_{\rm pre} \geq 0.3$)"),
               Line2D([], [], color="0.2", lw=1.5, ls=(0, (1.2, 1.4)), label=f"all classified ({100 * g['dusty'].mean():.0f} %)")]
        ax.legend(handles=hnd, loc="upper left", frameon=False, fontsize=FONT["legend"] - 1.0)
        ax.set_xticks(range(len(XCLASSES))); ax.set_xticklabels([XSHORT[c] for c in XCLASSES], fontsize=FONT["tick"])
        ax.set_ylim(0, 108); ax.set_ylabel(f"% of the cell in the tail (log $M_{{\\rm dust}}/M_\\star \\geq {P5C_TAIL:g}$)" if i == 0 else "")
        ax.tick_params(direction="in", top=True, right=True, labelsize=FONT["tick"]); ax.grid(False)
        ax.set_title(f"({'abcdef'[i]}) {BOXES[box]['label'].split(' (')[0]}: the tail per coupling x onset cell", loc="left", fontsize=FONT["tag"] - 0.8)
        # (bottom) the dose-age plane
        ax = axes[1][i]
        m_d = g["dusty"].to_numpy()
        ax.scatter(g.loc[~m_d, "age"], g.loc[~m_d, "e_x"], s=7, color="0.75", lw=0, zorder=1, label=f"not in the tail (N = {int((~m_d).sum())})")
        for oc in XCLASSES:
            for cc in P5C_COUP:
                m = m_d & (g["xray_class"] == oc).to_numpy() & (g["agn_class"] == cc).to_numpy()
                if not m.any():
                    continue
                ax.scatter(g.loc[m, "age"], g.loc[m, "e_x"], s=26, marker="o", facecolors="none" if cc == "weak" else XCOL[oc], edgecolors=XCOL[oc], lw=1.2,
                           zorder=3, label=f"tail, {XABBR[oc]} x {cc} (N = {int(m.sum())})")
        aa, ee = g["age"].to_numpy(float), g["e_x"].to_numpy(float); ok = np.isfinite(aa) & np.isfinite(ee)
        if ok.sum() >= 4 * P9_MIN:                                     # the dose typical of a classified galaxy of that age
            edges = np.quantile(aa[ok], np.linspace(0, 1, 7)); xm, ym = [], []
            for j in range(6):
                m = ok & (aa >= edges[j]) & (aa <= edges[j + 1])
                if m.sum() >= P9_MIN:
                    xm.append(float(np.median(aa[m]))); ym.append(float(np.median(ee[m])))
            ax.plot(xm, ym, color="0.15", lw=2.0, ls=(0, (1.2, 1.4)), zorder=4, path_effects=[withStroke(linewidth=3.4, foreground="white")],
                    label="median $E_x$ at this age (all classified)")
        ax.axvline(P5C_AGE, color="0.5", lw=1.0, ls=":", zorder=1)
        d = g[g["dusty"]]
        _lines = []
        for lab, m2 in ((f"age < {P5C_AGE:g}", d["age"] < P5C_AGE), (f"age $\\geq$ {P5C_AGE:g}", d["age"] >= P5C_AGE)):
            h = d[m2]
            if len(h) >= 3:
                _lines.append(f"tail, {lab}: N = {len(h)}, med $E_x$ {h['e_x'].median():.2f} Gyr, {XABBR[XREF]} {100 * (h['xray_class'] == XREF).mean():.0f} %, "
                              f"strong {100 * (h['agn_class'] == 'strong').mean():.0f} %")
        if _lines:
            ax.text(0.98, 0.03, "\n".join(_lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT["note"] - 0.5, color="0.15", zorder=6,
                    linespacing=1.35, path_effects=[withStroke(linewidth=2.6, foreground="white")])
        ax.set_xlim(0, float(np.nanmax(aa)) * 1.05 if np.isfinite(aa).any() else 10); ax.set_ylim(-0.15, max(2.0, float(np.nanmax(ee)) * 1.08 if np.isfinite(ee).any() else 2.0))
        ax.set_xlabel("mass-weighted stellar age at the anchor [Gyr]"); ax.set_ylabel(r"$E_x$ [Gyr]" if i == 0 else "")
        ax.tick_params(direction="in", top=True, right=True, labelsize=FONT["tick"]); ax.grid(False)
        ax.set_title(f"({'abcdef'[_nb + i]}) the dose-age plane: who the dusty galaxies are", loc="left", fontsize=FONT["tag"] - 0.8)
        ax.legend(loc="upper left", frameon=False, fontsize=FONT["legend"] - 1.8, handlelength=1.2, borderaxespad=0.3)
    fig.suptitle(wrap_caption(f"the coupling class crossed with the X-ray class on the dusty tail (log M_dust/M* >= {P5C_TAIL:g}; zero dust at the floor {P9_FLOOR:g}): "
                              f"(top) the tail fraction of every coupling x onset cell (68 % Wilson bars; hatched = weak, solid = strong, the colour = the X-ray class; dotted = the box); "
                              f"(bottom) the dusty galaxies on the dose-age plane against the median $E_x$ of the classified galaxies at that age — a tail galaxy below the line "
                              f"is under-dosed for its age, one on it received the full dose and is dusty anyway; quenched galaxies, {SEL_TXT}; classes = {XTITLE}: "
                              + ", ".join(f"{XSHORT[c]} = {XNAME[c]}" for c in XCLASSES), width=int(13 * fig.get_figwidth())), fontsize=FONT["note"] + 1, y=0.995)
    fig.subplots_adjust(top=0.91, bottom=0.06)
    paper_save(fig, "coupling_onset")
    plt.show()
