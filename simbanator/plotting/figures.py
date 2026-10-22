"""Figure mechanics shared by the notebooks: saving, paper style, panel grids, labels, captions,
axis spaces, robust limits, and the median-track / significance-strip drawings.

Only matplotlib and numpy; nothing here reads data products.
"""
import contextlib
import functools
import os
import re

import numpy as np

__all__ = [
    "PAPER_RC", "paper_style", "save_figure", "figure_path", "apply_font_floor",
    "panel_grid", "equalise_rows", "panel_letter", "panel_label", "zone_frame", "quiet_legend",
    "scalebar", "caption_lines", "wrap_caption",
    "to_plot_space", "from_plot_space", "robust_limits", "clipped_limits",
    "draw_track", "stage_medians", "draw_median_track", "significance_strip",
    "count_histogram_behind", "kde_contour_or_scatter", "ks_axes",
]

# The paper figures: large text, no grids (feedback: no titles, large labels, minimal in-panel text).
PAPER_RC = {"font.size": 18, "axes.labelsize": 21, "xtick.labelsize": 18, "ytick.labelsize": 18,
            "legend.fontsize": 17, "axes.titlesize": 18, "axes.grid": False}


@contextlib.contextmanager
def paper_style(rc=None):
    """``with paper_style(): ...`` draws with :data:`PAPER_RC` (plus any overrides in ``rc``)."""
    import matplotlib as mpl
    with mpl.rc_context({**PAPER_RC, **(rc or {})}):
        yield


# ═════════════════════════════════════════════════════════════════════════════
# Saving
# ═════════════════════════════════════════════════════════════════════════════

def save_figure(fig, stem, outdir=None, formats=("png", "pdf"), dpi=200, prefix="", verbose=True):
    """Save ``fig`` as ``<outdir>/<prefix><stem>.<ext>`` for every format; returns the paths.

    ``stem`` may already be a path (then ``outdir`` is omitted). The folder is created if needed.
    """
    base = os.path.join(outdir, prefix + stem) if outdir else (os.path.join(os.path.dirname(stem), prefix + os.path.basename(stem)))
    d = os.path.dirname(base)
    if d:
        os.makedirs(d, exist_ok=True)
    paths = []
    for ext in formats:
        p = f"{base}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    if verbose:
        print("figure ->", f"{base}.{{{','.join(formats)}}}")
    return paths


def figure_path(plotdir, name, router=None, default="99_misc"):
    """Path for figure ``name`` inside ``plotdir``; with ``router`` = ``[(filename prefix, subfolder), ...]``
    the figure goes into the first matching subfolder (``default`` otherwise), created on demand."""
    base = os.path.basename(name)
    if router is None:
        return os.path.join(plotdir, base)
    sub = next((s for pre, s in router if base.startswith(pre)), default)
    d = os.path.join(plotdir, sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, base)


def apply_font_floor(floor, classes=None, methods=("text", "set_xlabel", "set_ylabel", "set_title", "legend",
                                                    "annotate", "suptitle", "set_xticklabels", "set_yticklabels")):
    """Raise every numeric ``fontsize=`` / ``size=`` below ``floor`` to ``floor`` (strings like 'small' are left alone).

    Patches the drawing methods of matplotlib's ``Axes`` and ``Figure`` (or ``classes``) once;
    calling it again changes nothing.
    """
    import matplotlib.axes, matplotlib.figure
    classes = classes or (matplotlib.axes.Axes, matplotlib.figure.Figure)

    def bump(fs):
        if isinstance(fs, (int, float)) and not isinstance(fs, bool) and fs < floor:
            return floor
        return fs

    for cls in classes:
        for nm in methods:
            orig = getattr(cls, nm, None)
            if orig is None or getattr(orig, "_fontfloor", False):
                continue

            @functools.wraps(orig)
            def wrapper(*a, __orig=orig, **kw):
                for key in ("fontsize", "size"):
                    if key in kw:
                        kw[key] = bump(kw[key])
                return __orig(*a, **kw)
            wrapper._fontfloor = True
            setattr(cls, nm, wrapper)


# ═════════════════════════════════════════════════════════════════════════════
# Grids, labels, captions
# ═════════════════════════════════════════════════════════════════════════════

def panel_grid(nrows, column_names, height=3.1, width=4.4, header="text"):
    """``nrows x len(column_names)`` panels, inward ticks on all sides, no grid, the column names on the first row.

    ``header="text"`` writes each name above its panel; ``header="title"`` uses ``set_title``.
    Returns ``(fig, axs)`` with ``axs`` always 2-D.
    """
    import matplotlib.pyplot as plt
    ncol = len(column_names)
    fig, axs = plt.subplots(nrows, ncol, figsize=(width * ncol, height * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.grid(False)
        ax.tick_params(direction="in", top=True, right=True)
    for j, name in enumerate(column_names):
        if header == "title":
            axs[0, j].set_title(name, fontsize=9.5, loc="left", pad=16)
        else:
            axs[0, j].text(0.0, 1.02, name, transform=axs[0, j].transAxes, ha="left", va="bottom", fontsize=9.5, color="0.25")
    return fig, axs


def equalise_rows(fig, axs, frame=True, pad=0.008):
    """Common x and y limits along every row of ``axs`` (so columns compare), then ``tight_layout``.

    Rows whose axes carry ``_auto_y = True`` take their y range from the drawn lines (plus 15 %),
    so bands that run off to outliers do not set it; a log y axis is spanned in log space.
    Inverted y axes keep their sense. With ``frame``, a thin box is drawn around the first column.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for row in axs:
        row_ = [ax for ax in row if ax.has_data()]
        if len(row_) <= 1:
            continue
        if all(getattr(a, "_auto_y", False) for a in row_):
            logy = row_[0].get_yscale() == "log"
            ys = np.concatenate([np.asarray(l.get_ydata(), float) for a in row_ for l in a.get_lines()
                                 if len(l.get_ydata()) > 1 and l.get_transform() == a.transData] or [np.array([])])
            ys = ys[np.isfinite(ys) & ((ys > 0) if logy else True)]
            if logy:
                ys = np.log10(ys)
            if ys.size:
                lo, hi = ys.min(), ys.max()
                lo, hi = lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo)
                for a in row_:
                    a.set_ylim(*((10 ** lo, 10 ** hi) if logy else (lo, hi)))
        else:
            lims = np.array([a.get_ylim() for a in row_])
            lo, hi = (lims.min(), lims.max()) if lims[0, 0] <= lims[0, 1] else (lims.max(), lims.min())
            for a in row_:
                a.set_ylim(lo, hi)
        lims = np.array([a.get_xlim() for a in row_])
        for a in row_:
            a.set_xlim(lims.min(), lims.max())
    plt.tight_layout(w_pad=1.2, h_pad=1.0)
    if frame:
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        bbs = [ax.get_tightbbox(fig.canvas.get_renderer()).transformed(inv) for ax in axs[:, 0]]
        x0, x1 = min(b.x0 for b in bbs) - pad, max(b.x1 for b in bbs) + pad
        y0, y1 = min(b.y0 for b in bbs) - pad, max(b.y1 for b in bbs) + pad * 1.6
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fill=False, ec="0.4", lw=0.9, zorder=20))
    return fig


def panel_letter(ax, i, j, text, fontsize=9):
    """Title ``"(a1) text"`` at the right of the panel: row letter ``i``, column number ``j + 1``."""
    ax.set_title(f"({'abcdefghijklmnop'[i]}{j + 1}) {text}", fontsize=fontsize, loc="right", pad=3 if i else 4)


def panel_label(ax, text, color="w", loc="left", fontsize=14, stroke=True):
    """Bold label in the top corner of an image panel, with a dark outline so it reads on any image."""
    from matplotlib.patheffects import withStroke
    x, ha = (0.04, "left") if loc == "left" else (0.96, "right")
    ax.text(x, 0.94, text, transform=ax.transAxes, ha=ha, va="top", color=color, fontsize=fontsize, fontweight="bold",
            path_effects=[withStroke(linewidth=2.5, foreground="k")] if stroke else None)


def zone_frame(ax, color, tag=None, tag_fontsize=10):
    """Colour a panel's spines and ticks; with ``tag``, a rotated label box left of the y axis (for the first column)."""
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.7)
    ax.tick_params(color=color)
    if tag:
        ax.text(-0.36, 0.5, tag, transform=ax.transAxes, rotation=90, ha="center", va="center", color="white",
                fontsize=tag_fontsize, fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"), zorder=30)


def quiet_legend(ax, loc="best", fontsize=8, **kw):
    """A frameless legend (the notebooks' default look)."""
    kw.setdefault("handlelength", 1.8)
    return ax.legend(frameon=False, fontsize=fontsize, loc=loc, **kw)


def scalebar(ax, extent, length_kpc, color="w", lw=2.5, fontsize=11, pad=0.06):
    """A horizontal scale bar in the lower-left corner of an image with ``extent = (x0, x1, y0, y1)`` in kpc.

    The label reads pc below 1 kpc and Mpc from 1000 kpc.
    """
    dx = extent[1] - extent[0]; dy = extent[3] - extent[2]
    x0 = extent[0] + pad * dx; y0 = extent[2] + pad * dy
    ax.plot([x0, x0 + length_kpc], [y0, y0], lw=lw, color=color, solid_capstyle="butt")
    if length_kpc >= 1000:
        label = f"{length_kpc / 1000:.0f} Mpc"
    elif length_kpc >= 1:
        label = f"{length_kpc:.0f} kpc"
    else:
        label = f"{length_kpc * 1e3:.0f} pc"
    ax.text(x0 + length_kpc / 2, y0 + 0.035 * dy, label, ha="center", va="bottom", color=color, fontsize=fontsize)


def caption_lines(clauses, width=150, sep="; "):
    """Join caption clauses into lines of at most ``width`` characters, never splitting a clause (mathtext stays balanced)."""
    lines, cur = [], ""
    for c in clauses:
        if cur and len(cur) + len(sep) + len(c) > width:
            lines.append(cur); cur = c
        else:
            cur = (cur + sep + c) if cur else c
    return "\n".join(lines + ([cur] if cur else []))


def wrap_caption(text, width=150):
    """Greedy line wrap at ``width`` characters that never splits a ``$...$`` mathtext span (existing newlines kept).

    Long one-line captions widen the tight bounding box and shrink the panels.
    """
    out = []
    for para in text.split("\n"):
        words, cur = [], ""
        for t in re.findall(r"\$[^$]*\$|[^\s$]+|\s+", para):
            if t.isspace():
                if cur:
                    words.append(cur); cur = ""
            else:
                cur += t
        if cur:
            words.append(cur)
        line = ""
        for w in words:
            if line and len(line) + 1 + len(w) > width:
                out.append(line); line = w
            else:
                line = (line + " " + w) if line else w
        out.append(line)
    return "\n".join(out)


# ═════════════════════════════════════════════════════════════════════════════
# Axis spaces and limits
# ═════════════════════════════════════════════════════════════════════════════

def to_plot_space(spec, v):
    """Raw values -> plot space for an axis ``spec = {"log": bool, "floor": optional}``.

    A log axis takes log10; with a floor, values below it sit at the floor, otherwise
    non-positive values become NaN. A linear axis returns the values unchanged.
    """
    v = np.asarray(v, float)
    if not spec["log"]:
        return v
    fl = spec.get("floor")
    with np.errstate(divide="ignore", invalid="ignore"):
        if fl:
            return np.where(np.isfinite(v), np.log10(np.maximum(v, fl)), np.nan)
        return np.where(np.isfinite(v) & (v > 0), np.log10(np.where(v > 0, v, 1.0)), np.nan)


def from_plot_space(spec, v):
    """Inverse of :func:`to_plot_space` (10**v on a log axis)."""
    return 10 ** np.asarray(v, float) if spec["log"] else np.asarray(v, float)


def robust_limits(*arrays, k=1.7, pad=0.06, zero_iqr_fallback=True):
    """Axis range over the pooled finite values: Tukey fences (quartiles +- k IQR) clipped to the data, plus ``pad``.

    Outliers cannot stretch the view. None with fewer than 4 values or a zero range. A zero
    IQR falls back to the 1-99 % range (or gives None with ``zero_iqr_fallback=False``).
    """
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrays]) if arrays else np.array([])
    v = v[np.isfinite(v)]
    if v.size < 4:
        return None
    q1, q3 = np.percentile(v, [25, 75])
    if q3 > q1:
        lo = max(float(v.min()), q1 - k * (q3 - q1))
        hi = min(float(v.max()), q3 + k * (q3 - q1))
    elif zero_iqr_fallback:
        lo, hi = np.percentile(v, [1, 99])
    else:
        return None
    if not hi > lo:
        return None
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def clipped_limits(v, log=False, nsig=3.5, pad=0.08, nmin=10):
    """Axis range after a MAD sigma clip, computed in plot space and returned in data units (None below ``nmin`` values)."""
    v = np.asarray(v, float)
    if log:
        with np.errstate(all="ignore"):
            v = np.log10(np.where(v > 0, v, np.nan))
    v = v[np.isfinite(v)]
    if v.size < nmin:
        return None
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    keep = v[np.abs(v - med) < nsig * max(mad, 1e-12)]
    if keep.size < 5:
        keep = v
    lo, hi = float(keep.min()), float(keep.max()); span = max(hi - lo, 1e-6)
    lo, hi = lo - pad * span, hi + pad * span
    return (10 ** lo, 10 ** hi) if log else (lo, hi)


# ═════════════════════════════════════════════════════════════════════════════
# Tracks and strips
# ═════════════════════════════════════════════════════════════════════════════

def draw_track(ax, track, xspec, yspec, color, lw=2.8, ls="-", marker=None, band=True, zorder=4):
    """A running-median track (columns ``x``, ``y``, ``lo``, ``hi`` in plot space) drawn in data units, white halo, optional band."""
    from matplotlib.patheffects import withStroke
    X, Y = from_plot_space(xspec, track["x"]), from_plot_space(yspec, track["y"])
    if band:
        ax.fill_between(X, from_plot_space(yspec, track["lo"]), from_plot_space(yspec, track["hi"]),
                        color=color, alpha=0.14, lw=0, zorder=zorder - 1)
    ax.plot(X, Y, color=color, lw=lw, ls=ls, marker=marker, ms=7, mec="k", mew=0.6, zorder=zorder,
            path_effects=[withStroke(linewidth=lw + 1.8, foreground="white")])


def stage_medians(T, stages, xcol="logSigmaH2", ycol="logSigmaSFR", nmin=5, stage_col="stage", ul_col="is_ul"):
    """Per-stage medians of x and y in a table of (galaxy, stage) rows.

    For each stage: ``x``, ``y`` and their 16 / 84 % over the uncensored rows (present when at
    least ``nmin``), and ``x_all``, ``y_all`` over all finite rows, with censored y at its floor:
    an upper limit on the stage median. Also the counts ``n``, ``n_ul``, ``n_all``.
    """
    from ..io.tables import as_str
    stg, ul = as_str(T[stage_col]), np.asarray(T[ul_col], bool)
    x, y = np.asarray(T[xcol], float), np.asarray(T[ycol], float)
    out = []
    for st in stages:
        m_all = (stg == st) & np.isfinite(x) & np.isfinite(y)
        m = m_all & ~ul
        d = dict(stage=st, n=int(m.sum()), n_ul=int(((stg == st) & np.isfinite(x) & ul).sum()), n_all=int(m_all.sum()))
        if m_all.sum() >= nmin:
            d.update(x_all=np.median(x[m_all]), y_all=np.median(y[m_all]))
        if m.sum() >= nmin:
            d.update(x=np.median(x[m]), x16=np.percentile(x[m], 16), x84=np.percentile(x[m], 84),
                     y=np.median(y[m]), y16=np.percentile(y[m], 16), y84=np.percentile(y[m], 84))
        out.append(d)
    return out


def draw_median_track(ax, T, color, stages, markers, end_stages=(), xcol="logSigmaH2", ycol="logSigmaSFR",
                      nmin=5, lw=2.2, ls="-", zorder=6, ms=1.0):
    """The :func:`stage_medians` of ``T`` joined in stage order, one marker per stage (``markers[stage]``).

    A stage with too few uncensored rows but enough in total is drawn at its all-rows median as
    an open marker with a down arrow (an upper limit). ``end_stages`` get larger, black-edged
    markers. Returns the medians.
    """
    meds = stage_medians(T, stages, xcol=xcol, ycol=ycol, nmin=nmin)
    pts = [(m["x"], m["y"], m["stage"], False) if "x" in m else (m["x_all"], m["y_all"], m["stage"], True)
           for m in meds if ("x" in m) or ("x_all" in m)]
    if len(pts) >= 2:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=color, lw=lw, alpha=0.9, zorder=zorder)
    for x, y, st, is_ul in pts:
        s = (110 if st in end_stages else 70) * ms
        if is_ul:
            ax.scatter(x, y, s=s, marker=markers[st], facecolors="white", edgecolors=color, linewidths=1.4, zorder=zorder + 1)
            ax.annotate("", xy=(x, y - 0.35), xytext=(x, y), zorder=zorder + 1,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=9, shrinkA=4, shrinkB=0))
        else:
            ax.scatter(x, y, s=s, marker=markers[st], c=[color], edgecolors="k" if st in end_stages else "none",
                       linewidths=0.9, zorder=zorder + 1)
    return meds


def significance_strip(ax, edges, values_a, values_b, y0, y1, nmin=5):
    """A strip in axes-y ``[y0, y1]``: per x bin, the Mann-Whitney p of two groups as a shade
    (black p < 0.001, dark grey < 0.01, light grey < 0.05, nothing otherwise).

    ``values_a[i]`` / ``values_b[i]`` are the per-galaxy values in bin ``[edges[i], edges[i+1])``.
    Returns the p-values (NaN where a group has fewer than ``nmin``).
    """
    from scipy.stats import mannwhitneyu
    ps = []
    for (lo, hi), a, b in zip(zip(edges[:-1], edges[1:]), values_a, values_b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < nmin or len(b) < nmin:
            ps.append(np.nan); continue
        p = mannwhitneyu(a, b).pvalue
        ps.append(p)
        col = "k" if p < 1e-3 else ("0.55" if p < 0.01 else ("0.85" if p < 0.05 else None))
        if col:
            ax.fill_between([lo, hi], y0, y1, color=col, lw=0, transform=ax.get_xaxis_transform())
    return np.array(ps)


def count_histogram_behind(ax, x, counts_by_group, colors, headroom=2.8, label=True):
    """The number of contributing galaxies per x as faint bars behind the tracks, on a hidden twin axis.

    ``counts_by_group`` maps group -> counts (same length as ``x``); ``colors`` maps group -> colour.
    Returns the twin axis.
    """
    axb = ax.twinx(); axb.set_zorder(ax.get_zorder() - 1); ax.patch.set_visible(False)
    top = max(1.0, max((np.nanmax(c) if len(c) else 0.0) for c in counts_by_group.values()))
    w = 0.92 * (x[1] - x[0]) if len(x) > 1 else 0.9
    for g, c in counts_by_group.items():
        axb.bar(x, c, width=w, color=colors[g], alpha=0.13, lw=0, align="center", zorder=1)
    axb.set_ylim(0, top * headroom); axb.set_yticks([])
    if label:
        ax.text(0.985, 0.965, f"N $\\leq$ {int(top)}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.45")
    return axb


def kde_contour_or_scatter(ax, x, y, color, label=None, nmin=12, levels=4):
    """2-D KDE contours of (x, y) in ``color``, or a scatter when there are fewer than ``nmin`` finite points."""
    from scipy.stats import gaussian_kde
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) >= nmin:
        try:
            k = gaussian_kde(np.vstack([x[m], y[m]]))
            xi = np.linspace(np.nanpercentile(x[m], 1), np.nanpercentile(x[m], 99), 60)
            yi = np.linspace(np.nanpercentile(y[m], 1), np.nanpercentile(y[m], 99), 60)
            XX, YY = np.meshgrid(xi, yi)
            ZZ = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contour(XX, YY, ZZ, levels=levels, colors=color, linewidths=1.4, alpha=0.9)
            ax.plot([], [], color=color, lw=1.4, label=label)
            return "contour"
        except Exception:
            pass
    ax.scatter(x[m], y[m], s=12, c=color, alpha=0.5, edgecolor="none", label=label)
    return "scatter"


def ks_axes(ax, aperture_title, half_mass=True, he_factor=1.36, xlim=(-0.5, 3.6), ylim=(-4.6, 1.1),
            sfr_note="SFR over 100 Myr (stars)", grid=False):
    """Axis labels and limits of a Kennicutt-Schmidt panel (log Σ_H2 [Msun/pc²] vs log Σ_SFR [Msun/yr/kpc²]).

    ``half_mass`` labels the half-mass convention (0.5 M / π R50²) rather than M(<r)/π r²; the He factor
    on the simulated H2 is written into the x label.
    """
    conv = r"$0.5M/\pi R_{50}^2$" if half_mass else r"$M(<r)/\pi r^2$"
    ax.set_xlabel(r"$\log(\Sigma_{\rm H_2}/M_\odot\,{\rm pc}^{-2})$  [" + aperture_title + ", " + conv + rf", $\times{he_factor:g}$ He]")
    ax.set_ylabel(r"$\log(\Sigma_{\rm SFR}/M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2})$  [" + sfr_note + "]")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(grid)
