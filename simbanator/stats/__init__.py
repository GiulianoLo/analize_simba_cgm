"""Statistics shared across the projects.

Every function takes plain arrays; NaNs are dropped where a statistic needs finite values, and
small samples return NaN (or an empty table) instead of raising. Bootstrap functions take a
``seed`` so a figure redraws identically.

- Two-population scores: :func:`mw_auc`, :func:`auc_rank`, :func:`stratified_auc`,
  :func:`cell_rank`, :func:`auc_within_cells`, :func:`auc_per_column`, :func:`fdr_bh`,
  :func:`cramers_v`, :func:`cramers_v_corrected`, :func:`fisher_2x2`
- Correlations: :func:`spearman`, :func:`partial_spearman`, :func:`partial_corr`, :func:`theil_sen_ci`
- Fractions: :func:`wilson`, :func:`beta_interval`
- Bootstraps and binned summaries: :func:`bootstrap_median`, :func:`group_medians`,
  :func:`grouped_bootstrap_median`, :func:`cluster_bootstrap`, :func:`median_band`,
  :func:`running_median`, :func:`running_quantile`, :func:`binned_median`, :func:`binned_stats`,
  :func:`log_band_with_floor`, :func:`median_iqr`, :func:`jackknife`
- Weights and groups: :func:`weighted_quantile`, :func:`time_weighted_mean`, :func:`terciles`,
  :func:`ridge_residual`
- Densities and modes: :func:`scaled_kde`, :func:`kde_dip`, :func:`n_modes`,
  :func:`critical_bandwidth`, :func:`silverman_test`, :func:`kde_levels`
- Survival: :func:`kaplan_meier`, :func:`km_quantiles`, :func:`log_rank`
"""
import warnings

import numpy as np

__all__ = [
    "mw_auc", "auc_rank", "stratified_auc", "cell_rank", "auc_within_cells", "auc_per_column", "fdr_bh",
    "cramers_v", "cramers_v_corrected", "fisher_2x2",
    "spearman", "partial_spearman", "partial_corr", "theil_sen_ci",
    "wilson", "beta_interval",
    "bootstrap_median", "group_medians", "grouped_bootstrap_median", "cluster_bootstrap", "median_band",
    "running_median", "running_quantile", "binned_median", "binned_stats", "log_band_with_floor",
    "median_iqr", "jackknife",
    "weighted_quantile", "time_weighted_mean", "terciles", "ridge_residual",
    "scaled_kde", "kde_dip", "n_modes", "critical_bandwidth", "silverman_test", "kde_levels",
    "kaplan_meier", "km_quantiles", "log_rank",
]


# ═════════════════════════════════════════════════════════════════════════════
# Two-population scores
# ═════════════════════════════════════════════════════════════════════════════

def mw_auc(x, m1, nmin=3):
    """AUC of ``x`` for population 1 (``m1``) against the rest, with the two-sided Mann-Whitney p.

    AUC = P(x_1 > x_0): 0.5 = no separation, > 0.5 = population 1 sits higher. Returns
    ``(auc, p, n1, n0)``; NaNs when either population has fewer than ``nmin`` finite values or
    all values are identical.
    """
    from scipy.stats import mannwhitneyu
    x = np.asarray(x, float); m1 = np.asarray(m1, bool)
    a, b = x[m1 & np.isfinite(x)], x[~m1 & np.isfinite(x)]
    if len(a) < nmin or len(b) < nmin or np.nanstd(np.concatenate([a, b])) == 0:
        return np.nan, np.nan, len(a), len(b)
    u = mannwhitneyu(a, b, alternative="two-sided")
    return float(u.statistic) / (len(a) * len(b)), float(u.pvalue), len(a), len(b)


def auc_rank(x, y, nmin=3):
    """AUC = P(x | y > x | not y) from ranks alone (no p-value; fast enough inside a bootstrap)."""
    from scipy.stats import rankdata
    x = np.asarray(x, float); y = np.asarray(y, bool)
    fin = np.isfinite(x); x, y = x[fin], y[fin]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < nmin or n0 < nmin:
        return np.nan
    r = rankdata(x)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stratified_auc(x, pos, confounders, nperm=1000, seed=0, nmin_total=40, n_tercile_min=60):
    """AUC of ``x`` for ``pos`` with the confounders held fixed, and its permutation p.

    The sample is cut into the quantile cells of every confounder (terciles, or medians when
    fewer than ``n_tercile_min`` rows), the Mann-Whitney U is summed over the cells, and the
    labels are permuted inside each cell only. Returns ``(auc, two-sided p, n_used)``; NaN when
    the sample is smaller than ``nmin_total`` or either class has fewer than 3 members.
    """
    from scipy.stats import rankdata
    x, pos = np.asarray(x, float), np.asarray(pos, bool)
    ok = np.isfinite(x)
    for c in confounders:
        ok &= np.isfinite(np.asarray(c, float))
    xs, ps = x[ok], pos[ok]
    if len(xs) < nmin_total or ps.sum() < 3 or (~ps).sum() < 3:
        return np.nan, np.nan, int(len(xs))
    nq = 3 if len(xs) >= n_tercile_min else 2
    sid = np.zeros(len(xs), int)
    for c in confounders:
        v = np.asarray(c, float)[ok]
        e = np.unique(np.quantile(v, np.linspace(0, 1, nq + 1)))
        sid = sid * (nq + 1) + np.clip(np.searchsorted(e, v, side="right") - 1, 0, max(len(e) - 2, 0))
    rng = np.random.default_rng(seed)
    num = den = 0.0; pnum = np.zeros(nperm); used = 0
    for u in np.unique(sid):
        m = sid == u
        n1, n0 = int(ps[m].sum()), int((~ps[m]).sum())
        if n1 < 1 or n0 < 1:
            continue
        r = rankdata(xs[m])
        num += float(r[ps[m]].sum()) - n1 * (n1 + 1) / 2; den += n1 * n0; used += int(m.sum())
        pick = np.argsort(rng.random((nperm, int(m.sum()))), axis=1)[:, :n1]
        pnum += r[pick].sum(axis=1) - n1 * (n1 + 1) / 2
    if den == 0:
        return np.nan, np.nan, used
    auc = num / den
    return float(auc), float((np.sum(np.abs(pnum / den - 0.5) >= abs(auc - 0.5)) + 1) / (nperm + 1)), used


def cell_rank(values, controls, q=3, nmin=4):
    """Each value replaced by its rank inside its cell of the control quantiles, in [0, 1].

    The standard way to ask a question at fixed epoch and mass: ``controls`` is a list of
    arrays (e.g. anchor time, log M*), each cut at ``q`` quantiles. Cells with fewer than
    ``nmin`` finite values give NaN.
    """
    y = np.asarray(values, float)
    lab = np.zeros(len(y), int)
    for c in controls:
        v = np.asarray(c, float)
        e = np.nanpercentile(v[np.isfinite(v)], np.linspace(0, 100, q + 1)[1:-1]) if np.isfinite(v).sum() > q else []
        lab = lab * (q + 1) + np.digitize(v, e)
    out = np.full(len(y), np.nan)
    for u in np.unique(lab):
        m = (lab == u) & np.isfinite(y)
        if m.sum() >= nmin:
            out[m] = (np.argsort(np.argsort(y[m])) + 0.5) / m.sum()
    return out


def auc_within_cells(values, m1, controls, q=3, nmin=4):
    """:func:`mw_auc` of the within-cell ranks (:func:`cell_rank`): the AUC at fixed controls."""
    return mw_auc(cell_rank(values, controls, q=q, nmin=nmin), np.asarray(m1, bool))


def auc_per_column(A1, A0, nmin=3):
    """:func:`mw_auc` column by column of two ``(n_gal, n_t)`` stacks -> arrays ``(auc, p, n)`` over the columns."""
    A1, A0 = np.asarray(A1, float), np.asarray(A0, float)
    lab = np.r_[np.ones(len(A1), bool), np.zeros(len(A0), bool)]
    res = [mw_auc(np.r_[A1[:, j], A0[:, j]], lab, nmin=nmin) for j in range(A1.shape[1])]
    auc, p, n1, n0 = (np.array(v) for v in zip(*res)) if res else (np.array([]),) * 4
    return auc, p, n1 + n0


def fdr_bh(p):
    """Benjamini-Hochberg q-values of the p-values ``p`` (NaN stays NaN)."""
    p = np.asarray(p, float); q = np.full(p.shape, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return q
    o = ok[np.argsort(p[ok])]
    r = p[o] * len(o) / np.arange(1, len(o) + 1)
    q[o] = np.clip(np.minimum.accumulate(r[::-1])[::-1], 0, 1)
    return q


def cramers_v(table):
    """Cramér's V of a contingency table (0 = no association, 1 = the row fixes the column)."""
    from scipy.stats import chi2_contingency
    tab = np.asarray(table, float); n = tab.sum()
    if n == 0 or min(tab.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tab)[0]
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def cramers_v_corrected(codes, outcome):
    """Bias-corrected Cramér's V (Bergsma 2013) between a categorical ``codes`` array (< 0 = no class)
    and a binary ``outcome``. The correction makes a 3-class and a 2-class axis comparable."""
    from scipy.stats import chi2_contingency
    codes, outcome = np.asarray(codes), np.asarray(outcome, bool)
    cats = np.unique(codes[codes >= 0])
    if len(cats) < 2:
        return np.nan
    tab = np.array([[np.sum((codes == c) & ~outcome), np.sum((codes == c) & outcome)] for c in cats], float)
    if tab.sum() < 2 or (tab.sum(0) == 0).any() or (tab.sum(1) == 0).any():
        return np.nan
    try:
        chi2 = chi2_contingency(tab, correction=False)[0]
    except ValueError:
        return np.nan
    n = tab.sum(); r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def fisher_2x2(a, b, small_n=20):
    """Fisher's exact test of two boolean arrays -> ``(table, odds ratio, p)``.

    The table is ``[[a&b, a&~b], [~a&b, ~a&~b]]``. With an empty cell and fewer than
    ``small_n`` rows the odds ratio and p are NaN.
    """
    from scipy.stats import fisher_exact
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    tab = np.array([[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]])
    if tab.min() == 0 and tab.sum() < small_n:
        return tab, np.nan, np.nan
    odds, p = fisher_exact(tab)
    return tab, odds, p


# ═════════════════════════════════════════════════════════════════════════════
# Correlations
# ═════════════════════════════════════════════════════════════════════════════

def spearman(x, y, nmin=8):
    """Spearman's rho on the jointly finite pairs -> ``(rho, p, n)``; NaNs below ``nmin`` pairs."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < nmin:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def partial_spearman(x, y, control, nmin=10):
    """Spearman(x, y) with ``control`` regressed out linearly, all on ranks -> ``(rho, p, n)``."""
    from scipy.stats import rankdata, spearmanr
    x, y, c = (np.asarray(v, float) for v in (x, y, control))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if m.sum() < nmin:
        return np.nan, np.nan, int(m.sum())
    rx, ry, rc = rankdata(x[m]), rankdata(y[m]), rankdata(c[m])
    C = np.column_stack([rc, np.ones(int(m.sum()))])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r, p = spearmanr(ex, ey)
    return float(r), float(p), int(m.sum())


def partial_corr(target, driver, controls, nmin=8):
    """Pearson correlation of ``target`` and ``driver`` after regressing both on ``controls`` (+ a constant) -> ``(r, n)``."""
    target = np.asarray(target, float); driver = np.asarray(driver, float)
    M = np.column_stack([np.ones_like(target)] + [np.asarray(c, float) for c in controls])
    g = np.isfinite(target) & np.isfinite(driver) & np.all(np.isfinite(M), axis=1)
    if g.sum() < nmin:
        return np.nan, int(g.sum())
    def resid(v):
        b, *_ = np.linalg.lstsq(M[g], v[g], rcond=None)
        return v[g] - M[g] @ b
    rt, rd = resid(target), resid(driver)
    if np.std(rt) == 0 or np.std(rd) == 0:
        return np.nan, int(g.sum())
    return float(np.corrcoef(rt, rd)[0, 1]), int(g.sum())


def theil_sen_ci(x, y, n=1000, seed=0, nmin=8):
    """Theil-Sen slope with a bootstrap 16-84 % interval -> ``(slope, lo, hi, n)``; None below ``nmin`` points."""
    from scipy.stats import theilslopes
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < nmin:
        return None
    rng, boot = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 2:
            boot.append(theilslopes(y[i], x[i])[0])
    return float(theilslopes(y, x)[0]), float(np.percentile(boot, 16)), float(np.percentile(boot, 84)), len(x)


# ═════════════════════════════════════════════════════════════════════════════
# Fractions
# ═════════════════════════════════════════════════════════════════════════════

def wilson(k, n, z=1.0):
    """k/n with its Wilson score interval at ``z`` sigma (z = 1 -> 68 %) -> ``(p, lo, hi)``.

    The interval is clamped to contain p and to [0, 1]: at k = n the upper end is 1 only up
    to rounding, and a 1e-16 overshoot is a negative error bar that matplotlib rejects.
    """
    k, n = float(k), float(n)
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n; d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h))


def beta_interval(k, n, z=1.0, prior="flat"):
    """k/n with its Beta-posterior credible band at +-z sigma -> ``(p, lo, hi)``.

    ``prior`` is ``"flat"`` (Beta(1, 1)) or ``"jeffreys"`` (Beta(1/2, 1/2)). Vectorised over
    ``k`` for a scalar ``n``; NaN for n <= 0.
    """
    from scipy.stats import beta as beta_dist, norm
    a = {"flat": 1.0, "jeffreys": 0.5}[prior]
    k = np.asarray(k, float); n = float(n)
    if n <= 0:
        return (np.full(k.shape, np.nan),) * 3 if k.ndim else (np.nan,) * 3
    qs = norm.cdf([-abs(z), abs(z)])
    lo = beta_dist.ppf(qs[0], k + a, n - k + a); hi = beta_dist.ppf(qs[1], k + a, n - k + a)
    p = k / n
    return (p, lo, hi) if k.ndim else (float(p), float(lo), float(hi))


# ═════════════════════════════════════════════════════════════════════════════
# Bootstraps and binned summaries
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_median(v, n=1000, seed=0, nmin=1):
    """Median of the finite values with the 16-84 % band of its bootstrap -> ``(median, lo, hi, N)``; NaNs below ``nmin``."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < max(nmin, 1):
        return np.nan, np.nan, np.nan, len(v)
    bs = np.median(v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(bs, 16)), float(np.percentile(bs, 84)), len(v)


def group_medians(values, keys):
    """The median of ``values`` within each group of ``keys`` (e.g. a galaxy's sightlines) -> ``(medians, unique keys)``."""
    values, keys = np.asarray(values, float), np.asarray(keys)
    uk = np.unique(keys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.nanmedian(values[keys == k]) for k in uk]), uk


def grouped_bootstrap_median(values, keys, n=1000, seed=0, nmin=1):
    """Median over groups of the per-group medians, with a bootstrap over the groups -> ``(median, lo, hi, n_groups)``.

    The usual case: several sightline rows per galaxy, so every galaxy counts once.
    """
    v, k = np.asarray(values, float), np.asarray(keys)
    ok = np.isfinite(v); v, k = v[ok], k[ok]
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    uk, inv = np.unique(k, return_inverse=True)
    if len(uk) < nmin:
        return np.nan, np.nan, np.nan, len(uk)
    pg = np.array([np.median(v[inv == i]) for i in range(len(uk))])
    bs = np.median(pg[np.random.default_rng(seed).integers(0, len(pg), (n, len(pg)))], axis=1)
    return float(np.median(pg)), float(np.percentile(bs, 16)), float(np.percentile(bs, 84)), len(pg)


def cluster_bootstrap(keys, statfn, n=1000, seed=0):
    """Bootstrap over groups: ``statfn(row_indices) -> value``, resampling whole groups -> ``(value, lo16, hi84)``.

    All rows of a resampled group (e.g. a galaxy's sightlines) ride along together.
    """
    keys = np.asarray(keys)
    uk = np.unique(keys)
    where = {k: np.where(keys == k)[0] for k in uk}
    rng = np.random.default_rng(seed)
    point = statfn(np.arange(len(keys)))
    s = np.array([statfn(np.concatenate([where[k] for k in rng.choice(uk, len(uk), replace=True)])) for _ in range(n)])
    return float(point), float(np.nanpercentile(s, 16)), float(np.nanpercentile(s, 84))


def median_band(A, nmin=5, nboot=500, seed=11, stat="median", n_columns=None):
    """A ``(n_gal, n_t)`` stack -> the central track, its bootstrap 16-84 % band, and the finite count per column.

    Columns with fewer than ``nmin`` finite values are blanked. ``stat`` is ``"median"`` or
    ``"mean"``. An empty stack gives NaN tracks of length ``n_columns``.
    """
    A = np.asarray(A, float)
    if A.ndim != 2 or not len(A):
        m = n_columns if n_columns is not None else (A.shape[1] if A.ndim == 2 else 0)
        return (np.full(m, np.nan),) * 3 + (np.zeros(m, int),)
    n_t = np.isfinite(A).sum(axis=0)
    f = np.nanmedian if stat == "median" else np.nanmean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cent = f(A, axis=0)
        rng = np.random.default_rng(seed)
        bs = np.array([f(A[rng.integers(0, len(A), len(A))], axis=0) for _ in range(nboot)])
        lo, hi = np.nanpercentile(bs, 16, axis=0), np.nanpercentile(bs, 84, axis=0)
    ok = n_t >= nmin
    return np.where(ok, cent, np.nan), np.where(ok, lo, np.nan), np.where(ok, hi, np.nan), n_t


def _equal_count_bins(x, nbins):
    e = np.quantile(x, np.linspace(0, 1, nbins + 1))
    return np.clip(np.searchsorted(e, x, side="right") - 1, 0, nbins - 1)


def running_median(x, y, nbins=8, nmin=10, nboot=1000, seed=0, keys=None, nmin_small=None):
    """Running median of y in equal-count bins of x, with a bootstrap 16-84 % band -> DataFrame.

    Columns: ``x`` (bin median), ``y``, ``lo``, ``hi``, ``n``. With ``keys`` (one per row, e.g. the
    galaxy of each sightline row) the bootstrap resamples whole groups, ``nmin`` counts groups,
    and the table gains ``n_gal``. With ``nmin_small``, a sample smaller than ``2 * nmin``
    falls back to 4 bins of at least ``min(nmin, nmin_small)``.
    """
    import pandas as pd
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    k = None if keys is None else np.asarray(keys)[ok]
    x, y = x[ok], y[ok]
    cols = ["x", "y", "lo", "hi", "n"] + (["n_gal"] if k is not None else [])
    if nmin_small is not None and len(x) < 2 * nmin:
        nbins, nmin = 4, min(nmin, nmin_small)
    count = len(set(k.tolist())) if k is not None else len(x)
    nb = min(nbins, count // nmin)
    if nb < 2:
        return pd.DataFrame(columns=cols)
    ib = _equal_count_bins(x, nb)
    rng, out = np.random.default_rng(seed), []
    for b in range(nb):
        idx = np.where(ib == b)[0]
        yy = y[idx]
        if k is None:
            bs = np.median(yy[rng.integers(0, len(yy), (nboot, len(yy)))], axis=1)
            out.append((np.median(x[idx]), np.median(yy), np.percentile(bs, 16), np.percentile(bs, 84), len(yy)))
        else:
            uk, inv = np.unique(k[idx], return_inverse=True)
            rows = [idx[inv == i] for i in range(len(uk))]
            bs = [np.median(y[np.concatenate([rows[j] for j in rng.integers(0, len(uk), len(uk))])]) for _ in range(nboot)]
            out.append((np.median(x[idx]), np.median(yy), np.percentile(bs, 16), np.percentile(bs, 84), len(idx), len(uk)))
    return pd.DataFrame(out, columns=cols)


def running_quantile(x, y, q, nbins=8, nmin=10):
    """Running quantile ``q`` of y in equal-count bins of x -> DataFrame (``x``, ``y``, ``n``)."""
    import pandas as pd
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    nb = min(nbins, len(x) // nmin)
    if nb < 2:
        return pd.DataFrame(columns=["x", "y", "n"])
    ib = _equal_count_bins(x, nb)
    return pd.DataFrame([(np.median(x[ib == b]), np.quantile(y[ib == b], q), int((ib == b).sum())) for b in range(nb)],
                        columns=["x", "y", "n"])


def binned_median(x, y, edges, nmin=5, nboot=1000, seed=0):
    """Median of y in the fixed x bins ``edges`` (``(lo, hi]``), with a bootstrap 16-84 % band -> DataFrame.

    When ``edges[0] == edges[1]`` the first bin is ``x == edges[0]`` exactly (a pile at a floor,
    e.g. zero exposure, gets its own bin). Bins with fewer than ``nmin`` points are left out.
    Columns: ``x`` (median of x in the bin), ``y``, ``lo``, ``hi``, ``n``.
    """
    import pandas as pd
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    rng, out = np.random.default_rng(seed), []
    for i in range(len(edges) - 1):
        m = (x == edges[0]) if (i == 0 and edges[0] == edges[1]) else ((x > edges[i]) & (x <= edges[i + 1]))
        if m.sum() < nmin:
            continue
        yy = y[m]
        bs = np.median(yy[rng.integers(0, len(yy), (nboot, len(yy)))], axis=1)
        out.append((float(np.median(x[m])), float(np.median(yy)), float(np.percentile(bs, 16)),
                    float(np.percentile(bs, 84)), int(m.sum())))
    return pd.DataFrame(out, columns=["x", "y", "lo", "hi", "n"])


def binned_stats(x, y, bins, pcts=(25, 75), nmin=3, closed="left", min_bins=0, filled_only=False):
    """Median and percentile band of y per x bin (no bootstrap: the scatter, not the error of the median).

    ``bins`` is an array of edges, or an int for that many equal-count bins (edges at the
    quantiles of the finite x). ``closed`` sets the bin edges: ``"left"`` ``[lo, hi)``,
    ``"both"`` ``[lo, hi]``. Returns a dict of arrays ``x`` (bin centre for fixed edges, median
    of x for equal-count bins), ``y``, ``lo``, ``hi``, ``n``, one entry per bin (NaN below
    ``nmin``), or only the filled bins with ``filled_only``; None when fewer than ``min_bins``
    bins are filled.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = np.isfinite(x) & np.isfinite(y)
    equal = np.ndim(bins) == 0
    e = np.nanquantile(x[g], np.linspace(0, 1, int(bins) + 1)) if equal else np.asarray(bins, float)
    nb = len(e) - 1
    out = {k: np.full(nb, np.nan) for k in ("x", "y", "lo", "hi")}
    out["n"] = np.zeros(nb, int)
    for b in range(nb):
        hi_ok = (x <= e[b + 1]) if closed == "both" else (x < e[b + 1])
        m = g & (x >= e[b]) & hi_ok
        out["n"][b] = int(m.sum())
        out["x"][b] = np.median(x[m]) if (equal and m.any()) else 0.5 * (e[b] + e[b + 1])
        if m.sum() >= nmin:
            out["y"][b], out["lo"][b], out["hi"][b] = np.percentile(y[m], [50, pcts[0], pcts[1]])
    if int(np.isfinite(out["y"]).sum()) < min_bins:
        return None
    if filled_only:
        keep = np.isfinite(out["y"])
        out = {k: v[keep] for k, v in out.items()}
    return out


def log_band_with_floor(A, floor=None, band=(16, 84)):
    """Median and percentile band of log10(A) over the rows of a stack, keeping zeros as a floor.

    ``A`` is ``(n_gal, n_bins)`` (e.g. radial profiles). Zeros are set to ``floor`` instead of
    dropped, so an empty annulus shows as a dip to the floor. ``floor=None`` puts it half a dex
    below the smallest positive value; ``floor=np.nan`` drops zeros. Returns
    ``(median, lo, hi, detected fraction per column, floor used)``.
    """
    A = np.asarray(A, float)
    if floor is None:
        posv = A[A > 0]
        floor = float(np.floor(np.log10(posv.min())) - 0.5) if posv.size else 0.0
    out = np.full(A.shape, np.nan, float)
    out[A > 0] = np.log10(A[A > 0]); out[A == 0] = floor
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(out, axis=0)
        lo = np.nanpercentile(out, band[0], axis=0); hi = np.nanpercentile(out, band[1], axis=0)
    present = np.isfinite(A).sum(0); ndet = (A > 0).sum(0)
    frac = np.where(present > 0, ndet / np.maximum(present, 1), np.nan)
    return med, lo, hi, frac, floor


def median_iqr(v):
    """``(median, median - q25, q75 - median, n)`` of the finite values (errorbar-ready offsets)."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, 0.0, 0.0, 0
    md = np.median(v)
    return float(md), float(md - np.percentile(v, 25)), float(np.percentile(v, 75) - md), len(v)


def jackknife(values_per_unit, estimator):
    """Delete-one jackknife over sub-volume units (e.g. the 8 octants of a box).

    ``values_per_unit`` is a dict ``unit -> data``; ``estimator(list of kept units) -> scalar``.
    Returns ``(estimate with all units, jackknife sigma)``; sigma is NaN with fewer than 2 units.
    """
    keys = sorted(values_per_unit)
    theta_all = estimator(keys)
    if len(keys) < 2:
        return theta_all, np.nan
    thetas = np.array([estimator([k for k in keys if k != kk]) for kk in keys])
    K = len(keys)
    return theta_all, float(np.sqrt((K - 1) / K * np.sum((thetas - thetas.mean()) ** 2)))


# ═════════════════════════════════════════════════════════════════════════════
# Weights and groups
# ═════════════════════════════════════════════════════════════════════════════

def weighted_quantile(v, w, q=0.5):
    """Quantile ``q`` of ``v`` with weights ``w`` (midpoint CDF; non-finite and non-positive weights dropped)."""
    v = np.asarray(v, float); w = np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.nan
    v, w = v[ok], w[ok]
    o = np.argsort(v); v, w = v[o], w[o]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def time_weighted_mean(t, y):
    """Time-weighted mean of y(t), trapezoidal: ``∫ y dt / (t_last - t_first)``. Duplicate times are dropped."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    t, y = t[ok], y[ok]
    o = np.argsort(t); t, y = t[o], y[o]
    keep = np.r_[True, np.diff(t) > 0]
    t, y = t[keep], y[keep]
    if len(t) < 2 or t[-1] <= t[0]:
        return np.nan
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, t) / (t[-1] - t[0]))


def terciles(v, groups=None, nmin=3):
    """0 / 1 / 2 = low / mid / high tercile of ``v`` (NaN -> -1), edges at the 1/3 and 2/3 quantiles.

    With ``groups`` (one label per row, e.g. the anchor snapshot) the edges are computed within
    each group; groups with fewer than ``nmin`` finite values stay -1. Returns ``(labels, edges)``
    where ``edges`` is the global pair, or a dict ``group -> pair``.
    """
    v = np.asarray(v, float)
    out = np.full(len(v), -1, int)
    labels = np.zeros(len(v), int) if groups is None else np.asarray(groups)
    edges = {}
    for g in np.unique(labels):
        m = (labels == g) & np.isfinite(v)
        if m.sum() < nmin:
            continue
        e = np.quantile(v[m], [1.0 / 3.0, 2.0 / 3.0])
        out[m] = np.searchsorted(e, v[m], side="right")
        edges[g if groups is not None else None] = e
    return out, (edges.get(None) if groups is None else edges)


def ridge_residual(x, y, z, nzbins=3, nxbins=8, nmin=10):
    """Residual of y about the running-median y|x ridge, the ridge built in redshift bins.

    E.g. Δlog M_halo at fixed log M* (``x`` = log M*, ``y`` = log M_halo, ``z`` = anchor redshift).
    All rows in one redshift bin when z spans less than 0.05. Returns ``(residuals, ridge)``,
    ``ridge[k] = (z_lo, z_hi, x_mids, y_medians)``.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    res, ridge = np.full(x.shape, np.nan), {}
    g = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if g.sum() < 2 * nmin:
        return res, ridge
    if np.nanmax(z[g]) - np.nanmin(z[g]) < 0.05:
        nzbins = 1
    zed = np.nanquantile(z[g], np.linspace(0, 1, nzbins + 1)); zed[-1] += 1e-9
    for k in range(nzbins):
        inz = g & (z >= zed[k]) & (z < zed[k + 1])
        if inz.sum() < nmin:
            continue
        xe = np.nanquantile(x[inz], np.linspace(0, 1, nxbins + 1))
        mids, meds = [], []
        for b in range(nxbins):
            inb = inz & (x >= xe[b]) & (x <= xe[b + 1])
            if inb.sum() >= 5:
                mids.append(np.median(x[inb])); meds.append(np.median(y[inb]))
        if len(mids) < 3:
            continue
        ridge[k] = (float(zed[k]), float(zed[k + 1]), np.asarray(mids), np.asarray(meds))
        res[inz] = y[inz] - np.interp(x[inz], mids, meds)
    return res, ridge


# ═════════════════════════════════════════════════════════════════════════════
# Densities and modes
# ═════════════════════════════════════════════════════════════════════════════

def scaled_kde(v, grid, bw_factor=1.0, nmin=5):
    """Gaussian KDE of ``v`` on ``grid`` scaled to its own maximum (Scott's bandwidth x ``bw_factor``); None when degenerate."""
    from scipy.stats import gaussian_kde
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < nmin or np.std(v) <= 0:
        return None
    try:
        d = gaussian_kde(v)
        d.set_bandwidth(d.factor * bw_factor)
        y = d(grid)
    except Exception:
        return None
    m = float(np.max(y))
    return y / m if m > 0 else None


def kde_dip(x, bw=None, ngrid=512):
    """The KDE of x and the minimum between its two highest maxima.

    Returns ``(dip location, grid, density, number of modes)``; the dip is NaN with one mode,
    and a degenerate sample (all values identical) gives empty arrays and 0 modes.
    """
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    try:
        kde = gaussian_kde(x, bw_method=bw)
    except Exception:
        return np.nan, np.array([]), np.array([]), 0
    gx = np.linspace(x.min() - 0.5 * x.std(), x.max() + 0.5 * x.std(), ngrid)
    d = kde(gx)
    pk = np.flatnonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])) + 1
    if len(pk) < 2:
        return np.nan, gx, d, len(pk)
    a, b = sorted(pk[np.argsort(d[pk])[-2:]])
    return float(gx[a + int(np.argmin(d[a:b + 1]))]), gx, d, len(pk)


def n_modes(x, h, ngrid=512):
    """Number of modes of the Gaussian KDE of x at bandwidth ``h`` -> ``(n, grid, density)``."""
    x = np.asarray(x, float)
    gx = np.linspace(x.min() - 3 * h, x.max() + 3 * h, ngrid)
    d = np.exp(-0.5 * ((gx[:, None] - x[None, :]) / h) ** 2).sum(axis=1)
    return int(((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])).sum()), gx, d


def critical_bandwidth(x, k=1, lo=None, hi=None, iterations=40, ngrid=512):
    """Silverman's critical bandwidth: the smallest h whose KDE has at most ``k`` modes (bisection)."""
    x = np.asarray(x, float)
    s = np.std(x, ddof=1)
    lo = s * 1e-3 if lo is None else lo
    hi = s * 3.0 if hi is None else hi
    while n_modes(x, hi, ngrid)[0] > k and hi < s * 100:
        hi *= 2.0
    for _ in range(iterations):
        m = np.sqrt(lo * hi)
        if n_modes(x, m, ngrid)[0] > k:
            lo = m
        else:
            hi = m
    return hi


def silverman_test(x, k=1, nboot=200, seed=0, nmin=20, ngrid=512):
    """Silverman's bootstrap test of H0 'at most k modes' -> ``(p, critical bandwidth)``.

    Resamples from the density smoothed at the critical bandwidth (rescaled to keep the
    variance) and counts how often a larger critical bandwidth is needed. A small p means
    ``k`` modes are not enough.
    """
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < nmin or nboot <= 0:
        return np.nan, np.nan
    h = critical_bandwidth(x, k, ngrid=ngrid)
    s, xb, rng = np.std(x, ddof=1), x.mean(), np.random.default_rng(seed)
    cnt = 0
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        y = xb + (x[i] - xb + h * rng.standard_normal(len(x))) / np.sqrt(1.0 + h * h / (s * s))
        cnt += int(critical_bandwidth(y, k, ngrid=ngrid) > h)
    return float(cnt) / nboot, float(h)


def kde_levels(x, y, xlim, ylim, fractions, n=160):
    """2-D Gaussian KDE on an ``n x n`` grid and the density levels enclosing the given mass ``fractions``.

    Returns ``(X, Y, Z, levels)`` ready for ``contour(X, Y, Z, levels)``.
    """
    from scipy.stats import gaussian_kde
    k = gaussian_kde(np.vstack([x, y]))
    gx = np.linspace(xlim[0], xlim[1], n); gy = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(gx, gy)
    Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    f = np.sort(Z.ravel())[::-1]; cs = np.cumsum(f) / f.sum()
    lev = sorted({float(f[min(int(np.searchsorted(cs, q)), len(f) - 1)]) for q in fractions})
    return X, Y, Z, lev


# ═════════════════════════════════════════════════════════════════════════════
# Survival
# ═════════════════════════════════════════════════════════════════════════════

def kaplan_meier(t, event):
    """Kaplan-Meier survival curve with Greenwood's standard error.

    ``event`` False = censored: the row contributes its time at risk and no event (e.g. a galaxy
    still dusty at its anchor). Negative or non-finite times are dropped. Returns
    ``(times, S, se, n)`` starting at ``(0, 1, 0)``.
    """
    t, ev = np.asarray(t, float), np.asarray(event, bool)
    m = np.isfinite(t) & (t >= 0)
    t, ev = t[m], ev[m]
    if not len(t):
        return np.array([0.0]), np.array([1.0]), np.array([0.0]), 0
    o = np.argsort(t); t, ev = t[o], ev[o]
    ts, S, se, s, v, n = [0.0], [1.0], [0.0], 1.0, 0.0, len(t)
    for u in np.unique(t[ev]):
        n_at = int((t >= u).sum()); d = int(((t == u) & ev).sum())
        if n_at <= 0:
            continue
        s *= 1.0 - d / n_at
        v += d / (n_at * max(n_at - d, 1))
        ts.append(float(u)); S.append(s); se.append(s * np.sqrt(v))
    return np.asarray(ts), np.asarray(S), np.asarray(se), n


def km_quantiles(t, event, levels=(0.5, 0.75, 0.25)):
    """Times at which the Kaplan-Meier curve first drops to each survival level (NaN if it never does) + n.

    The default gives ``(median, time a quarter has fallen, time three quarters have fallen, n)``.
    """
    ts, S, _, n = kaplan_meier(t, event)
    q = []
    for lev in levels:
        i = np.flatnonzero(S <= lev)
        q.append(float(ts[i[0]]) if len(i) else np.nan)
    return (*q, n)


def log_rank(t, event, group1):
    """Log-rank test of two groups under censoring -> ``(chi2, p)`` (1 dof); NaNs when undefined."""
    from scipy.stats import chi2 as chi2_dist
    t, ev, g1 = np.asarray(t, float), np.asarray(event, bool), np.asarray(group1, bool)
    m = np.isfinite(t) & (t >= 0)
    t, ev, g1 = t[m], ev[m], g1[m]
    O = E = V = 0.0
    for u in np.unique(t[ev]):
        n, n1 = int((t >= u).sum()), int(((t >= u) & g1).sum())
        d, d1 = int(((t == u) & ev).sum()), int(((t == u) & ev & g1).sum())
        if n < 2 or d < 1:
            continue
        O += d1; E += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / max(n - 1, 1)
    if V <= 0:
        return np.nan, np.nan
    chi2 = (O - E) ** 2 / V
    return float(chi2), float(chi2_dist.sf(chi2, 1))
