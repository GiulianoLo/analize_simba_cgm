"""Tests for simbanator.stats with cases whose answer is known."""
import numpy as np
import pytest

from simbanator import stats as S


def test_mw_auc_separation_and_guards():
    x = np.r_[np.arange(10) + 100.0, np.arange(10)]
    m = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    auc, p, n1, n0 = S.mw_auc(x, m)
    assert auc == 1.0 and p < 1e-3 and (n1, n0) == (10, 10)
    assert S.mw_auc(-x, m)[0] == 0.0
    assert np.isnan(S.mw_auc(np.ones(20), m)[0])                      # identical values
    assert np.isnan(S.mw_auc(x[:4], m[:4] | True)[0])                  # one population empty
    assert S.auc_rank(x, m) == pytest.approx(1.0)


def test_stratified_auc_and_cells():
    rng = np.random.default_rng(0)
    c = rng.normal(size=300)
    pos = rng.random(300) < 0.5
    x = c + 3.0 * pos                                                  # separates at any fixed c
    auc, p, n = S.stratified_auc(x, pos, [c], nperm=200)
    assert auc > 0.9 and p < 0.01 and n == 300
    y = c + 0.0 * pos                                                  # only the confounder
    assert abs(S.stratified_auc(y, pos, [c], nperm=200)[0] - 0.5) < 0.1
    r = S.cell_rank(x, [c], q=3)
    assert np.nanmin(r) > 0 and np.nanmax(r) < 1
    assert S.auc_within_cells(x, pos, [c])[0] > 0.9


def test_fdr_bh_known_values():
    q = S.fdr_bh([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(q[:3], [0.03, 0.04, 0.04])
    assert np.isnan(q[3])


def test_association_measures():
    assert S.cramers_v([[10, 0], [0, 10]]) == pytest.approx(1.0, abs=0.1)
    codes = np.r_[np.zeros(50, int), np.ones(50, int)]
    assert S.cramers_v_corrected(codes, codes.astype(bool)) > 0.9
    tab, odds, p = S.fisher_2x2(np.r_[np.ones(20, bool), np.zeros(20, bool)], np.r_[np.ones(20, bool), np.zeros(20, bool)])
    assert tab.tolist() == [[20, 0], [0, 20]] and p < 1e-6


def test_correlations():
    x = np.arange(30.0); y = x ** 3
    assert S.spearman(x, y)[0] == pytest.approx(1.0)
    assert np.isnan(S.spearman(x[:5], y[:5])[0])
    rng = np.random.default_rng(1)
    c = rng.normal(size=200); a = c + 0.1 * rng.normal(size=200); b = c + 0.1 * rng.normal(size=200)
    assert S.spearman(a, b)[0] > 0.9 and abs(S.partial_spearman(a, b, c)[0]) < 0.3
    assert abs(S.partial_corr(a, b, [c])[0]) < 0.3
    s, lo, hi, n = S.theil_sen_ci(x, 2 * x + 1, n=50)
    assert s == pytest.approx(2.0) and lo <= 2.0 <= hi


def test_wilson_and_beta_bracket_p():
    for k, n in ((0, 5), (5, 5), (3, 10)):
        p, lo, hi = S.wilson(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0
        p, lo, hi = S.beta_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0
    assert np.isnan(S.wilson(0, 0)[0])


def test_bootstraps():
    v = np.r_[np.arange(11.0), np.nan]
    med, lo, hi, n = S.bootstrap_median(v, n=200)
    assert med == 5.0 and lo <= 5.0 <= hi and n == 11
    vals = np.array([1.0, 1.0, 1.0, 10.0])                             # galaxy a has 3 sightlines, b one
    keys = np.array(["a", "a", "a", "b"])
    assert S.grouped_bootstrap_median(vals, keys, n=50)[3] == 2
    assert S.group_medians(vals, keys)[0].tolist() == [1.0, 10.0]
    point, lo, hi = S.cluster_bootstrap(keys, lambda i: float(np.mean(vals[i])), n=50)
    assert point == pytest.approx(3.25)


def test_median_band_blanks_sparse_columns():
    A = np.ones((10, 3)); A[:8, 2] = np.nan
    cent, lo, hi, n = S.median_band(A, nmin=5, nboot=20)
    assert cent[:2].tolist() == [1.0, 1.0] and np.isnan(cent[2]) and n.tolist() == [10, 10, 2]
    assert len(S.median_band(np.empty((0, 4)), n_columns=4)[0]) == 4


def test_running_and_binned():
    x = np.arange(100.0); y = 2 * x
    t = S.running_median(x, y, nbins=5, nmin=10, nboot=50)
    assert len(t) == 5 and np.allclose(t["y"], 2 * t["x"], atol=2)
    assert len(S.running_median(x[:5], y[:5], nbins=5, nmin=10)) == 0
    tq = S.running_quantile(x, y, 0.5, nbins=4, nmin=10)
    assert len(tq) == 4 and tq["n"].sum() == 100
    keys = np.repeat(np.arange(50), 2)
    tg = S.running_median(x, y, nbins=5, nmin=5, nboot=20, keys=keys)
    assert "n_gal" in tg and tg["n_gal"].sum() >= 50              # a galaxy whose two rows straddle a bin edge counts in both
    xe = np.r_[np.zeros(10), np.linspace(0.1, 2, 30)]
    b = S.binned_median(xe, xe, [0, 0, 1, 2], nmin=5, nboot=20)
    assert b["n"].tolist()[0] == 10 and b["y"].iloc[0] == 0.0         # the floor pile has its own bin
    bs = S.binned_stats(x, y, np.array([0, 50, 100]), pcts=(25, 75))
    assert bs["n"].tolist() == [50, 50] and bs["x"].tolist() == [25.0, 75.0]
    assert S.binned_stats(x, y, 5, closed="both", filled_only=True)["y"].size == 5


def test_log_band_with_floor_keeps_holes():
    A = np.array([[1.0, 0.0], [10.0, 0.0], [100.0, 1.0]])
    med, lo, hi, frac, floor = S.log_band_with_floor(A)
    assert med[0] == pytest.approx(1.0) and med[1] == floor and frac.tolist() == [1.0, pytest.approx(1 / 3)]


def test_weights_groups_jackknife():
    assert S.weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == pytest.approx(2.0)
    assert S.time_weighted_mean([0, 1, 1, 3], [2, 2, 2, 2]) == pytest.approx(2.0)
    lab, edges = S.terciles(np.arange(9.0))
    assert np.bincount(lab).tolist() == [3, 3, 3]
    lab, edges = S.terciles(np.arange(9.0), groups=np.r_[np.zeros(6), np.ones(3)])
    assert set(edges) == {0.0, 1.0}
    vals = {k: np.full(5, float(k)) for k in range(8)}
    est = lambda keep: float(np.mean(np.concatenate([vals[k] for k in keep])))
    theta, sig = S.jackknife(vals, est)
    assert theta == pytest.approx(3.5) and sig > 0
    rng = np.random.default_rng(2)
    xs = rng.normal(10, 0.5, 400); ys = xs + rng.normal(0, 0.05, 400); zs = rng.random(400)
    res, ridge = S.ridge_residual(xs, ys, zs)
    assert np.nanstd(res) < 0.2 and len(ridge) >= 1


def test_modes():
    rng = np.random.default_rng(3)
    bi = np.r_[rng.normal(-3, 0.5, 300), rng.normal(3, 0.5, 300)]
    dip, gx, d, nm = S.kde_dip(bi)
    assert nm >= 2 and abs(dip) < 1.0
    assert S.n_modes(bi, 0.3)[0] >= 2
    p, h = S.silverman_test(bi, k=1, nboot=30)
    assert p < 0.1
    assert S.scaled_kde(bi, np.linspace(-5, 5, 50)).max() == pytest.approx(1.0)
    X, Y, Z, lev = S.kde_levels(bi, rng.normal(size=bi.size), (-5, 5), (-5, 5), (0.5, 0.9), n=30)
    assert Z.shape == (30, 30) and len(lev) == 2


def test_survival():
    t = np.array([1.0, 2.0, 3.0, 4.0]); ev = np.ones(4, bool)
    ts, S_, se, n = S.kaplan_meier(t, ev)
    np.testing.assert_allclose(S_, [1.0, 0.75, 0.5, 0.25, 0.0])
    med, q25, q75, n = S.km_quantiles(t, ev)
    assert (med, q25, q75, n) == (2.0, 1.0, 3.0, 4)
    rng = np.random.default_rng(4)
    tt = rng.exponential(1, 200); g = rng.random(200) < 0.5
    assert S.log_rank(tt, np.ones(200, bool), g)[1] > 0.01
    assert S.log_rank(np.r_[tt[:100], tt[100:] * 5], np.ones(200, bool), np.r_[np.ones(100, bool), np.zeros(100, bool)])[1] < 1e-3
