"""Tests for the history and AGN helpers lifted from the notebooks (simbanator.histories.*, simbanator.agn)."""
import os
import types

import h5py
import numpy as np
import pytest

from simbanator import agn as A
from simbanator.histories import events as E, trees as TR, properties as PR, sfh as SF


# ── events ────────────────────────────────────────────────────────────────────

def _quenching_history(n=60, t_q=8e9, zero_after=False):
    t = np.linspace(1e9, 13.5e9, n)
    mstar = np.full(n, 1e11)
    sfr = mstar * np.where(t < t_q, 1e-9, 1e-13)
    if zero_after:
        sfr[t > t_q + 1e9] = 0.0
    return t, mstar, sfr


def test_quench_event_finds_the_transition():
    t, m, s = _quenching_history()
    ev = E.quench_event(t, m, s)
    assert ev["n_events"] >= 1 and 7e9 < ev["t_qt"] < 9.5e9 and ev["t_sft"] <= ev["t_qt"]
    assert np.isnan(E.quench_event(t[:3], m[:3], s[:3])["t_qt"])           # too few snapshots


def test_ssfr_track_keep_zero_option():
    t, m, s = _quenching_history(zero_after=True)
    tz, sz = E.ssfr_track(t, m, s, keep_zero=True)
    tp, sp = E.ssfr_track(t, m, s, keep_zero=False)
    assert (sz == 0).any() and len(tz) > len(tp) and (sp > 0).all()


def test_quench_records_every_column_gets_a_record():
    t, m, s = _quenching_history()
    P = {"masses.stellar": np.column_stack([m, m]), "sfr": np.column_stack([s, m * 1e-9])}   # second never quenches
    recs = E.quench_records(P, t, np.linspace(6, 0, len(t)), [10, 11], [0, 1])
    assert [r["gid"] for r in recs] == [10, 11] and np.isfinite(recs[0]["t_qt"]) and np.isnan(recs[1]["t_qt"])


def test_regrid_and_stacks():
    t = np.array([0.0, 1.0, 2.0, 3.0]); y = t * 2
    g = E.regrid(t, y, [-1, 0.5, 2.5, 4])
    assert np.isnan(g[0]) and np.isnan(g[3]) and g[1] == 1.0 and g[2] == 5.0
    t_yr = np.linspace(1e9, 10e9, 10)
    Q = np.tile(10 ** np.linspace(0, 9, 10)[:, None], (1, 3))
    S = E.aligned_stack(t_yr, Q, [5e9, np.nan, 2e9], np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(S[0], [3.0, 4.0, 5.0]) and np.isnan(S[1]).all()
    xc, prof = E.binned_event_profile(t_yr, Q, [5e9, 5e9, np.nan], xwin_gyr=2, nbins=4, ref0=True)
    assert len(xc) == 4 and np.nanmin(np.abs(prof[0])) == 0.0 and np.isnan(prof[2]).all()


def test_values_at_rows_and_crossings():
    arr = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(E.values_at_rows(arr, [0, 2, -1], [1, 3, 0]), [1.0, 11.0, np.nan])
    np.testing.assert_array_equal(E.ratio_at_rows(arr, arr, [0, 1], [0, 1]), [np.nan, 1.0])
    assert E.first_downward_crossing([0, 1, 2], [1.0, 0.0, -1.0], 0.5) == pytest.approx(0.5)
    assert np.isnan(E.first_downward_crossing([0, 1], [0.0, 1.0], 0.5))
    ts, ys = E.log_track([3, 1, 2], [100.0, 10.0, -1.0], nmin=2)
    assert ts.tolist() == [1, 3] and ys.tolist() == [1.0, 2.0]


# ── trees / properties / sfh ──────────────────────────────────────────────────

@pytest.fixture
def chain(tmp_path):
    """Three catalogues; galaxy 0 at snap 102 descends from 1 at 101, which descends from 2 at 100."""
    prog = {102: [[1, -1], [0, -1]], 101: [[-1, -1], [2, -1], [-1, -1]], 100: [[-1, -1], [-1, -1], [-1, -1]]}
    for s, p in prog.items():
        with h5py.File(tmp_path / f"c_{s}.hdf5", "w") as f:
            n = len(p)
            f.create_dataset("galaxy_data/dicts/masses.stellar", data=np.arange(1.0, n + 1) * s)
            f.create_dataset("tree_data/progen_galaxy_star", data=np.array(p))
    return types.SimpleNamespace(get_caesar_file=lambda s: str(tmp_path / f"c_{int(s)}.hdf5"), name="toy")


def test_progenitors_descendants(chain, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert TR.main_progenitors(chain, 102).tolist() == [1, 0]
    np.testing.assert_array_equal(TR.main_descendants(chain, 101, [0, 1, 2]), [1, 0, np.nan])


def test_trace_main_branches(chain, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def read(s):
        with h5py.File(chain.get_caesar_file(s), "r") as f:
            return {"mstar": f["galaxy_data/dicts/masses.stellar"][:]}, float(102 - s)
    out = PR.trace_main_branches({102: [0]}, 102, 100, read, lambda s: TR.main_progenitors(chain, s),
                                verbose=False)
    st = out[102]
    assert st["snaps"].tolist() == [102, 101, 100]
    np.testing.assert_allclose(st["data"]["mstar"][:, 0], [102.0, 2 * 101.0, 3 * 100.0])   # 0 -> 1 -> 2


def test_align_by_ids():
    arrs = {"a": np.arange(6.0).reshape(3, 2), "keep": np.arange(5.0)}
    out, n = TR.align_by_ids(arrs, [7, 8, 9], [9, 7, 1], verbose=False)
    assert n == 2 and np.isnan(out["a"][2]).all() and out["a"][0].tolist() == [4.0, 5.0]
    assert out["keep"] is arrs["keep"]


def test_property_conversions():
    assert PR.safe_ratio(np.array([1.0, 1.0]), np.array([2.0, 0.0]))[0] == 0.5
    assert np.isnan(PR.safe_ratio(np.array([1.0]), np.array([0.0]))[0]) and PR.safe_ratio(None, 1) is None
    assert PR.gas_oh12(np.array([0.0134]))[0] == pytest.approx(8.69)
    dtm = PR.dust_to_metal_devis(np.array([1.0]), np.array([0.0]), np.array([8.69]))
    assert dtm[0] == pytest.approx(1.0)
    t = np.array([2.0, 0.0, 1.0]); y = 3 * t
    np.testing.assert_allclose(PR.time_derivative(y, t), [3.0, 3.0, 3.0])


def test_sfh_helpers(tmp_path):
    tf = np.array([0.05, 0.05, 0.3]); m = np.array([1e9, 1e9, 2e9])
    tc, sfr = SF.formed_mass_sfh(tf, m, 1.0, bin_myr=100)
    assert np.isclose((sfr * 1e8).sum(), m.sum())                         # mass conserved
    t = np.linspace(0, 10e9, 50)
    assert SF.cumulative_ssfr(t, np.full(50, 1.0), np.full(50, 1e10), 2e9, 1.0) == pytest.approx(0.1)
    assert np.isnan(SF.cumulative_ssfr(t, np.ones(50), np.ones(50), 9.5e9, 1.0))
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("gal1/0/core", data=np.column_stack([[0.0, 1.0], [2.0, 3.0]])); d.attrs["t_obs_gyr"] = 5.0
        f.attrs["made_by"] = b"part 7c"
    a = SF.load_sfh_archive(str(p))
    assert a["gal1"]["0"]["core"][2] == 5.0 and SF.sfh_archive_attrs(str(p))["made_by"] == "part 7c"
    assert SF.sfh_archive_attrs(str(tmp_path / "none.h5")) == {}
    age, frec, cum = SF.sfh_shape_stats(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 3.0, recent_gyr=1.5)
    assert age == pytest.approx(1500.0) and frec == 0.5 and cum[-1] == 1.0


# ── agn ───────────────────────────────────────────────────────────────────────

def test_jet_weight_conventions():
    mbh = np.array([1e9, 1e9, 1e9, 1e7, np.nan]); fedd = np.array([0.2, 0.002, 0.0, 0.001, 0.01])
    w = A.jet_weight(mbh, fedd)
    np.testing.assert_allclose(w, [0.0, 1.0, 0.0, 0.0, 0.0])
    w_old = A.jet_weight(mbh, fedd, require_accretion=False, missing_bh="nan")
    assert w_old[2] == 1.0 and np.isnan(w_old[4])                          # the x_coup convention
    x, fg, _ = A.xray_coupling(mbh, fedd, np.array([1, 1, 1, 1, 1.0]), np.array([10, 1, 10, 10, 10.0]))
    assert x[1] == 0.0                                                      # gas-rich: gate closed
    x, _, _ = A.xray_coupling(mbh, fedd, np.full(5, 1.0), np.full(5, 10.0), gate=0.6)
    assert x[1] == 1.0


def test_agn_states_partition():
    rng = np.random.default_rng(0)
    shape = (30, 20)
    mstar = 10 ** rng.uniform(9, 11, shape); mgas = mstar * 10 ** rng.uniform(-2, 0, shape)
    mbh = 10 ** rng.uniform(6, 9, shape); fedd = 10 ** rng.uniform(-4, 0, shape); fedd[rng.random(shape) < 0.2] = 0
    S, D, X = A.agn_states(mstar, mgas, mbh, fedd)
    total = sum(S[k] for k in S)
    assert np.allclose(total, 1.0)                                          # exclusive and complete
    assert np.all((S["xray"] == 1) <= (D["jet"] == 1))


def test_event_times_and_doses():
    t = np.linspace(0, 10e9, 101)
    md = np.where(t > 4e9, 1.0, 1e-4)
    assert A.ignition_time(t, md, 8e9) == pytest.approx(4.1e9)
    assert np.isnan(A.ignition_time(t, np.zeros(101), 8e9))
    x = (t >= 6e9).astype(float)
    assert A.threshold_onset(t, x, 0.5) == pytest.approx(6e9)
    assert A.relative_onset_time(t, x, 5e9) == pytest.approx(6e9)
    assert A.onset_category(3.0, 4.0, 5.0) == "lead" and A.onset_category(np.nan, 4, 5, has_data=False) == "no_AGN"
    m, n = A.pre_event_mean(t, x, 7e9, window_yr=1e9)
    assert m == 1.0 and n == 11
    assert A.post_event_mean(t, x, 6e9) == 1.0
    tg = t / 1e9
    r = A.exposure_integrals(tg, np.ones(101), np.ones(101), np.full(101, 0.1), 4.0, 6.0)
    assert r["e_x"] == pytest.approx(6.0) and r["e_x_qt"] == pytest.approx(4.0) and r["t_gaspoor"] == pytest.approx(6.0)
    assert r["f_x"] == pytest.approx(1.0) and r["gate_at_sft"] == 1.0 and r["e_x_pre"] == pytest.approx(1.0)
    assert np.isnan(A.exposure_integrals(tg, np.ones(101), np.ones(101), np.ones(101), np.nan, 6.0)["e_x"])
    bm = A.bh_window_metrics(t, np.ones(101), np.linspace(1e8, 2e8, 101), np.full(101, 0.01), 0.0, 10e9)
    assert bm["dMbh"] == pytest.approx(1e8) and bm["jet_frac"] == 1.0
    assert bm["E_agn"] == pytest.approx(0.1 * 10e9 * A.C2_ERG_PER_MSUN)


def test_eddington_round_trip_and_bh_history(tmp_path):
    md = A.mdot_from_fedd(0.05, 1e8)
    assert A.eddington_ratio(1e8, md) == pytest.approx(0.05)
    assert A.bolometric_luminosity(1.0) == pytest.approx(0.1 * A.C2_ERG_PER_MSUN / 3.15576e7)
    for s, vals in ((100, [1.0, 2.0, 3.0]), (99, [10.0, 20.0])):
        with h5py.File(tmp_path / f"c_{s}.hdf5", "w") as f:
            f.create_dataset("galaxy_data/dicts/bhmass", data=vals)
    sb = types.SimpleNamespace(get_caesar_file=lambda s: str(tmp_path / f"c_{int(s)}.hdf5"))
    pidx = np.array([[2.0, 0.0], [1.0, np.nan]])
    skip = set()
    BH = A.build_bh_history(sb, pidx, [100, 99], {"bh_mass": ["bh_mass", "bhmass"]}, out_path=str(tmp_path / "bh.h5"),
                            skip_snaps=skip, verbose=False)
    np.testing.assert_array_equal(BH["bh_mass"], [[3.0, 1.0], [20.0, np.nan]])
    assert A.read_bh_history(str(tmp_path / "bh.h5"))["bh_mass"].shape == (2, 2)
    BH = A.build_bh_history(sb, pidx, [100, 98], {"bh_mass": ["bhmass"]}, skip_snaps=skip, verbose=False)
    assert 98 in skip and np.isnan(BH["bh_mass"][1]).all()
