"""Tests for the RT, photometry, comparative-data and selection helpers lifted from the notebooks."""
import os
import sys
import types

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from simbanator.rt import outputs as RO
from simbanator.photometry import attenuation as PA
import simbanator.obs as OB
import simbanator.selection as SE


@pytest.fixture
def fake_hyperion(monkeypatch):
    class _SED:
        def __init__(self, w, v): self.wav, self.val = w, v
    class ModelOutput:
        def __init__(self, path): self.path = path
        def get_sed(self, inclination="all", aperture=-1):
            w = np.array([10.0, 1.0, 0.1])                          # unsorted on purpose
            return _SED(w, np.vstack([[3.0, 2.0, 1.0], [5.0, 4.0, 3.0]]))
    hm = types.ModuleType("hyperion.model"); hm.ModelOutput = ModelOutput
    monkeypatch.setitem(sys.modules, "hyperion", types.ModuleType("hyperion"))
    monkeypatch.setitem(sys.modules, "hyperion.model", hm)


def test_paths_and_completeness(tmp_path):
    p = RO.tree_rtout_path(str(tmp_path), "dust_on", 91, 12)
    assert p.endswith("dust_on/powderday_sed_out/snap_091/gal_12/snap091.galaxy000012.rtout.sed")
    os.makedirs(os.path.dirname(p))
    with h5py.File(p, "w") as f:
        f.create_group("Peeled")                                    # started, not finished
    assert not RO.sed_is_complete(p)
    with h5py.File(p, "a") as f:
        f.create_group("Peeled/group_00001/seds")
    assert RO.sed_is_complete(p) and not RO.sed_is_complete(None) and not RO.sed_is_complete(str(tmp_path / "x"))
    rows = RO.rt_census(str(tmp_path), "dust_on", [91])
    assert rows[0]["gid"] == 12 and rows[0]["complete"] and rows[0]["phase"].startswith("no LOG")


def test_log_phase_and_masters(tmp_path):
    g = tmp_path / "gal_1"; g.mkdir()
    (g / "a.LOG").write_text("x\nstarting final iteration\n")
    assert RO.log_phase(str(g))[0] == "starting final iteration"
    (g / "a.LOG").write_text("Total CPU time elapsed: 10")
    assert RO.log_phase(str(g))[0] == "hyperion finished"
    m = tmp_path / "parameters_master.py"
    m.write_text('BH_SED = True\nnenkova_params = [5, 30, 60, 1.5]\nTHETA = [0]\n')
    v, _ = RO.parse_master(str(m))
    assert v["BH_SED"] == "True" and v["incl"] == 60 and v["n_theta"] == 1 and v["BH_var"] is None
    jd = tmp_path / "out" / "arm" / "powderday_sed_out" / "snap_091"; jd.mkdir(parents=True)
    (jd / "parameters_master.py").write_text("BH_SED = False\n"); (jd / "ids.txt").write_text("1\n2\n")
    st = RO.staged_master_values(str(tmp_path / "out"), "arm", [91, 96])
    assert st[91]["values"]["BH_SED"] == "False" and st[91]["n_ids"] == 2 and st[96]["staged"] is False
    ok, bad, good = RO.check_bh_field_names('add_field(("bh","Mass"))', "BAD", ["Mass"])
    assert ok and bad == 0


def test_read_rt_sed_and_stacks(tmp_path, fake_hyperion):
    p = tmp_path / "x.rtout.sed"; p.write_bytes(b"")
    w, v = RO.read_rt_sed(str(p))
    assert w.tolist() == [0.1, 1.0, 10.0] and v.tolist() == [2.0, 3.0, 4.0]       # sorted, mean of 2 sightlines
    assert RO.read_rt_sed(str(p), sightline=0)[1].tolist() == [1.0, 2.0, 3.0]
    assert RO.read_rt_sed(str(p), sightline="all")[1].shape == (2, 3)
    assert RO.read_rt_sed(str(tmp_path / "none")) is None
    st = RO.stack_seds([(w, v), (w, 2 * v)], np.array([0.1, 1.0, 10.0]))
    assert st["n"] == 2 and np.allclose(st["med"], 1.5 * v)
    assert np.isnan(RO.on_log_grid(w, v, np.array([100.0]))[0])


def test_flux_conversions_round_trip():
    wav = np.array([0.5, 1.0, 2.0]); L = np.array([1e43, 2e43, 3e43])
    w_obs, fnu = RO.sed_to_flux_density(wav, L, 1.0)
    w_rest, L2 = PA.fnu_to_nulnu(w_obs, fnu, 1.0)
    np.testing.assert_allclose(w_rest, wav); np.testing.assert_allclose(L2, L)
    w_f, fac = PA.nulnu_to_fnu_factor(wav, 1.0)
    np.testing.assert_allclose(L * fac, fnu)


def test_attenuation_helpers():
    np.testing.assert_allclose(PA.av_from_pair([1.0, 0.5], [1.0, 1.0]), [0.0, -2.5 * np.log10(0.5)])
    assert PA.av_from_pair([2.0], [1.5], magnitudes=True)[0] == pytest.approx(0.5)
    w = np.linspace(0.1, 2.0, 50)
    assert PA.av_in_window((w, np.full(50, 0.5)), (w, np.ones(50))) == pytest.approx(-2.5 * np.log10(0.5))
    a = PA.attenuation_curve((w, np.full(50, 0.5)), (w, np.ones(50)), np.array([0.55]))
    assert a[0] == pytest.approx(0.7526, abs=1e-3)
    # a flat nu L_nu has L_bol = L ln(nu_max / nu_min)
    wl = np.geomspace(1.0, 10.0, 2000)
    assert PA.bolometric_from_nulnu(wl, np.ones_like(wl)) == pytest.approx(np.log(10), rel=1e-4)
    f850 = PA.mbb_flux_mjy(1e8, 2.0, np.array([850.0]), 30.0)
    assert 0 < f850[0] < 100


def test_anchor_av(tmp_path):
    p = tmp_path / "ann.fits"
    Table(dict(snap=[100, 100, 100], gal_id=[1, 1, 2], aperture=["ap10kpc"] * 3, incl=["0", "90", "0"],
               A_V=[0.2, 0.4, 1.0])).write(p)
    av, how = PA.anchor_av("ap10kpc", annulus_av_path=str(p))
    assert av[(100, 1)] == pytest.approx(0.3) and "2 sightlines" in how
    with pytest.raises(KeyError):
        PA.anchor_av("nope", annulus_av_path=str(p))


def test_relations_and_matching():
    assert set(OB.RELATIONS) == {"K98", "B08", "RK19"} and "VERTICO" in OB.RELATIONS_EXTRA
    # VERTICO passes through log t_dep = 1.86 Gyr at Sigma_mol = 10 Msun/pc^2
    assert OB.relation_y("VERTICO", 1.0) == pytest.approx(np.log10(10 * 1e6 / 1.86e9))
    assert OB.relation_y("B08", 1.0, shift_dex=OB.imf_shift_dex("kroupa")) < OB.relation_y("B08", 1.0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    pts, mu, Ci = OB.mahalanobis_cloud(X)
    d2 = OB.mahalanobis_d2(np.array([[0.0, 0.0], [3.0, 0.0]]), mu, Ci)
    assert d2[0] < 0.1 and 7 < d2[1] < 11


def test_overlays_draw():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    items = OB.draw_ks_relations(ax, (-0.5, 3.6))
    assert len(items) == 2
    t = pd.DataFrame(dict(logSigmaH2=[1.0, 2.0], logSigmaSFR=[-2.0, -1.0], sigma_is_ll=[False, True],
                          logSigmaSFR_elo=[np.nan, 0.1], logSigmaSFR_ehi=[0.2, 0.1], logSigmaH2_err=[0.1, np.nan], id=["a", "b"]))
    assert len(OB.overlay_ks_observations(ax, t, (-4.6, 1.1), annotate=True)) == 2
    OB.draw_observed_points(ax, pd.DataFrame(dict(x=[1.0], y=[2.0], size=[40], marker=["o"], edge=["k"], yc=["ul"])),
                            "x", "y", ycens="yc")
    plt.close(fig)


def test_readers(tmp_path):
    bm = tmp_path / "bm.fits"
    Table(dict(wavelength=[2000.0, 1000.0], Fnu=[1.0, 2.0], L_lambda_total=[10.0, 10.0],
               **{"agn.SKIRTOR2016_torus": [1.0, 5.0]})).write(bm)
    wl, f, fg = OB.read_cigale_best_model(str(bm))
    assert wl.tolist() == [1.0, 2.0] and fg.tolist() == [1.0, 0.9]
    ob = tmp_path / "obs.fits"
    Table({"id": ["719", "2962"], "b1": [1.0, -1.0], "b1_err": [0.1, 0.1]}).write(ob)
    assert OB.read_observed_photometry(str(ob), "719", {"b1": 0.5, "b9": 2.0})[1].tolist() == [1.0]
    assert len(OB.read_observed_photometry(str(ob), "2962", {"b1": 0.5})[0]) == 0


def test_selection():
    ms = np.array([1e11, 1e11, 1e9, 1e11]); sfr = np.array([0.1, 50.0, 0.0, 0.1])
    m, cuts = SE.quenched_sample(ms, sfr, 1e10, ngas=[30, 30, 30, 5], nstar=[30, 30, 30, 30],
                                 mdust=[1e6, 1e6, 1e6, 1e6], mh2=[1e9, 1e9, 1e9, 1e9])
    assert m.tolist() == [True, False, False, False] and set(cuts) == {"massive", "passive", "ngas", "nstar", "dust/H2"}
    m2, _ = SE.quenched_sample(ms, sfr, 1e10)                     # missing inputs skip their cut
    assert m2.tolist() == [True, False, False, True]
    dusty, poor = SE.dust_label([1e6, 1e4, np.nan], [1e10, 1e10, 1e10])
    assert dusty.tolist() == [True, False, False] and poor.tolist() == [False, True, False]
    assert SE.mass_bins([9.9, 10.2, 11.5, np.nan], [10, 11, 12]).tolist() == [-1, 0, 1, -1]
    ctl = SE.sf_controls([10.0, 10.5, 11.0, 10.9], [True, True, True, True], quenched_ids=[2], quenched_logm=[11.0])
    assert ctl == [(3, 2, pytest.approx(-0.1))]
