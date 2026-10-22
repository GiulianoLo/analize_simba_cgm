"""Tests for the particle helpers lifted from the notebooks (simbanator.particles.*)."""
import h5py
import numpy as np
import pytest

from simbanator.particles import cutouts as CU, frames as FR, profiles as PF, grains as GR, ks as KS
import ks_tracks_lib as kl


def _red(evecs=np.eye(3)):
    gpos = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 5.0], [4.0, 0, 0]])
    return dict(evecs=evecs, rmax_kpc=10.0,
                gas=dict(pos=gpos, m_gas=np.ones(4), m_H2=np.array([1.0, 1.0, 1.0, 0.0]), m_dust=np.ones(4),
                         m_HI=np.ones(4), sfr=np.array([1.0, 0.0, 0.0, 1.0]), temp=np.array([1e3, 5e4, 1e6, 1e3]),
                         member=np.array([True, True, False, True])),
                star=dict(pos=gpos, m_star=np.ones(4), member=np.ones(4, bool)))


def test_ks_tracks_lib_is_the_library():
    from simbanator.particles.ks import measure_ks
    assert kl.measure_ks is measure_ks and kl.STAGES_KS[0] == "sf_peak" and "B08" in kl.RELATIONS


def test_face_on_frame_uses_columns():
    # the principal axes are the COLUMNS: a frame whose first axis is z puts (0, 0, 5) in the face-on plane
    E = np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 0]]).T
    R = FR.reduced_radius(np.array([[0.0, 0, 5]]), E, "cyl")
    assert R[0] == pytest.approx(5.0)
    assert FR.reduced_radius(np.array([[3.0, 4, 0]]), np.eye(3), "3d")[0] == pytest.approx(5.0)


def test_select_gas_regions():
    red = _red()
    R, m, reg = FR.select_gas(red, "m_H2", region="ism")
    assert reg.tolist() == [True, False, False, True]
    assert FR.select_gas(red, "m_nope")[0] is None
    assert FR.select_gas(red, "m_H2", member_only=True)[2].tolist() == [True, True, False, True]


def test_sigma_profiles_and_apertures():
    red = _red()
    edges = np.array([0.0, 1.5, 3.0, 10.0])
    rmid, s = PF.reduced_sigma_profile(red, "H2", edges)                 # face-on R: 1, 2, 0 (z axis), 4
    assert s[0] == pytest.approx(2.0 / (np.pi * 1.5 ** 2))                 # (0,0,5) projects to R = 0
    assert PF.reduced_sigma_scalar(red, "H2", 0.5) == pytest.approx(1.0 / (np.pi * 0.25))
    assert PF.reduced_sigma_scalar(red, "H2", 0.5, geom="3d") == 0.0      # gas exists, none inside 0.5 kpc (3-D)
    assert PF.reduced_aperture_mass(red, "H2", 10.0, geom="3d") == pytest.approx(3.0)
    _, cold = PF.reduced_phase_profile(red, "cold", edges)
    assert cold.sum() > 0 and np.isnan(PF.reduced_phase_profile(dict(gas=dict(pos=np.zeros((1, 3))), evecs=np.eye(3)))[1]).all()
    _, T = PF.reduced_temperature_profile(red, edges)
    assert T[2] == pytest.approx(1e3)                                       # only the R = 4 particle there
    assert PF.member_radii(red, "star", nmin=2)[0.5] > 0


def test_cog_enclosed_and_inside():
    R = np.array([1.0, 3.16, 10.0]); M = np.array([2.0, 5.0, 10.0])
    r = PF.radius_from_curve_of_growth(M, R)
    assert 1.0 < r < 10.0
    assert PF.radius_from_curve_of_growth(np.array([9.0, 9.5, 10.0]), R) < 1.0
    edges = np.array([0.0, 1.0, 2.0]); prof = np.array([1.0, 1.0])               # M(<1) = pi, M(<2) = 4 pi
    assert PF.enclosed_radii(prof, edges, fracs=(0.25, 0.625))[0] == pytest.approx(1.0)
    assert PF.enclosed_radii(prof, edges, fracs=(0.625,))[0] == pytest.approx(1.5)
    assert PF.mean_sigma_inside(np.array([2.0, 4.0]), np.array([0.5, 1.5]), 1.0) == 2.0
    assert len(PF.log_edges(0.1, 100, 20)) == 21 and PF.log_edges(0.1, 100, 20)[0] == 0.0


def test_profile_cache_round_trip(tmp_path):
    red = _red()
    p = tmp_path / "m25n512_snap105_gal000042.h5"; p.write_bytes(b"")
    cache = PF.profile_cache_path(str(tmp_path), 3, 0.5, 10, "cyl", "all")
    edges = PF.log_edges(0.5, 10, 3)
    n = PF.build_profile_cache(cache, [str(p)], lambda path: red, edges)
    assert n == 1 and PF.build_profile_cache(cache, [str(p)], lambda path: red, edges) == 0   # incremental
    got = PF.load_profile_cache(cache, nbins=3)
    assert (105, 42) in got and set(got[(105, 42)]) >= {"H2", "star", "Tprof", "phase_cold"}


def test_annulus_surface_densities(tmp_path):
    p = tmp_path / "r.h5"
    with h5py.File(p, "w") as f:
        f.attrs["evecs"] = np.eye(3)
        f.create_dataset("star/pos", data=[[1.0, 0, 0]]); f.create_dataset("star/m_star", data=[1e6])
        f.create_dataset("gas/pos", data=[[2.0, 0, 0]]); f.create_dataset("gas/m_dust", data=[1e3]); f.create_dataset("gas/m_H2", data=[1e5])
    out = PF.annulus_surface_densities(str(p), [0.5, 1.5, 3.0], apertures_kpc=[1.5])
    assert out["n_star"].tolist() == [1.0, 0.0] and out["sig_h2"][1] > 0 and out["nstar_ap"].tolist() == [1.0]
    assert PF.annulus_surface_densities(str(tmp_path / "none.h5"), [0, 1]) is None


def test_cutout_helpers(tmp_path):
    assert CU.cutout_path("/h", "m25n512", 5, 42) == "/h/snap_005/m25n512_snap005_gal000042.h5"
    assert CU.parse_reduced_name("x_snap105_gal000042.h5") == (105, 42) and CU.parse_reduced_name("junk.h5") is None
    p = tmp_path / "c.h5"
    with h5py.File(p, "w") as f:
        h = f.create_group("Header"); h.attrs.update(Time=0.5, HubbleParam=0.7)
        f.create_dataset("PartType4/Coordinates", data=np.full((30, 3), 10.0))
        f.create_dataset("PartType4/StellarFormationTime", data=np.linspace(0.1, 0.5, 30))
        f.create_dataset("PartType4/Masses", data=np.full(30, 1e-4))
    c = CU.read_cutout(str(p), np.zeros(3), "PartType4", fields=("Masses", "Nope"))
    assert c["pos"][0, 0] == pytest.approx(10 * 0.5 / 0.7) and c["Nope"] is None
    a, m = CU.read_star_formation(str(p))
    assert len(a) == 30 and m[0] == pytest.approx(1e6 / 0.7)
    with h5py.File(p, "r") as f:
        assert CU.header_units(f) == (0.5, 0.7)
    assert CU.reduced_has(str(p), (("PartType4", "Masses"),)) and not CU.reduced_has(str(p))
    plan = CU.write_extraction_plan(str(tmp_path / "plans" / "p.h5"), [105, 105], [1, 2], "cis25", attrs=dict(stages="sft"))
    with h5py.File(plan, "r") as f:
        assert f.attrs["sim_name"] == "cis25" and f["entry_gx"][:].tolist() == [1, 2]


def test_grains():
    m = np.array([1.0, 1.0])
    d, hi, h2 = GR.gas_components(m, None, np.array([0.0, 1.0]), np.array([0.5, 0.5]), nH=np.array([1.0, 0.01]))
    np.testing.assert_allclose(h2, [0.38, 0.0])                    # not multiplied by f_neut; no H2 below the density cut
    np.testing.assert_allclose(hi, [0.0, 0.76])                    # below the cut the hydrogen is all HI
    assert GR.growth_timescale(100.0, 20.0, 0.0134) == pytest.approx(0.0254)
    assert GR.growth_timescale(1000.0, 20.0, 0.0134) == pytest.approx(0.00254)
    assert np.isnan(GR.growth_timescale(0.0, 20.0, 0.0134))
    assert GR.weighted_fraction(np.ones(6), np.ones(6, bool), np.r_[np.ones(3, bool), np.zeros(3, bool)]) == 0.5
    assert np.isnan(GR.number_fraction(np.ones(3, bool), np.ones(3, bool)))
    assert GR.stellar_dust_budget(1.0, 1.0, 0.01) == pytest.approx(1e7)


def test_ks_helpers():
    x = np.array([1.0, 1.0, 1.0, np.nan])
    from simbanator.obs import relation_y
    y = relation_y("B08", x) + np.array([-1.0, 0.0, 1.0, 0.0])
    reg, off = KS.ks_region(x, y, np.zeros(4, bool))
    assert reg.tolist() == ["below", "within", "above", "undef"]
    reg, _ = KS.ks_region(x, y, np.array([False, True, True, False]))
    assert reg.tolist()[:3] == ["below", "undef", "undef"]
    assert KS.sf_law_depletion_time_gyr(1.0) > KS.sf_law_depletion_time_gyr(100.0)
