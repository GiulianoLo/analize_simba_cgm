"""Local harness for ks_tracks_lib (no cluster data, no simbanator import).

Run with:  python -m pytest tests/test_ks_tracks.py -v
Synthetic reduced particle files are written in the exact schema of build_reduced_particles_job.py
(attrs + gas/{idx,pos,m_gas,m_dust,m_HI,m_H2,sfr,temp,member} + star/{idx,pos,m_star,member,tform}).
"""
import importlib.util
import os
import sys
import types

import h5py
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ks_tracks_lib as kl  # noqa: E402

RNG = np.random.default_rng(12345)
T_OBS = 8.0          # Gyr, "snapshot" cosmic time of the synthetic files


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────
def _fake_a_to_t():
    """Linear a -> t map (Gyr): t = 13 * a (only monotonicity matters for the tests)."""
    def a_to_t(a):
        return 13.0 * np.asarray(a, float)
    return a_to_t


def _random_rotation():
    q, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _exp_disc(n, rd, hz=0.05):
    """Face-on exponential disc in the frame's (x, y) plane, z thin."""
    r = RNG.gamma(2.0, rd, n)                    # Sigma ~ exp(-r/rd) -> p(r) ~ r exp(-r/rd)
    phi = RNG.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), RNG.normal(0, hz, n)])


def write_reduced(path, evecs, gas_pos_frame, star_pos_frame, m_h2, sfr_gas, m_star, tform,
                  gas_member=None, star_member=None, with_tform=True, a=0.75,
                  gas_vel_frame=None, star_vel_frame=None, with_vel=True):
    """Write a synthetic reduced file. Positions are given in the PRINCIPAL frame and rotated back
    to the 'original' frame with evecs (columns = axes), exactly as the job stores them:
    stored pos satisfies pos @ evecs == frame coordinates. Velocities (frame, km/s) are rotated the
    same way; with_vel=False writes a pre-2026-08-28 file (no `vel`)."""
    gpos = np.asarray(gas_pos_frame, float) @ evecs.T
    spos = np.asarray(star_pos_frame, float) @ evecs.T
    ng, ns = len(gpos), len(spos)
    gvel = (np.asarray(gas_vel_frame, float) if gas_vel_frame is not None else np.zeros((ng, 3))) @ evecs.T
    svel = (np.asarray(star_vel_frame, float) if star_vel_frame is not None else np.zeros((ns, 3))) @ evecs.T
    gas_member = np.ones(ng, bool) if gas_member is None else np.asarray(gas_member, bool)
    star_member = np.ones(ns, bool) if star_member is None else np.asarray(star_member, bool)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as o:
        o.attrs["sim_name"] = "m25n512"
        o.attrs["snap"] = 134
        o.attrs["gx"] = 7
        o.attrs["redshift"] = 1.0 / a - 1.0
        o.attrs["a"] = a
        o.attrs["hub"] = 0.68
        o.attrs["rmax_kpc"] = 100.0
        o.attrs["h2_recipe"] = "caesar-v1"                # current HI/H2 split (build_profiles_job.H2_RECIPE)
        o.attrs["center_kpc"] = np.zeros(3)
        o.attrs["evecs"] = evecs
        g = o.create_group("gas")
        g.create_dataset("idx", data=np.arange(ng, dtype=np.int64), compression="lzf")
        g.create_dataset("pos", data=gpos.astype(np.float32), compression="lzf")
        m_h2 = np.asarray(m_h2, np.float32)
        g.create_dataset("m_gas", data=(m_h2 * 3 + 1e5).astype(np.float32), compression="lzf")
        g.create_dataset("m_dust", data=(m_h2 * 0.01).astype(np.float32), compression="lzf")
        g.create_dataset("m_HI", data=(m_h2 * 2).astype(np.float32), compression="lzf")
        g.create_dataset("m_H2", data=m_h2, compression="lzf")
        g.create_dataset("sfr", data=np.asarray(sfr_gas, np.float32), compression="lzf")
        g.create_dataset("temp", data=np.full(ng, 5e3, np.float32), compression="lzf")
        g.create_dataset("member", data=gas_member, compression="lzf")
        if with_vel:
            g.create_dataset("vel", data=gvel.astype(np.float32), compression="lzf")
        s = o.create_group("star")
        s.create_dataset("idx", data=np.arange(ns, dtype=np.int64), compression="lzf")
        s.create_dataset("pos", data=spos.astype(np.float32), compression="lzf")
        s.create_dataset("m_star", data=np.asarray(m_star, np.float32), compression="lzf")
        s.create_dataset("member", data=star_member, compression="lzf")
        if with_tform:
            s.create_dataset("tform", data=np.asarray(tform, np.float32), compression="lzf")
        if with_vel:
            s.create_dataset("vel", data=svel.astype(np.float32), compression="lzf")
    return path


def _disc_velocities(pos_frame, v_circ, sigma=0.0):
    """Circular rotation about the frame's z axis (v_phi = v_circ) + an isotropic Gaussian dispersion."""
    pos_frame = np.asarray(pos_frame, float)
    R = np.hypot(pos_frame[:, 0], pos_frame[:, 1])
    R = np.where(R > 0, R, 1e-6)
    v = np.column_stack([-v_circ * pos_frame[:, 1] / R, v_circ * pos_frame[:, 0] / R, np.zeros(len(R))])
    if sigma > 0:
        v = v + RNG.normal(0.0, sigma, v.shape)
    return v


@pytest.fixture
def disc_file(tmp_path):
    """A gas-rich disc: 4000 H2 particles (rd = 2 kpc), 6000 stars (rd = 3 kpc) of which a known
    subset formed in the last 100 / 25 Myr; 300 extra NON-member gas particles far out."""
    ev = _random_rotation()
    ng, ns = 4000, 6000
    gpos = _exp_disc(ng, 2.0)
    spos = _exp_disc(ns, 3.0)
    m_h2 = np.full(ng, 2.0e5)
    sfr_gas = np.full(ng, 1e-3)
    m_star = np.full(ns, 1.6e6)
    a_to_t = _fake_a_to_t()
    # formation epochs: old, except 200 stars in (T_OBS-0.1, T_OBS] and 50 of those in the last 25 Myr
    t_form = RNG.uniform(1.0, T_OBS - 0.5, ns)
    t_form[:150] = RNG.uniform(T_OBS - 0.099, T_OBS - 0.026, 150)
    t_form[150:200] = RNG.uniform(T_OBS - 0.024, T_OBS, 50)
    tform = t_form / 13.0                                          # inverse of _fake_a_to_t
    # non-member gas ring at 40 kpc
    nn = 300
    phi = RNG.uniform(0, 2 * np.pi, nn)
    ring = np.column_stack([40 * np.cos(phi), 40 * np.sin(phi), np.zeros(nn)])
    gpos_all = np.vstack([gpos, ring])
    m_h2_all = np.concatenate([m_h2, np.full(nn, 5e5)])
    sfr_all = np.concatenate([sfr_gas, np.zeros(nn)])
    gmem = np.concatenate([np.ones(ng, bool), np.zeros(nn, bool)])
    p = write_reduced(str(tmp_path / "snap_134" / "m25n512_snap134_gal000007.h5"), ev,
                      gpos_all, spos, m_h2_all, sfr_all, m_star, tform, gas_member=gmem)
    return dict(path=p, evecs=ev, gpos=gpos, spos=spos, m_h2=m_h2, m_star=m_star,
                a_to_t=a_to_t, n_young100=200, n_young25=50, sfr_gas_tot=ng * 1e-3)


# ── geometry ──────────────────────────────────────────────────────────────────────────────────────
def test_face_on_R_uses_columns_as_axes(disc_file):
    red = kl.load_reduced(disc_file["path"])
    R = kl.face_on_R(red["gas"]["pos"][:4000], red["evecs"])
    R_true = np.hypot(disc_file["gpos"][:, 0], disc_file["gpos"][:, 1])
    assert np.allclose(R, R_true, atol=2e-3)                      # float32 storage
    # the wrong convention (pos @ evecs.T) does NOT recover the in-plane radius
    proj_wrong = red["gas"]["pos"][:4000] @ red["evecs"].T
    R_wrong = np.hypot(proj_wrong[:, 0], proj_wrong[:, 1])
    assert not np.allclose(R_wrong, R_true, atol=0.1)


def test_half_mass_radius_uniform_disc():
    n = 200000
    R = 5.0 * np.sqrt(RNG.uniform(0, 1, n))                       # uniform disc of radius 5
    r50 = kl.half_mass_radius(R, np.ones(n))
    assert abs(r50 - 5.0 / np.sqrt(2)) < 0.03
    assert np.isnan(kl.half_mass_radius(R, np.zeros(n)))
    assert np.isnan(kl.half_mass_radius(R[:3], np.ones(3), n_min=5))
    assert kl.half_mass_radius([2.0], [1.0]) == 2.0


# ── SFR windows ───────────────────────────────────────────────────────────────────────────────────
def test_sfr_windows_exact(disc_file):
    red = kl.load_reduced(disc_file["path"])
    rows = kl.measure_ks(red, T_OBS, disc_file["a_to_t"], ngas_min=10)
    r10 = [r for r in rows if r["ap_label"] == "ap10kpc"][0]
    assert r10["has_tform"]
    # all stars inside 10 kpc? not necessarily -> compare with the direct count inside the aperture
    Rs = np.hypot(disc_file["spos"][:, 0], disc_file["spos"][:, 1])
    with h5py.File(disc_file["path"], "r") as f:
        tform = f["star/tform"][:]
    tform_gyr = disc_file["a_to_t"](tform)
    ins = Rs <= 10.0
    exp100 = disc_file["m_star"][ins & (tform_gyr >= T_OBS - 0.1)].sum() / 1e8
    exp25 = disc_file["m_star"][ins & (tform_gyr >= T_OBS - 0.025)].sum() / 25e6
    assert np.isclose(r10["sfr100"], exp100, rtol=1e-6)
    assert np.isclose(r10["sfr25"], exp25, rtol=1e-6)
    assert r10["n_young100"] == int((ins & (tform_gyr >= T_OBS - 0.1)).sum())
    assert np.isclose(r10["sfr100_tot"], disc_file["m_star"][tform_gyr >= T_OBS - 0.1].sum() / 1e8)
    assert np.isclose(r10["sfr_inst"], 1e-3 * int((np.hypot(disc_file["gpos"][:, 0],
                                                             disc_file["gpos"][:, 1]) <= 10).sum()),
                      rtol=1e-3)


def test_no_tform_flags_nan(tmp_path, disc_file):
    ev = disc_file["evecs"]
    p = write_reduced(str(tmp_path / "x" / "f.h5"), ev, disc_file["gpos"], disc_file["spos"],
                      disc_file["m_h2"], np.full(4000, 1e-3), disc_file["m_star"],
                      np.zeros(6000), with_tform=False)
    rows = kl.measure_ks(kl.load_reduced(p), T_OBS, disc_file["a_to_t"])
    r = rows[0]
    assert not r["has_tform"]
    assert np.isnan(r["sfr100"]) and np.isnan(r["sfr25"]) and np.isnan(r["sfr100_tot"])
    assert r["r50_sfr_src"] == "gas" and np.isfinite(r["r50_sfr"])
    assert np.isfinite(r["m_H2"])                                    # gas side unaffected


# ── apertures / conventions ───────────────────────────────────────────────────────────────────────
def test_r50_h2_row_is_half_total_over_area(disc_file):
    red = kl.load_reduced(disc_file["path"])
    rows = kl.measure_ks(red, T_OBS, disc_file["a_to_t"])
    r = [x for x in rows if x["ap_label"] == "R50_H2"][0]
    assert np.isfinite(r["ap_kpc"])
    # member-only total (the 300 non-member ring particles are excluded)
    assert np.isclose(r["m_H2_tot"], disc_file["m_h2"].sum(), rtol=1e-6)
    # M(<R50)/area == 0.5 M_tot/area to within one particle (interpolated crossing)
    assert abs(r["m_H2"] - 0.5 * r["m_H2_tot"]) <= 2.0e5 + 1.0
    # exponential disc: R50 ~ 1.678 rd
    assert abs(r["ap_kpc"] - 1.678 * 2.0) < 0.15
    cols = kl.ks_columns({k: np.array([x[k] for x in rows]) for k in kl.MEASURE_COLUMNS})
    i = [x["ap_label"] for x in rows].index("R50_H2")
    j = [x["ap_label"] for x in rows].index("ap3kpc")
    exp = np.log10(1.36 * 0.5 * r["m_H2_tot"] / (np.pi * r["ap_kpc"] ** 2) / 1e6)
    assert abs(cols["logSigmaH2"][i] - exp) < 0.01
    # the observed-convention SFR column on the R50_H2 row is 0.5 * SFR_tot / area
    exp_sfr = np.log10(0.5 * r["sfr100_tot"] / (np.pi * r["ap_kpc"] ** 2))
    assert np.isclose(cols["logSigmaSFR_obs"][i], exp_sfr)
    assert np.isclose(cols["logSigmaSFR"][i], exp_sfr)
    # fixed aperture: literal M(<3.162)/area, no 0.5, and obs == inside
    r3 = rows[j]
    assert np.isclose(cols["logSigmaH2"][j], np.log10(1.36 * r3["m_H2"] / r3["area_kpc2"] / 1e6))
    assert np.isclose(cols["logSigmaSFR_obs"][j], cols["logSigmaSFR_inside"][j])
    assert not cols["is_ul"][i] and not cols["is_ul"][j]
    # t_dep consistent with the Sigma columns
    assert np.isclose(cols["tdep_gyr"][i], 10 ** (cols["logSigmaH2"][i] + 6 - cols["logSigmaSFR"][i]) / 1e9)


def test_member_only_excludes_ring(disc_file):
    red = kl.load_reduced(disc_file["path"])
    rows_m = kl.measure_ks(red, T_OBS, disc_file["a_to_t"], member_only=True)
    rows_a = kl.measure_ks(red, T_OBS, disc_file["a_to_t"], member_only=False)
    assert rows_m[0]["n_gas_tot"] == 4000 and rows_a[0]["n_gas_tot"] == 4300
    assert rows_a[0]["m_H2_tot"] > rows_m[0]["m_H2_tot"]
    # the ring sits at 40 kpc: fixed 10 kpc apertures agree, R50_H2 moves outward without member cut
    a10 = [x for x in rows_a if x["ap_label"] == "ap10kpc"][0]
    m10 = [x for x in rows_m if x["ap_label"] == "ap10kpc"][0]
    assert np.isclose(a10["m_H2"], m10["m_H2"])
    assert rows_a[0]["r50_H2"] > rows_m[0]["r50_H2"]


def test_ngas_gate_keeps_counts(disc_file):
    red = kl.load_reduced(disc_file["path"])
    rows = kl.measure_ks(red, T_OBS, disc_file["a_to_t"], ngas_min=10 ** 6)
    r = [x for x in rows if x["ap_label"] == "ap3kpc"][0]
    assert r["n_gas"] > 0 and np.isnan(r["m_H2"]) and np.isnan(r["sfr_inst"])
    assert np.isfinite(r["sfr100"])                                  # stellar side not gated


def test_h2_free_and_sfr_zero(tmp_path, disc_file):
    """Quenched-galaxy corner cases: no H2 at all, no young stars -> NaN R50_H2 row, censored SFR."""
    ev = disc_file["evecs"]
    ns = 500
    spos = _exp_disc(ns, 1.5)
    gpos = _exp_disc(40, 3.0)
    tform = np.full(ns, 2.0 / 13.0)                                 # all formed at t = 2 Gyr
    p = write_reduced(str(tmp_path / "q" / "f.h5"), ev, gpos, spos, np.zeros(40), np.zeros(40),
                      np.full(ns, 1.6e6), tform)
    rows = kl.measure_ks(kl.load_reduced(p), T_OBS, disc_file["a_to_t"], ngas_min=10)
    labs = [x["ap_label"] for x in rows]
    assert labs == list(kl.AP_LABELS)
    rh2 = rows[labs.index("R50_H2")]
    assert np.isnan(rh2["ap_kpc"]) and np.isnan(rh2["area_kpc2"]) and rh2["n_gas"] == 0
    r3 = rows[labs.index("ap3kpc")]
    assert r3["sfr100"] == 0.0 and r3["n_young100"] == 0
    tab = {k: np.array([x[k] for x in rows]) for k in kl.MEASURE_COLUMNS}
    cols = kl.ks_columns(tab)
    j = labs.index("ap3kpc")
    assert cols["is_ul"][j]
    assert np.isclose(cols["logSigmaSFR_ul"][j], np.log10(1.6e6 / 1e8 / (np.pi * 3.162 ** 2)))
    assert np.isclose(cols["logSigmaSFR"][j], cols["logSigmaSFR_ul"][j])
    assert np.isnan(cols["tdep_gyr"][j])
    assert np.isnan(cols["logSigmaH2"][j])                          # m_H2 = 0 -> NaN, not -inf
    # instantaneous-SFR variant with a suffix
    inst = kl.ks_columns(tab, sfr_key="sfr_inst", sfr_tot_key="sfr_inst_tot", suffix="_inst")
    assert "logSigmaSFR_inst" in inst and "logSigmaH2" in inst


def test_load_reduced_missing_and_corrupt(tmp_path):
    assert kl.load_reduced(str(tmp_path / "nope.h5")) is None
    p = tmp_path / "bad.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    bad = []
    assert kl.load_reduced(str(p), bad=bad) is None and bad == [str(p)]
    assert kl.reduced_path("/r", "m25n512", 105, 42) == "/r/snap_105/m25n512_snap105_gal000042.h5"


# ── critical epochs with the REAL quenching finder (loaded standalone: numpy/scipy only) ─────────
def _load_quenching():
    spec = importlib.util.spec_from_file_location(
        "simbanator_quenching_standalone", os.path.join(ROOT, "simbanator", "analysis", "quenching.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.find_quenching_times


def test_build_stage_records_synthetic_history():
    fq = _load_quenching()
    # history rows: row 0 = anchor (latest), descending time like the cluster HDF5s
    t_gyr = np.linspace(1.0, 9.0, 81)[::-1]
    t_yr = t_gyr * 1e9
    z = 1.0 / (t_gyr / 9.0) ** (2 / 3) - 1.0
    # star-forming until 5 Gyr, exponential decline (tau 0.4 Gyr) afterwards, floor 1e-14
    ssfr = np.where(t_gyr < 5.0, 3.0 / t_yr, np.maximum(3.0 / t_yr * np.exp(-(t_gyr - 5.0) / 0.4), 1e-14))
    mstar = 1e10 * (1 - np.exp(-t_gyr / 3.0))
    gas = 5e9 * np.exp(-((t_gyr - 4.0) / 2.5) ** 2) + 1e7 * (t_gyr / 9.0)
    gas[t_gyr > 7.0] = 2e6                                            # trough late, then flat
    P = {"masses.stellar": mstar[:, None], "sfr": (ssfr * mstar)[:, None], "masses.gas": gas[:, None]}
    recs = kl.build_stage_records(P, t_yr, z, np.array([17]), [0], fq, age_of_z_gyr=lambda zz: 9.0)
    assert len(recs) == 1
    r = recs[0]
    assert r["gid"] == 17 and r["n_events"] >= 1
    assert 5.0e9 < r["t_sft"] < r["t_qt"] < r["t_post_quench"]
    assert r["tau_q"] > 0 and np.isfinite(r["tau_q_over_tH"])
    assert r["row_anchor"] == 0 and np.isclose(r["t_anchor"], t_yr[0])
    assert 0 <= r["row_qt"] < len(t_yr) and abs(t_yr[r["row_qt"]] - r["t_qt"]) <= 0.05e9 + 1
    assert r["t_sf_peak"] < r["t_sft"]                                 # sSFR peak precedes quenching
    # gas_min = trough after QT (the early formation-epoch minimum at t=1 Gyr must NOT be picked)
    assert r["t_gas_min"] >= r["t_qt"]
    assert r["row_gas_min"] >= 0 and t_yr[r["row_gas_min"]] >= r["t_qt"] - 0.06e9
    # a stage beyond the anchor keeps its time but has no row (here: shrink the history to end
    # before QT + persistence)
    short = t_gyr <= 6.6
    P3 = {k: v[short] for k, v in P.items()}
    r3 = kl.build_stage_records(P3, t_yr[short], z[short], np.array([17]), [0], fq)[0]
    assert np.isfinite(r3["t_post_quench"]) and r3["t_post_quench"] > t_yr[short][0]
    assert r3["row_post_quench"] == -1 and "post_quench" not in kl.stage_time_order(r3)
    assert kl.stage_time_order(r3)[-1] == "anchor"
    order = kl.stage_time_order(r)
    assert order.index("sf_peak") < order.index("sft") < order.index("qt") <= order.index("post_quench")
    assert order[-1] == "anchor"
    # a galaxy that never quenches still gets a record with NaN event stages
    ssfr2 = 3.0 / t_yr
    P2 = {"masses.stellar": mstar[:, None], "sfr": (ssfr2 * mstar)[:, None], "masses.gas": gas[:, None]}
    r2 = kl.build_stage_records(P2, t_yr, z, np.array([3]), [0], fq)[0]
    assert r2["n_events"] == 0 and np.isnan(r2["t_qt"]) and r2["row_qt"] == -1
    assert np.isfinite(r2["t_sf_peak"]) and r2["row_anchor"] == 0


def test_nearest_row_and_attach_bh_stages():
    t_yr = np.linspace(1.0, 9.0, 81)[::-1] * 1e9           # row 0 = anchor (9 Gyr), descending like the histories
    assert kl.nearest_row(t_yr, np.nan) == -1
    assert kl.nearest_row(t_yr, 9.0e9) == 0 and kl.nearest_row(t_yr, 1.0e9) == 80
    assert kl.nearest_row(t_yr, 5.04e9) == int(np.argmin(np.abs(t_yr - 5.04e9)))
    assert kl.nearest_row(t_yr, 9.5e9, t_anchor=t_yr[0]) == -1          # beyond the anchor -> no row
    assert kl.nearest_row(t_yr, 9.0e9 + 0.5, t_anchor=t_yr[0]) == 0     # inside the 1 yr tolerance
    recs = [dict(gid=17, t_sft=5.0e9), dict(gid=3, t_sft=np.nan), dict(gid=8, t_sft=8.9e9), dict(gid=5, t_sft=6.0e9)]
    dt = {17: {"agn_ign": -0.5, "jet_on": 0.2}, 8: {"agn_ign": -0.3, "jet_on": 0.4}, 3: {"agn_ign": -1.0, "jet_on": 0.0},
          5: {"agn_ign": np.nan}}
    out = kl.attach_bh_stages(recs, t_yr, dt)
    assert out is recs
    r = recs[0]
    assert np.isclose(r["t_agn_ign"], 4.5e9) and np.isclose(r["t_jet_on"], 5.2e9)
    assert r["row_agn_ign"] == kl.nearest_row(t_yr, 4.5e9) and r["row_jet_on"] == kl.nearest_row(t_yr, 5.2e9)
    assert t_yr[r["row_agn_ign"]] < r["t_sft"] < t_yr[r["row_jet_on"]]
    assert np.isnan(recs[1]["t_agn_ign"]) and recs[1]["row_agn_ign"] == -1       # no SFT -> undefined
    assert np.isclose(recs[2]["t_agn_ign"], 8.6e9) and recs[2]["row_agn_ign"] >= 0
    assert np.isclose(recs[2]["t_jet_on"], 9.3e9) and recs[2]["row_jet_on"] == -1  # jet onset beyond the anchor: time kept, no row
    assert np.isnan(recs[3]["t_agn_ign"]) and recs[3]["row_agn_ign"] == -1 and np.isnan(recs[3]["t_jet_on"]) and recs[3]["row_jet_on"] == -1
    # the records of build_stage_records accept the extra stages: every STAGES_KS entry has t_/row_ keys
    for st in kl.STAGES_BH:
        assert st in kl.STAGES_KS and all(f"t_{st}" in x and f"row_{st}" in x for x in recs)


def test_end_stage_anchor_or_last_h2_snapshot():
    fq = _load_quenching()
    t_gyr = np.linspace(1.0, 9.0, 81)[::-1]          # row 0 = anchor
    t_yr = t_gyr * 1e9
    z = 1.0 / (t_gyr / 9.0) ** (2 / 3) - 1.0
    ssfr = np.where(t_gyr < 5.0, 3.0 / t_yr, np.maximum(3.0 / t_yr * np.exp(-(t_gyr - 5.0) / 0.4), 1e-14))
    mstar = np.full_like(t_gyr, 1e10)
    gas = np.full_like(t_gyr, 1e9)
    # H2: 1e8 (fraction 1e-2) until 7 Gyr, then 1e5 (fraction 1e-5 < 1e-4): the anchor has no measurable H2
    h2 = np.where(t_gyr <= 7.0, 1e8, 1e5)
    P = {"masses.stellar": mstar[:, None], "sfr": (ssfr * mstar)[:, None], "masses.gas": gas[:, None],
         "masses.H2": h2[:, None]}
    r = kl.build_stage_records(P, t_yr, z, np.array([5]), [0], fq)[0]
    assert r["end_is_anchor"] is False and np.isclose(r["fh2_anchor"], 1e-5)
    assert r["row_end"] >= 0 and t_gyr[r["row_end"]] <= 7.0 and np.isclose(r["t_end"], t_yr[r["row_end"]])
    assert t_gyr[r["row_end"] - 1] > 7.0                        # the LAST qualifying snapshot, not an earlier one
    assert kl.stage_time_order(r, stages=["sft", "qt", "end", "anchor"]) == ["sft", "qt", "end", "anchor"]
    # anchor with H2 -> end == anchor
    P["masses.H2"] = np.full_like(t_gyr, 1e8)[:, None]
    r = kl.build_stage_records(P, t_yr, z, np.array([5]), [0], fq)[0]
    assert r["end_is_anchor"] is True and r["row_end"] == 0 and np.isclose(r["t_end"], t_yr[0])
    # a custom threshold; never any H2 above it -> no end row
    r = kl.build_stage_records(P, t_yr, z, np.array([5]), [0], fq, fh2_min=0.5)[0]
    assert r["row_end"] == -1 and np.isnan(r["t_end"]) and "end" not in kl.stage_time_order(r, stages=["end"])
    # no H2 column in the history -> end is the anchor
    del P["masses.H2"]
    r = kl.build_stage_records(P, t_yr, z, np.array([5]), [0], fq)[0]
    assert r["end_is_anchor"] is True and r["row_end"] == 0


# ── the job's tform producer (module imported with stubbed cluster-only deps) ─────────────────────
def _import_job(monkeypatch):
    for name in ("simbanator", "simbanator.io", "simbanator.io.simba", "simbanator.utils",
                 "simbanator.utils.geometry", "build_profiles_job"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["simbanator.io.simba"].Simulation = object
    sys.modules["simbanator.utils.geometry"].shrink_center = lambda *a, **k: None
    sys.modules["simbanator.utils.geometry"].principal_axes = lambda *a, **k: None
    bpj = sys.modules["build_profiles_job"]
    for fn in ("header_units", "_to_kpc", "_to_msun", "_detect", "_components", "_halo_of",
               "_temperature", "_XH", "_nH"):
        setattr(bpj, fn, lambda *a, **k: None)
    bpj.H2_RECIPE = "caesar-v1"
    monkeypatch.setenv("DUST_PLAN", "/dev/null")
    spec = importlib.util.spec_from_file_location("brpj_test", os.path.join(ROOT, "build_reduced_particles_job.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_job_tform_producer_and_backfill_detection(tmp_path, monkeypatch, disc_file):
    job = _import_job(monkeypatch)
    assert "tform" in job.STAR_FIELDS
    assert any("tform" in names and dep == job._DEP_SNAP for names, dep, _ in job.STAR_PRODUCERS)
    # an old-schema file (no tform) is reported as backfillable for exactly that field
    ev = disc_file["evecs"]
    p = write_reduced(str(tmp_path / "old.h5"), ev, disc_file["gpos"], disc_file["spos"],
                      disc_file["m_h2"], np.zeros(4000), disc_file["m_star"], np.zeros(6000),
                      with_tform=False, with_vel=False)
    miss = job._missing_fields(p)
    assert miss == {"gas": {"vel"}, "star": {"tform", "vel"}}
    assert job._needs_snapshot(miss)
    assert job._missing_fields(disc_file["path"]) == {"gas": set(), "star": set()}
    # a file built with the pre-2026-08-27 HI/H2 split (no / other h2_recipe stamp) gets both refreshed
    with h5py.File(p, "a") as f:
        f.attrs["h2_recipe"] = "devis-nh"
    assert job._missing_fields(p) == {"gas": {"m_HI", "m_H2", "vel"}, "star": {"tform", "vel"}}

    # fake ctx: 'take' serves a StellarFormationTime column; producer returns float32
    class Ctx:
        def __init__(self, has):
            self._src = {"gas": None, "star": ({"StellarFormationTime": None} if has else {})}
            self.gal = None

        def has_field(self, part, name):
            src = self._src[part]
            return src is not None and name in src

        def take(self, part, name, idx):
            return np.linspace(0.1, 0.9, len(idx))
    idx = np.arange(5)
    out = job._star_tform(Ctx(True), idx)
    assert out["tform"].dtype == np.float32 and np.allclose(out["tform"], np.linspace(0.1, 0.9, 5))
    out = job._star_tform(Ctx(False), idx)
    assert out["tform"].dtype == np.float32 and np.isnan(out["tform"]).all()
    # _produce_into routes only the requested field
    rec = {"gas": {}, "star": {"idx": idx}}
    job._produce_into(rec, Ctx(True), {"star": {"tform"}})
    assert set(rec["star"]) == {"idx", "tform"}


def test_job_vel_producer_and_backfill_detection(tmp_path, monkeypatch, disc_file):
    job = _import_job(monkeypatch)
    assert "vel" in job.GAS_FIELDS and "vel" in job.STAR_FIELDS
    for prods in (job.GAS_PRODUCERS, job.STAR_PRODUCERS):
        assert any("vel" in names and dep == job._DEP_SNAP for names, dep, _ in prods)
    # a 2026-08-27 file (tform but no vel) is backfillable for exactly `vel` in both groups
    ev = disc_file["evecs"]
    p = write_reduced(str(tmp_path / "novel.h5"), ev, disc_file["gpos"], disc_file["spos"],
                      disc_file["m_h2"], np.zeros(4000), disc_file["m_star"], np.zeros(6000), with_vel=False)
    assert job._missing_fields(p) == {"gas": {"vel"}, "star": {"vel"}}
    assert job._needs_snapshot(job._missing_fields(p))

    class Ctx:
        def __init__(self, has, a=0.25):
            self._src = {"gas": ({"Velocities": None} if has else {}), "star": ({"Velocities": None} if has else {})}
            self.gal, self.a = None, a

        def has_field(self, part, name):
            return name in self._src[part]

        def take(self, part, name, idx):
            return np.tile([2.0, -4.0, 6.0], (len(idx), 1))
    idx = np.arange(5)
    out = job._gas_vel(Ctx(True, a=0.25), idx)
    assert out["vel"].shape == (5, 3) and out["vel"].dtype == np.float32
    assert np.allclose(out["vel"], np.tile([1.0, -2.0, 3.0], (5, 1)))          # x sqrt(a) = 0.5
    out = job._star_vel(Ctx(False), idx)
    assert out["vel"].shape == (5, 3) and np.isnan(out["vel"]).all()
    rec = {"gas": {"idx": idx}, "star": {"idx": idx}}
    job._produce_into(rec, Ctx(True), {"gas": {"vel"}, "star": {"vel"}})
    assert set(rec["gas"]) == {"idx", "vel"} and set(rec["star"]) == {"idx", "vel"}


def test_kappa_rot_cold_disc_hot_sphere_and_floors():
    n = 3000
    pos = _exp_disc(n, 2.0, hz=0.01)
    w = np.full(n, 1.0)
    # a purely rotating thin disc: all kinetic energy is ordered rotation -> kappa = 1, spin axis = +z
    k, ax, used = kl.kappa_rot(pos, _disc_velocities(pos, 200.0), w)
    assert used == n and abs(k - 1.0) < 2e-3 and abs(abs(ax[2]) - 1.0) < 1e-2   # finite-sample bulk motion + thin z-scatter
    # an isotropic dispersion-dominated cloud: kappa ~ 1/3 (one of three velocity components in K_rot)
    sph = RNG.normal(0, 1.0, (n, 3))
    k, _, _ = kl.kappa_rot(sph, RNG.normal(0, 100.0, (n, 3)), w)
    assert 0.25 < k < 0.42
    # rotation with a dispersion in between; the bulk motion of the subset is removed first
    v = _disc_velocities(pos, 150.0, sigma=150.0) + np.array([500.0, -300.0, 100.0])
    k_bulk, _, _ = kl.kappa_rot(pos, v, w)
    k_rest, _, _ = kl.kappa_rot(pos, v - np.array([500.0, -300.0, 100.0]), w)
    assert 0.2 < k_bulk < 0.9 and abs(k_bulk - k_rest) < 1e-9
    # the weight decides who counts: zero-weight particles are dropped, too few -> NaN with the count
    w2 = w.copy()
    w2[10:] = 0.0
    k, ax, used = kl.kappa_rot(pos, _disc_velocities(pos, 200.0), w2, n_min=10)
    assert used == 10 and k > 0.9
    k, ax, used = kl.kappa_rot(pos, _disc_velocities(pos, 200.0), w2, n_min=11)
    assert used == 10 and np.isnan(k) and ax is None
    # unit independence: km/s or 1000x -> the same ratio
    v = _disc_velocities(pos, 150.0, sigma=80.0)
    assert abs(kl.kappa_rot(pos, v, w)[0] - kl.kappa_rot(pos, 1e3 * v, w)[0]) < 1e-9


def test_measure_zone_kinematics_synthetic(tmp_path):
    """Cold rotating H2 disc inside 3 kpc, pressure-supported gas outside it; stars: young rotating
    core, old dispersion-dominated outskirt; a non-member gas clump. Checks the zone masks, the
    per-component kappas, the ages and the member_only / no-vel behaviour."""
    ev = _random_rotation()
    a_to_t = _fake_a_to_t()
    # gas: 2000 cold disc particles at R < 2.5 kpc (all H2), 2000 hot particles in a 4-9 kpc shell (no H2)
    gin = _exp_disc(2000, 0.6, hz=0.02)
    gin = gin[np.hypot(gin[:, 0], gin[:, 1]) < 2.5]
    d = RNG.normal(size=(2000, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    gout = d * RNG.uniform(4.0, 9.0, 2000)[:, None]
    gpos = np.vstack([gin, gout])
    gvel = np.vstack([_disc_velocities(gin, 220.0), RNG.normal(0, 120.0, (2000, 3))])
    m_h2 = np.concatenate([np.full(len(gin), 1e5), np.zeros(2000)])
    gmem = np.ones(len(gpos), bool)
    # + 50 non-member particles at r ~ 6 kpc (H2-bearing) that member_only must drop
    clump = np.tile([5.0, 3.0, 0.5], (50, 1)) + RNG.normal(0, 0.05, (50, 3))
    gpos = np.vstack([gpos, clump]); gvel = np.vstack([gvel, RNG.normal(0, 30.0, (50, 3))])
    m_h2 = np.concatenate([m_h2, np.full(50, 1e5)]); gmem = np.concatenate([gmem, np.zeros(50, bool)])
    # stars: 3000 in a rotating core disc (R < 2.5 kpc, age 1 Gyr), 3000 in a hot 4-9 kpc shell (age 6 Gyr)
    sin = _exp_disc(3000, 0.6, hz=0.05)
    sin = sin[np.hypot(sin[:, 0], sin[:, 1]) < 2.5]
    d = RNG.normal(size=(3000, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    sout = d * RNG.uniform(4.0, 9.0, 3000)[:, None]
    spos = np.vstack([sin, sout])
    svel = np.vstack([_disc_velocities(sin, 180.0, sigma=20.0), RNG.normal(0, 150.0, (3000, 3))])
    m_star = np.full(len(spos), 1e6)
    t_form = np.concatenate([np.full(len(sin), T_OBS - 1.0), np.full(3000, T_OBS - 6.0)])
    p = write_reduced(str(tmp_path / "kin.h5"), ev, gpos, spos, m_h2, np.zeros(len(gpos)), m_star, t_form / 13.0,
                      gas_member=gmem, gas_vel_frame=gvel, star_vel_frame=svel)
    red = kl.load_reduced(p)
    rows = {r["zone"]: r for r in kl.measure_zone_kinematics(red, T_OBS, a_to_t)}
    assert set(rows) == {z for z, _, _ in kl.KIN_ZONES}
    assert set(rows["ap3kpc"]) == set(kl.ZONE_KIN_COLUMNS)
    core, out = rows["ap3kpc"], rows["ann10kpc"]
    assert core["has_vel"] and out["has_vel"]
    # zone masses / counts
    assert core["n_H2"] == len(gin) and core["m_H2"] == pytest.approx(len(gin) * 1e5)
    assert out["n_H2"] == 50 and out["n_gas"] == 2050 and core["n_star"] == len(sin) and out["n_star"] == 3000
    # kappas: cold disc -> ~1 (H2 and gas alike in the core), hot shell gas -> ~1/3, core stars rotating, outskirt stars not
    assert core["kappa_H2"] > 0.98 and core["kappa_gas"] > 0.98
    assert 0.25 < out["kappa_gas"] < 0.42
    assert core["kappa_star"] > 0.9 and out["kappa_star"] < 0.42
    assert abs(core["cos_H2_star"]) > 0.99                      # the two core discs share the spin axis
    # ages: the core is 1 Gyr old, the outskirt 6 Gyr
    assert core["age_mw_gyr"] == pytest.approx(1.0, abs=1e-6) and out["age_mw_gyr"] == pytest.approx(6.0, abs=1e-6)
    assert rows["ap10kpc"]["age_mw_gyr"] == pytest.approx((len(sin) * 1.0 + 3000 * 6.0) / (len(sin) + 3000), abs=1e-6)
    # member_only drops the non-member clump: the outskirt then has no H2 (kappa_H2 NaN, n_H2 = 0)
    rows_m = {r["zone"]: r for r in kl.measure_zone_kinematics(red, T_OBS, a_to_t, member_only=True)}
    assert rows_m["ann10kpc"]["n_H2"] == 0 and np.isnan(rows_m["ann10kpc"]["kappa_H2"])
    assert rows_m["ap3kpc"]["kappa_H2"] == pytest.approx(core["kappa_H2"])
    # the outskirt clump alone (50 particles) is above the floor of 10 but below a floor of 60
    assert np.isfinite(out["kappa_H2"])
    assert np.isnan(kl.measure_zone_kinematics(red, T_OBS, a_to_t, nkin_min=60)[1]["kappa_H2"])
    # a file without `vel`: kappas NaN, counts / masses / ages intact, has_vel False
    p0 = write_reduced(str(tmp_path / "kin_novel.h5"), ev, gpos, spos, m_h2, np.zeros(len(gpos)), m_star, t_form / 13.0,
                       gas_member=gmem, with_vel=False)
    r0 = {r["zone"]: r for r in kl.measure_zone_kinematics(kl.load_reduced(p0), T_OBS, a_to_t)}
    assert not r0["ap3kpc"]["has_vel"] and np.isnan(r0["ap3kpc"]["kappa_H2"]) and np.isnan(r0["ap3kpc"]["kappa_star"])
    assert r0["ap3kpc"]["n_H2"] == core["n_H2"] and r0["ap3kpc"]["age_mw_gyr"] == pytest.approx(1.0, abs=1e-6)
    # too few stars with a formation epoch -> no age
    assert np.isnan(kl.measure_zone_kinematics(red, T_OBS, a_to_t, nstar_min=10 ** 6)[0]["age_mw_gyr"])


def test_relations_and_tdep_ms():
    assert np.isclose(kl.tdep_ms_gyr(0.0), 10 ** 0.09)
    assert kl.tdep_ms_gyr(1.0) < kl.tdep_ms_gyr(0.0)
    y = kl.relation_y("B08", 1.0)
    assert np.isclose(y, -2.06)                                       # B08 normalisation at 10 Msun/pc^2
    assert set(kl.RELATIONS) == {"K98", "B08", "RK19"}


# ── histories on the quench clock (Part 5) ─────────────────────────────────────────────────────────
def test_interp_track_no_extrapolation_and_nan_gaps():
    grid = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    t = np.array([2.0, 0.0, 1.0, 0.5])            # unsorted, covers [0, 2]
    y = np.array([4.0, 0.0, np.nan, 1.0])         # the t = 1 sample is missing -> interpolated across the gap
    out = kl.interp_track(t, y, grid)
    assert np.isnan(out[0])                        # before the first sample: no extrapolation
    assert out[1] == 0.0 and out[2] == 1.0 and out[4] == 4.0
    assert np.isclose(out[3], 1.0 + 3.0 * (0.5 / 1.5))   # between t=0.5 (y=1) and t=2 (y=4) at t=1 -> 2.0
    assert np.isnan(kl.interp_track([1.0], [2.0], grid)).all()          # a single sample: undefined
    assert np.isnan(kl.interp_track([1.0, 1.0], [2.0, 3.0], grid)).all()  # duplicates collapse to one


def test_grid_stats_counts_and_nmin():
    tr = np.array([[1.0, np.nan, 3.0], [2.0, 5.0, np.nan], [3.0, 6.0, np.nan]])
    st = kl.grid_stats(tr, nmin=2)
    assert list(st["n"]) == [3, 2, 1]
    assert st["med"][0] == 2.0 and st["med"][1] == 5.5 and np.isnan(st["med"][2])
    assert np.isclose(st["p16"][0], np.percentile([1.0, 2.0, 3.0], 16))
    v, F = kl.ecdf([3.0, np.nan, 1.0, 2.0])
    assert list(v) == [1.0, 2.0, 3.0] and np.isclose(F[-1], 1.0) and np.isclose(F[0], 1 / 3)
    assert kl.ecdf([np.nan]).__getitem__(0).size == 0
