"""HI/H2 split must follow caesar (hydrogen_mass_calc.pyx: get_HIH2_masses) so particle sums reproduce
the catalogue masses.H2 / masses.HI. Regression for the 2026-08-27 bug (m_H * nh * fH2, which lost up
to 97% of the H2 in SIMBA's star-forming gas because NeutralHydrogenAbundance ~ 0.003 there)."""
import importlib.util
import os
import sys
import types

import h5py
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_simbanator(monkeypatch):
    for name in ("simbanator", "simbanator.io", "simbanator.io.simba", "simbanator.utils",
                 "simbanator.utils.geometry"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["simbanator.io.simba"].Simulation = object
    for fn in ("shrink_center", "principal_axes", "rotate_to_frame"):
        setattr(sys.modules["simbanator.utils.geometry"], fn, lambda *a, **k: None)


def _load(name, fname, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, fname))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bpj(monkeypatch):
    _stub_simbanator(monkeypatch)
    monkeypatch.setenv("DUST_PLAN", "/dev/null")
    return _load("build_profiles_job", "build_profiles_job.py", monkeypatch)


@pytest.fixture
def job(bpj, monkeypatch):
    return _load("brpj_h2_test", "build_reduced_particles_job.py", monkeypatch)


def test_components_caesar_split(bpj):
    m = np.array([10.0, 10.0, 10.0, 10.0])
    nh = np.array([0.003, 0.996, 0.5, 0.0])        # Grackle HI fraction (hot-EOS SF gas ~0, cold ~1)
    fh2 = np.array([0.9, 0.7, 0.2, 0.0])
    Z = np.tile([0.03, 0.37], (4, 1))              # enriched ISM: 1-Z-He = 0.60 must NOT be used
    m_dust, m_HI, m_H2 = bpj._components(m, None, Z, nh, fh2)
    np.testing.assert_allclose(m_H2, 0.76 * m * fh2)                       # no nh factor
    np.testing.assert_allclose(m_HI, 0.76 * m * np.minimum(nh, 1 - fh2))   # caesar clamp
    assert np.all(np.isnan(m_dust))


def test_components_density_cut_and_missing_fields(bpj):
    m = np.array([10.0, 10.0])
    nh = np.array([0.5, 0.5]); fh2 = np.array([0.4, 0.4])
    nH = np.array([1.0, 0.01])                     # second particle below caesar's 0.13 cm^-3
    _, m_HI, m_H2 = bpj._components(m, None, None, nh, fh2, nH=nH)
    np.testing.assert_allclose(m_H2, [0.76 * 10 * 0.4, 0.0])
    np.testing.assert_allclose(m_HI, [0.76 * 10 * 0.5, 0.76 * 10 * 0.5])
    _, m_HI, m_H2 = bpj._components(m, None, None, nh, None)      # no fH2 field
    assert np.all(np.isnan(m_H2)) and np.allclose(m_HI, 0.76 * m * nh)


def test_nH_recipe(bpj):
    a, hub = 0.5, 0.68                             # rho_code = 1e10 Msun/h per (ckpc/h)^3
    expect = 1e-3 * 1e10 * 1.989e33 * hub ** 2 / (3.085678e21 * a) ** 3 * 0.76 / 1.672622e-24
    np.testing.assert_allclose(bpj._nH(np.array([1e-3]), a, hub), expect, rtol=1e-6)


def test_detect_density(bpj, tmp_path):
    with h5py.File(tmp_path / "s.h5", "w") as f:
        g = f.create_group("PartType0")
        for k in ("Masses", "Density", "FractionH2", "NeutralHydrogenAbundance", "Metallicity"):
            g.create_dataset(k, data=np.zeros(2))
        fld = bpj._detect(f)
    assert fld["rho"] == "Density" and fld["fmol"] == "FractionH2"


def test_reduced_job_stale_stamp_and_producer(job, bpj, tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        for grp, fields in (("gas", job.GAS_FIELDS), ("star", job.STAR_FIELDS)):
            g = f.create_group(grp)
            for k in ("idx", "pos") + tuple(fields):
                g.create_dataset(k, data=np.zeros(3))
    miss = job._missing_fields(str(p))                       # complete but unstamped -> stale split
    assert miss == {"gas": {"m_HI", "m_H2"}, "star": set()} and job._needs_snapshot(miss)
    # backfilling m_HI/m_H2 must stamp the file so it is not redone next run
    job._append_fields(str(p), {"gas": {"m_HI": np.ones(3), "m_H2": np.ones(3)}})
    assert job._missing_fields(str(p)) == {"gas": set(), "star": set()}

    # producer: Masses*1e10/h * 0.76 * fH2, density cut applied through ctx.take('Density')
    class Ctx:
        fld = dict(dust=None, Z=None, fneut="NeutralHydrogenAbundance", fmol="FractionH2", rho="Density")
        a, hub, gal = 0.5, 0.68, None
        cols = {"Masses": np.array([1.0, 1.0]), "FractionH2": np.array([0.5, 0.5]),
                "NeutralHydrogenAbundance": np.array([0.9, 0.9]), "Density": np.array([1e-3, 1e-9])}
        def take(self, part, name, idx):
            return self.cols[name][idx]
    out = job._gas_components(Ctx(), np.array([0, 1]))
    np.testing.assert_allclose(out["m_H2"], [1e10 / 0.68 * 0.76 * 0.5, 0.0], rtol=1e-6)
    np.testing.assert_allclose(out["m_HI"], [1e10 / 0.68 * 0.76 * 0.5, 1e10 / 0.68 * 0.76 * 0.9], rtol=1e-6)
