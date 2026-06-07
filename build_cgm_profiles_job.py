#!/usr/bin/env python
"""SLURM array worker — CGM radial Sigma(R) profiles of gas & dust, parallel over snapshots.

Mirrors build_dust_profiles_job.py (face-on projected Sigma(R) in the central galaxy's stellar
principal frame), but operates on the parent HALO's *diffuse* gas — caesar ``halo.glist`` with
``SFR == 0``, i.e. the non-star-forming CGM (same selection as build_cgm_temperature_job.py) —
instead of the galaxy ISM, and out to a larger radius. The projection is centred on the tracked
galaxy's stellar centre, so R = 0 is the galaxy and Sigma(R) is the CGM seen face-on around it
(the tracked QG is normally its halo's central; for a satellite this centres on the satellite).

Reuses the SAME extraction plan as the dust/temperature jobs (``dust_profile_plan.hdf5``): same
(record, stage, galaxy, snapshot) entries — no new plan needed. The radial extent/binning come
from the environment (not the plan, whose rmax is the ISM 50 kpc). Each task writes a partial;
run ``merge_cgm_profiles.py`` afterwards.

Env: DUST_PLAN (plan, shared), DUST_OUTDIR (partials dir),
     CGM_RMAX_KPC (default 300), CGM_NBINS_R (default 30).
"""
import os
import gc
import numpy as np
import h5py

from simbanator.io.simba import Simulation
from simbanator.utils.geometry import shrink_center, principal_axes, rotate_to_frame

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
OUT_DIR = os.environ.get("DUST_OUTDIR", os.path.dirname(PLAN_PATH))
RMAX_KPC = float(os.environ.get("CGM_RMAX_KPC", 300.0))   # CGM extent [physical kpc]
NBINS_R = int(os.environ.get("CGM_NBINS_R", 30))


# ── unit + field helpers (mirror notebook §5 / build_dust_profiles_job) ──────
def header_units(f):
    h = f["Header"].attrs if "Header" in f else {}
    return float(h.get("Time", 1.0)), float(h.get("HubbleParam", 0.68))


def _to_kpc(x, a, hub):
    return x * a / hub


def _to_msun(m, hub):
    return m * 1e10 / hub


def _detect_gas_fields(f):
    have = set(f["PartType0"].keys())
    dust = next((k for k in ("Dust_Masses", "DustMasses") if k in have), None)
    fmol = next((k for k in ("FractionH2", "fH2", "GrackleH2", "f_H2") if k in have), None)
    Zk = "Metallicity" if "Metallicity" in have else None
    fnk = "NeutralHydrogenAbundance" if "NeutralHydrogenAbundance" in have else None
    sfrk = next((k for k in ("StarFormationRate", "Sfr", "SFR") if k in have), None)
    return dust, fmol, Zk, fnk, sfrk


def _components(gf, gmass_code, hub):
    """Return (m_gas, m_dust, m_HI, m_H2) [Msun] for the sliced gas arrays in gf."""
    m_gas = _to_msun(gmass_code, hub)
    m_dust = _to_msun(gf["dust"], hub) if gf.get("dust") is not None else np.full_like(m_gas, np.nan)
    if gf.get("Z") is not None and gf["Z"].ndim == 2 and gf["Z"].shape[1] >= 2:
        XH = np.clip(1.0 - gf["Z"][:, 0] - gf["Z"][:, 1], 0.0, 1.0)
    else:
        XH = np.full_like(m_gas, 0.76)
    m_H = m_gas * XH
    fneut = gf["fneut"] if gf.get("fneut") is not None else np.full_like(m_gas, np.nan)
    fmol = gf.get("fmol")
    if fmol is not None:
        m_H2 = m_H * fneut * fmol
        m_HI = m_H * fneut * (1.0 - fmol)
    else:
        m_H2 = np.full_like(m_gas, np.nan)
        m_HI = m_H * fneut
    return m_gas, m_dust, m_HI, m_H2


def make_profiler(rmax, nbins):
    edges = np.linspace(0.0, rmax, nbins + 1)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    def _profile_one(gpos, mgas, mdust, mHI, mH2, spos, smass):
        # face-on cylindrical radius in the central galaxy's stellar principal frame
        if len(spos) >= 10 and np.sum(smass) > 0:
            center = shrink_center(spos, masses=smass)
            _, _, evecs, _ = principal_axes(spos - center, masses=smass)
        elif len(gpos):
            center = np.median(gpos, axis=0)
            evecs = np.eye(3)
        else:
            return None
        R = np.sqrt(np.sum(rotate_to_frame(gpos, center, evecs)[:, :2] ** 2, axis=1)) if len(gpos) else np.array([])

        def _sig(mass):
            if mass is None or len(R) == 0 or not np.any(np.isfinite(mass)):
                return np.full(nbins, np.nan)
            s, _ = np.histogram(R, bins=edges, weights=np.nan_to_num(mass))
            return s / area

        def _renc(mass, frac):
            if mass is None or len(R) == 0:
                return np.nan
            m = np.nan_to_num(mass)
            o = np.argsort(R)
            c = np.cumsum(m[o])
            if c[-1] <= 0:
                return np.nan
            return float(R[o][np.searchsorted(c, frac * c[-1])])

        return dict(
            sigma_gas=_sig(mgas), sigma_dust=_sig(mdust), sigma_HI=_sig(mHI), sigma_H2=_sig(mH2),
            R50_gas=_renc(mgas, 0.5), R50_dust=_renc(mdust, 0.5),
            M_cgm=float(np.nansum(mgas)), Ngas=float(len(R)),
        )

    return _profile_one, rmid


def _halo_of(gal, cs):
    halo = getattr(gal, "halo", None)
    if halo is None:
        hi = getattr(gal, "parent_halo_index", -1)
        if hi is None or hi < 0:
            return None
        halo = cs.halos[hi]
    return halo


def process_snapshot(sim, snap, entries, profile_one):
    """entries: list of (ri, si, gx). Open the snapshot once; read per-halo CGM slices."""
    out = []
    cs = sim.load_catalog(snap=snap)
    with h5py.File(sim.get_snapshot_file(snap), "r") as f:
        a, hub = header_units(f)
        dk, fk, Zk, fnk, sfrk = _detect_gas_fields(f)
        g = f["PartType0"]
        s = f["PartType4"] if "PartType4" in f else None
        for (ri, si, gx) in entries:
            try:
                gal = cs.galaxies[gx]
            except (IndexError, KeyError):
                continue
            halo = _halo_of(gal, cs)
            if halo is None:
                continue
            hg = np.unique(np.asarray(getattr(halo, "glist", []), dtype=np.int64))
            if len(hg) == 0:
                continue
            sfr = np.asarray(g[sfrk][hg]) if sfrk else np.zeros(len(hg))
            idx = hg[np.asarray(sfr) == 0]                 # non-star-forming = diffuse CGM
            if len(idx) == 0:
                continue
            gpos = _to_kpc(g["Coordinates"][idx], a, hub)
            gf = {"dust": g[dk][idx] if dk else None, "Z": g[Zk][idx] if Zk else None,
                  "fneut": g[fnk][idx] if fnk else None, "fmol": g[fk][idx] if fk else None}
            mgas, mdust, mHI, mH2 = _components(gf, g["Masses"][idx], hub)
            sl = np.unique(np.asarray(getattr(gal, "slist", []), dtype=np.int64))
            if s is not None and len(sl):
                spos = _to_kpc(s["Coordinates"][sl], a, hub)
                smass = _to_msun(s["Masses"][sl], hub)
            else:
                spos, smass = np.empty((0, 3)), np.empty(0)
            prof = profile_one(gpos, mgas, mdust, mHI, mH2, spos, smass)
            if prof is None:
                continue
            out.append(dict(ri=ri, si=si, **prof))
    del cs
    gc.collect()
    return out


SIG_KEYS = ("gas", "dust", "HI", "H2")
SCAL_KEYS = ("R50_gas", "R50_dust", "M_cgm", "Ngas")


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_task = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    with h5py.File(PLAN_PATH, "r") as f:
        sim_name = f.attrs["sim_name"]
        ri = f["entry_ri"][:]
        si = f["entry_si"][:]
        gx = f["entry_gx"][:]
        snap = f["entry_snap"][:]

    sim = Simulation(str(sim_name))
    profile_one, _ = make_profiler(RMAX_KPC, NBINS_R)

    snaps_all = np.sort(np.unique(snap))[::-1]
    my_snaps = np.array_split(snaps_all, n_task)[task_id] if n_task > 1 else snaps_all
    print(f"[task {task_id}/{n_task}] {len(my_snaps)} snapshots; RMAX={RMAX_KPC} kpc NBINS={NBINS_R}: "
          f"{list(map(int, my_snaps))}", flush=True)

    results = []
    for k, sn in enumerate(my_snaps):
        m = snap == sn
        entries = list(zip(ri[m].tolist(), si[m].tolist(), gx[m].tolist()))
        try:
            results += process_snapshot(sim, int(sn), entries, profile_one)
        except OSError as e:
            print(f"  [skip] snapshot {sn}: {e}", flush=True)
            continue
        print(f"  [task {task_id}] {k + 1}/{len(my_snaps)} snap {sn}: {len(entries)} entries", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    part = os.path.join(OUT_DIR, f"cgm_profiles_part_{task_id:03d}.hdf5")
    n = len(results)
    with h5py.File(part, "w") as out:
        out.attrs["rmax_kpc"] = RMAX_KPC
        out.attrs["nbins_r"] = NBINS_R
        out.create_dataset("ri", data=np.array([r["ri"] for r in results], np.int32))
        out.create_dataset("si", data=np.array([r["si"] for r in results], np.int32))
        for c in SCAL_KEYS:
            out.create_dataset(c, data=np.array([r[c] for r in results], np.float64))
        for c in SIG_KEYS:
            arr = np.array([r[f"sigma_{c}"] for r in results], np.float64) if n else np.empty((0, NBINS_R))
            out.create_dataset(f"sigma_{c}", data=arr, compression="gzip")
    print(f"[task {task_id}] wrote {n} entries -> {part}", flush=True)


if __name__ == "__main__":
    main()
