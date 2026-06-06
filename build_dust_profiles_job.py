#!/usr/bin/env python
"""SLURM array worker — build the dust-profile reduced product in parallel.

Snapshots are independent, so each array task processes a disjoint chunk of snapshots
(split by SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT) and writes a partial HDF5.
Run ``merge_dust_profiles.py`` afterwards to assemble the final product.

Inputs come from the extraction plan written by the notebook cell ``5a·plan``
(``dust_profile_plan.hdf5``). Mirrors the per-galaxy logic of notebook §5a/§5-helpers.

Env overrides: DUST_PLAN (plan file), DUST_OUTDIR (where partials are written).
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


# ── unit + field helpers (mirror notebook §5 helpers) ───────────────────────
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
    return dust, fmol, Zk, fnk


def _components(gf, gmass_code, hub):
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
    return m_dust, m_HI, m_H2


def make_profiler(rmax, nbins):
    edges = np.linspace(0.0, rmax, nbins + 1)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    def _profile_one(gpos, gdust, gHI, gH2, spos, smass):
        if len(spos) >= 10 and np.sum(smass) > 0:
            center = shrink_center(spos, masses=smass)
            _, _, evecs, _ = principal_axes(spos - center, masses=smass)
        elif len(gpos):
            center = np.median(gpos, axis=0)
            evecs = np.eye(3)
        else:
            return None
        R = np.sqrt(np.sum(rotate_to_frame(gpos, center, evecs)[:, :2] ** 2, axis=1)) if len(gpos) else np.array([])
        Rs = np.sqrt(np.sum(rotate_to_frame(spos, center, evecs)[:, :2] ** 2, axis=1)) if len(spos) else np.array([])

        def _sig(Rarr, mass):
            if mass is None or len(Rarr) == 0 or not np.any(np.isfinite(mass)):
                return np.full(nbins, np.nan)
            s, _ = np.histogram(Rarr, bins=edges, weights=np.nan_to_num(mass))
            return s / area

        def _renc(Rarr, mass, frac):
            if mass is None or len(Rarr) == 0:
                return np.nan
            m = np.nan_to_num(mass)
            o = np.argsort(Rarr)
            c = np.cumsum(m[o])
            if c[-1] <= 0:
                return np.nan
            return float(Rarr[o][np.searchsorted(c, frac * c[-1])])

        return dict(
            sigma_dust=_sig(R, gdust), sigma_HI=_sig(R, gHI), sigma_H2=_sig(R, gH2), sigma_star=_sig(Rs, smass),
            R50_dust=_renc(R, gdust, 0.5), R90_dust=_renc(R, gdust, 0.9),
            R50_HI=_renc(R, gHI, 0.5), R50_H2=_renc(R, gH2, 0.5), R50_star=_renc(Rs, smass, 0.5),
        )

    return _profile_one, rmid


def process_snapshot(sim, snap, entries, profile_one):
    """entries: list of (ri, si, gx). Open the snapshot once; read per-galaxy slices."""
    out = []
    cs = sim.load_catalog(snap=snap)
    with h5py.File(sim.get_snapshot_file(snap), "r") as f:
        a, hub = header_units(f)
        dk, fk, Zk, fnk = _detect_gas_fields(f)
        g = f["PartType0"]
        s = f["PartType4"] if "PartType4" in f else None
        for (ri, si, gx) in entries:
            gal = cs.galaxies[gx]
            gl = np.unique(np.asarray(gal.glist, dtype=np.int64))
            sl = np.unique(np.asarray(getattr(gal, "slist", []), dtype=np.int64))
            if len(gl) == 0:
                continue
            gpos = _to_kpc(g["Coordinates"][gl], a, hub)
            gf = {"dust": g[dk][gl] if dk else None, "Z": g[Zk][gl] if Zk else None,
                  "fneut": g[fnk][gl] if fnk else None, "fmol": g[fk][gl] if fk else None}
            m_dust, m_HI, m_H2 = _components(gf, g["Masses"][gl], hub)
            if s is not None and len(sl):
                spos = _to_kpc(s["Coordinates"][sl], a, hub)
                smass = _to_msun(s["Masses"][sl], hub)
            else:
                spos, smass = np.empty((0, 3)), np.empty(0)
            prof = profile_one(gpos, m_dust, m_HI, m_H2, spos, smass)
            if prof is None:
                continue
            out.append(dict(ri=ri, si=si, **prof))
    del cs
    gc.collect()
    return out


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_task = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    with h5py.File(PLAN_PATH, "r") as f:
        sim_name = f.attrs["sim_name"]
        rmax = float(f.attrs["rmax_kpc"])
        nbins = int(f.attrs["nbins_r"])
        store_sigma = bool(int(f.attrs["store_sigma"]))
        ri = f["entry_ri"][:]
        si = f["entry_si"][:]
        gx = f["entry_gx"][:]
        snap = f["entry_snap"][:]

    sim = Simulation(str(sim_name))
    profile_one, _ = make_profiler(rmax, nbins)

    # split snapshots (newest first) across array tasks
    snaps_all = np.sort(np.unique(snap))[::-1]
    my_snaps = np.array_split(snaps_all, n_task)[task_id] if n_task > 1 else snaps_all
    print(f"[task {task_id}/{n_task}] {len(my_snaps)} snapshots: {list(map(int, my_snaps))}", flush=True)

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
    part = os.path.join(OUT_DIR, f"dust_profiles_part_{task_id:03d}.hdf5")
    n = len(results)
    with h5py.File(part, "w") as out:
        out.attrs["store_sigma"] = int(store_sigma)
        out.create_dataset("ri", data=np.array([r["ri"] for r in results], np.int32))
        out.create_dataset("si", data=np.array([r["si"] for r in results], np.int32))
        for c in ("dust", "HI", "H2", "star"):
            out.create_dataset(f"R50_{c}", data=np.array([r[f"R50_{c}"] for r in results], np.float64))
        out.create_dataset("R90_dust", data=np.array([r["R90_dust"] for r in results], np.float64))
        if store_sigma:
            for c in ("dust", "HI", "H2", "star"):
                arr = np.array([r[f"sigma_{c}"] for r in results], np.float64) if n else np.empty((0, nbins))
                out.create_dataset(f"sigma_{c}", data=arr, compression="gzip")
    print(f"[task {task_id}] wrote {n} entries -> {part}", flush=True)


if __name__ == "__main__":
    main()
