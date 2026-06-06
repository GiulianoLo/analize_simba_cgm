#!/usr/bin/env python
"""SLURM array worker — CGM temperature at each critical point, parallel over snapshots.

Reuses the SAME extraction plan as the dust-profile job (dust_profile_plan.hdf5): same
(record, stage, galaxy, snapshot) entries. For each entry it takes the parent halo's gas
particles (caesar halo.glist), keeps the non-star-forming gas (the diffuse CGM), computes a
per-particle temperature, and reduces to mass-weighted CGM temperature + phase fractions.

Temperature (Gizmo/SIMBA): T = (gamma-1) * u * mu * m_p / k_B, with
  mu = 4 / (1 + 3*XH + 4*XH*ne)   [mean molecular weight in proton masses]
  u_cgs = InternalEnergy * 1e10   [(km/s)^2 -> (cm/s)^2]
If the snapshot already stores PartType0/Temperature, that is used directly.

Each task writes a partial; run merge_cgm_temperature.py afterwards.
Env: DUST_PLAN (plan file, shared with the dust job), DUST_OUTDIR (partials dir).
"""
import os
import gc
import numpy as np
import h5py

from simbanator.io.simba import Simulation

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
OUT_DIR = os.environ.get("DUST_OUTDIR", os.path.dirname(PLAN_PATH))

GAMMA = 5.0 / 3.0
KB = 1.380649e-16     # erg/K
MP = 1.672622e-24     # g
T_HOT = 1.0e5         # K, hot/warm boundary
T_COLD = 1.0e4        # K, warm/cold boundary


def header_units(f):
    h = f["Header"].attrs if "Header" in f else {}
    return float(h.get("Time", 1.0)), float(h.get("HubbleParam", 0.68))


def _to_msun(m, hub):
    return m * 1e10 / hub


def _detect_fields(f):
    have = set(f["PartType0"].keys())
    Tdir = "Temperature" if "Temperature" in have else None
    u = next((k for k in ("InternalEnergy", "internal_energy") if k in have), None)
    ne = next((k for k in ("ElectronAbundance", "electron_abundance") if k in have), None)
    sfr = next((k for k in ("StarFormationRate", "Sfr", "SFR") if k in have), None)
    Zk = "Metallicity" if "Metallicity" in have else None
    return Tdir, u, ne, sfr, Zk


def _temperature(u_code, ne, XH):
    u_cgs = u_code * 1e10
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * ne)
    return (GAMMA - 1.0) * u_cgs * mu * MP / KB


def _reduce_one(gal, cs, g, hub, fields):
    """Return dict of CGM temperature/phase reductions for one galaxy's halo, or None."""
    Tdir, uk, nek, sfrk, Zk = fields
    halo = getattr(gal, "halo", None)
    if halo is None:
        hi = getattr(gal, "parent_halo_index", -1)
        if hi is None or hi < 0:
            return None
        halo = cs.halos[hi]
    hg = np.unique(np.asarray(getattr(halo, "glist", []), dtype=np.int64))
    if len(hg) == 0:
        return None

    sfr = g[sfrk][hg] if sfrk else np.zeros(len(hg))
    cgm = np.asarray(sfr) == 0                      # non-star-forming = diffuse CGM
    idx = hg[cgm]
    if len(idx) == 0:
        return None

    mass = _to_msun(g["Masses"][idx], hub)
    if Zk is not None:
        Z = g[Zk][idx]
        XH = np.clip(1.0 - Z[:, 0] - Z[:, 1], 0.0, 1.0) if (Z.ndim == 2 and Z.shape[1] >= 2) else np.full(len(idx), 0.76)
    else:
        XH = np.full(len(idx), 0.76)

    if Tdir is not None:
        T = np.asarray(g[Tdir][idx], dtype=np.float64)
    elif uk is not None and nek is not None:
        T = _temperature(np.asarray(g[uk][idx], dtype=np.float64),
                         np.asarray(g[nek][idx], dtype=np.float64), XH)
    else:
        return None

    good = np.isfinite(T) & np.isfinite(mass) & (mass > 0) & (T > 0)
    T, mass = T[good], mass[good]
    if len(T) == 0:
        return None

    M = float(mass.sum())
    w = mass / M
    T_mw = float(10.0 ** np.sum(w * np.log10(T)))   # mass-weighted geometric-mean T
    o = np.argsort(T)
    cw = np.cumsum(mass[o])
    T_med = float(T[o][np.searchsorted(cw, 0.5 * cw[-1])])
    f_hot = float(mass[T > T_HOT].sum() / M)
    f_cold = float(mass[T < T_COLD].sum() / M)
    f_warm = float(max(0.0, 1.0 - f_hot - f_cold))
    return dict(T_mw=T_mw, T_med=T_med, M_cgm=M, f_hot=f_hot, f_warm=f_warm,
                f_cold=f_cold, Ngas=float(len(T)))


def process_snapshot(sim, snap, entries):
    out = []
    cs = sim.load_catalog(snap=snap)
    with h5py.File(sim.get_snapshot_file(snap), "r") as f:
        _, hub = header_units(f)
        fields = _detect_fields(f)
        g = f["PartType0"]
        for (ri, si, gx) in entries:
            try:
                red = _reduce_one(cs.galaxies[gx], cs, g, hub, fields)
            except (IndexError, KeyError):
                red = None
            if red is None:
                continue
            out.append(dict(ri=ri, si=si, **red))
    del cs
    gc.collect()
    return out


KEYS = ["T_mw", "T_med", "M_cgm", "f_hot", "f_warm", "f_cold", "Ngas"]


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
    snaps_all = np.sort(np.unique(snap))[::-1]
    my_snaps = np.array_split(snaps_all, n_task)[task_id] if n_task > 1 else snaps_all
    print(f"[task {task_id}/{n_task}] {len(my_snaps)} snapshots: {list(map(int, my_snaps))}", flush=True)

    results = []
    for k, sn in enumerate(my_snaps):
        m = snap == sn
        entries = list(zip(ri[m].tolist(), si[m].tolist(), gx[m].tolist()))
        try:
            results += process_snapshot(sim, int(sn), entries)
        except OSError as e:
            print(f"  [skip] snapshot {sn}: {e}", flush=True)
            continue
        print(f"  [task {task_id}] {k + 1}/{len(my_snaps)} snap {sn}: {len(entries)} entries", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    part = os.path.join(OUT_DIR, f"cgm_temperature_part_{task_id:03d}.hdf5")
    with h5py.File(part, "w") as out:
        out.create_dataset("ri", data=np.array([r["ri"] for r in results], np.int32))
        out.create_dataset("si", data=np.array([r["si"] for r in results], np.int32))
        for kk in KEYS:
            out.create_dataset(kk, data=np.array([r[kk] for r in results], np.float64))
    print(f"[task {task_id}] wrote {len(results)} entries -> {part}", flush=True)


if __name__ == "__main__":
    main()
