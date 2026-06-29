#!/usr/bin/env python
"""SLURM array worker — write LEAN reduced particle files for flexible Σ / profile work.

Motivation
----------
The fixed-binning products from ``build_profiles_job.py`` bake in one radial grid and select gas
by the CAESAR *galaxy member list* (``gal.glist``, ISM only). This worker instead stores the raw
ingredients so profiles and surface densities can be re-binned at will in the notebook, and selects
particles by a **100 kpc physical aperture around the galaxy** — which captures ISM *and* CGM, not
just the listed galaxy particles.

Loading every gas particle of an m100n1024 snapshot is infeasible (~10^9 particles), so the
candidate set is the **parent halo** gas+star lists (``halo.glist`` / ``halo.slist`` — these already
include the diffuse CGM that ``gal.glist`` omits), spatially cut to ``REDUCED_RMAX`` physical kpc of
the galaxy centre with periodic minimum-image wrapping. For a central galaxy the 100 kpc sphere lies
well inside its FOF halo, so the halo candidate set loses nothing relative to a global spatial query.

For each unique (snapshot, galaxy) in the shared plan (the SAME dust_profile_plan_<tag>.hdf5 used by
build_profiles_job.py) it writes ONE lean HDF5 per galaxy:

  output/<sim>/reduced_particles/snap_NNN/<prefix>_snap<NNN>_gal<GX>.h5
    attrs : sim_name, snap, gx, redshift, a, hub, box_kpc, rmax_kpc,
            center_kpc(3), evecs(3,3)              # stellar principal frame, for face-on projection
    gas/  : pos(n,3) [kpc, RELATIVE to centre], m_gas, m_dust, m_HI, m_H2, sfr   [Msun, Msun/yr]
    star/ : pos(m,3) [kpc, relative], m_star                                     [Msun]

Component masses (dust/HI/H2) are precomputed per particle with the same recipe as the profile job,
so only the minimal columns needed for Σ are stored. Bin them face-on (project onto evecs, R=√(x'²+y'²))
or in 3D (r=‖pos‖); split ISM vs CGM with sfr>0 / sfr==0; choose any aperture — all in the notebook.

Files are keyed globally by (snapshot, galaxy id), so running this per anchor is naturally
idempotent: a galaxy already extracted for one anchor is skipped (before any particle load) when
it recurs in another anchor's plan — no recompute, and every existing file still feeds the stats.
Set REDUCED_OVERWRITE=1 to force re-extraction.

Env: DUST_PLAN (plan, shared with build_profiles_job), REDUCED_RMAX_KPC (default 100),
     REDUCED_PREFIX (default 'm100n1024'), REDUCED_OVERWRITE (default 0).
"""
import os
import gc
import numpy as np
import h5py

from simbanator.io.simba import Simulation
from simbanator.utils.geometry import shrink_center, principal_axes
# reuse the EXACT unit/field recipes the profile job is validated against
from build_profiles_job import header_units, _to_kpc, _to_msun, _detect, _components, _halo_of

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
RMAX = float(os.environ.get("REDUCED_RMAX_KPC", 100.0))     # aperture [physical kpc]
PREFIX = os.environ.get("REDUCED_PREFIX", "m100n1024")
OVERWRITE = int(os.environ.get("REDUCED_OVERWRITE", 0)) == 1


def _box_kpc(f, a, hub):
    bs = float(f["Header"].attrs.get("BoxSize", 0.0))
    return _to_kpc(bs, a, hub) if bs > 0 else None


def _min_image(d, L):
    """Periodic minimum-image displacement (kpc); L None -> no wrapping."""
    return d - L * np.round(d / L) if L else d


def _frame(spos, smass, gpos):
    """Stellar principal frame (centre, eigenvectors); fall back to gas median."""
    if len(spos) >= 10 and np.sum(smass) > 0:
        center = shrink_center(spos, masses=smass)
        _, _, evecs, _ = principal_axes(spos - center, masses=smass)
    elif len(gpos):
        center = np.median(gpos, axis=0)
        evecs = np.eye(3)
    else:
        return None, None
    return center, evecs


def _read_subset(grp, key, idx):
    return np.asarray(grp[key][idx]) if (key and len(idx)) else None


def process_snapshot(sim, snap, gxs):
    """Return list of per-galaxy reduced dicts for the galaxies `gxs` in `snap`."""
    out = []
    cs = sim.load_catalog(snap=snap)
    with h5py.File(sim.get_snapshot_file(snap), "r") as f:
        a, hub = header_units(f)
        L = _box_kpc(f, a, hub)
        z = 1.0 / a - 1.0
        fld = _detect(f)
        g = f["PartType0"]
        s = f["PartType4"] if "PartType4" in f else None
        for gx in gxs:
            try:
                gal = cs.galaxies[gx]
            except (IndexError, KeyError):
                continue
            # --- stellar principal frame from the galaxy's OWN member stars ---
            gsl = np.unique(np.asarray(getattr(gal, "slist", []), dtype=np.int64))
            if s is not None and len(gsl):
                fpos = _to_kpc(s["Coordinates"][gsl], a, hub)
                fmass = _to_msun(s["Masses"][gsl], hub)
            else:
                fpos, fmass = np.empty((0, 3)), np.empty(0)
            gml = np.unique(np.asarray(gal.glist, dtype=np.int64))
            gmpos = _to_kpc(g["Coordinates"][gml], a, hub) if len(gml) else np.empty((0, 3))
            center, evecs = _frame(fpos, fmass, gmpos)
            if center is None:
                continue

            # --- candidate set: parent halo gas+stars (incl. CGM), cut to RMAX of centre ---
            halo = _halo_of(gal, cs)
            cand_g = (np.unique(np.asarray(getattr(halo, "glist", []), dtype=np.int64))
                      if halo is not None else gml)
            cand_s = (np.unique(np.asarray(getattr(halo, "slist", []), dtype=np.int64))
                      if halo is not None else gsl)

            rec = dict(gx=int(gx), z=float(z), a=float(a), hub=float(hub),
                       center=np.asarray(center, float), evecs=np.asarray(evecs, float))

            # gas inside the aperture
            if len(cand_g):
                gpos = _to_kpc(g["Coordinates"][cand_g], a, hub)
                d = _min_image(gpos - center, L)
                keep = np.sum(d * d, axis=1) < RMAX * RMAX
                if np.any(keep):
                    kg = cand_g[keep]
                    mgas = _to_msun(g["Masses"][kg], hub)
                    m_dust, m_HI, m_H2 = _components(
                        mgas,
                        _to_msun(g[fld["dust"]][kg], hub) if fld["dust"] else None,
                        g[fld["Z"]][kg] if fld["Z"] else None,
                        g[fld["fneut"]][kg] if fld["fneut"] else None,
                        g[fld["fmol"]][kg] if fld["fmol"] else None)
                    sfr = (np.asarray(g[fld["sfr"]][kg], float) if fld["sfr"]
                           else np.zeros(len(kg)))
                    rec["gas_pos"] = d[keep].astype(np.float32)
                    rec["m_gas"] = mgas.astype(np.float32)
                    rec["m_dust"] = np.asarray(m_dust, np.float32)
                    rec["m_HI"] = np.asarray(m_HI, np.float32)
                    rec["m_H2"] = np.asarray(m_H2, np.float32)
                    rec["sfr"] = sfr.astype(np.float32)

            # stars inside the aperture
            if s is not None and len(cand_s):
                spos = _to_kpc(s["Coordinates"][cand_s], a, hub)
                d = _min_image(spos - center, L)
                keep = np.sum(d * d, axis=1) < RMAX * RMAX
                if np.any(keep):
                    ks = cand_s[keep]
                    rec["star_pos"] = d[keep].astype(np.float32)
                    rec["m_star"] = _to_msun(s["Masses"][ks], hub).astype(np.float32)
            out.append(rec)
    del cs
    gc.collect()
    return out


def _outname(snap, gx):
    """Per-galaxy output filename (snap, gx) — the global dedup key shared across anchors."""
    return f"{PREFIX}_snap{int(snap):03d}_gal{int(gx):06d}.h5"


def _write(rec, snap, out_dir):
    fpath = os.path.join(out_dir, _outname(snap, rec["gx"]))
    if os.path.exists(fpath) and not OVERWRITE:
        return fpath, True
    with h5py.File(fpath, "w") as o:
        o.attrs["sim_name"] = PREFIX
        o.attrs["snap"] = int(snap)
        o.attrs["gx"] = int(rec["gx"])
        o.attrs["redshift"] = rec["z"]
        o.attrs["a"] = rec["a"]
        o.attrs["hub"] = rec["hub"]
        o.attrs["rmax_kpc"] = RMAX
        o.attrs["center_kpc"] = rec["center"]
        o.attrs["evecs"] = rec["evecs"]
        gg = o.create_group("gas")
        for k, ds in (("pos", "gas_pos"), ("m_gas", "m_gas"), ("m_dust", "m_dust"),
                      ("m_HI", "m_HI"), ("m_H2", "m_H2"), ("sfr", "sfr")):
            if ds in rec:
                gg.create_dataset(k, data=rec[ds], compression="gzip")
        ss = o.create_group("star")
        for k, ds in (("pos", "star_pos"), ("m_star", "m_star")):
            if ds in rec:
                ss.create_dataset(k, data=rec[ds], compression="gzip")
    return fpath, False


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_task = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    with h5py.File(PLAN_PATH, "r") as f:
        sim_name = str(f.attrs["sim_name"])
        gx = f["entry_gx"][:]
        snap = f["entry_snap"][:]

    sim = Simulation(sim_name)
    out_root = os.path.join(os.getcwd(), "output", sim_name, "reduced_particles")

    snaps_all = np.sort(np.unique(snap))[::-1]                 # newest first
    my_snaps = np.array_split(snaps_all, n_task)[task_id] if n_task > 1 else snaps_all
    print(f"[task {task_id}/{n_task}] {len(my_snaps)} snapshots; aperture={RMAX:g} kpc "
          f"prefix='{PREFIX}': {list(map(int, my_snaps))}", flush=True)

    n_written = n_skipped = 0
    for k, sn in enumerate(my_snaps):
        sn = int(sn)
        gxs = np.unique(gx[snap == sn]).astype(np.int64).tolist()   # one file per (snap, galaxy)
        out_dir = os.path.join(out_root, f"snap_{sn:03d}")
        os.makedirs(out_dir, exist_ok=True)
        # dedup: skip galaxies whose reduced file already exists (e.g. extracted for another anchor)
        # BEFORE the heavy particle load — lossless, the existing file still feeds the statistics.
        n_plan = len(gxs)
        if not OVERWRITE:
            gxs = [g for g in gxs if not os.path.exists(os.path.join(out_dir, _outname(sn, g)))]
        n_skipped += n_plan - len(gxs)
        if not gxs:
            print(f"  [task {task_id}] {k + 1}/{len(my_snaps)} snap {sn}: "
                  f"all {n_plan} galaxies already present -> skipped (total skipped {n_skipped})", flush=True)
            continue
        try:
            recs = process_snapshot(sim, sn, gxs)
        except OSError as e:
            print(f"  [skip] snapshot {sn}: {e}", flush=True)
            continue
        for rec in recs:
            _, existed = _write(rec, sn, out_dir)
            n_written += (not existed); n_skipped += existed
        print(f"  [task {task_id}] {k + 1}/{len(my_snaps)} snap {sn}: "
              f"{len(gxs)} new of {n_plan} planned ({n_written} written, {n_skipped} skipped so far)", flush=True)

    print(f"[task {task_id}] done: {n_written} written, {n_skipped} pre-existing -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
