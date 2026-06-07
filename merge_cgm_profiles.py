#!/usr/bin/env python
"""Merge CGM radial-profile partials into the final product (cgm_profiles_allcrit.hdf5).

Output schema (loaded by notebook §7h):
  rmid                       (nbins,)        radial bin midpoints [physical kpc]
  sigma_gas/dust/HI/H2       (nrec, nst, nbins)  face-on projected surface density [Msun/kpc^2]
  R50_gas/R50_dust/M_cgm/Ngas(nrec, nst)     half-mass radii [kpc], CGM gas mass [Msun], N particles
  gid                        (nrec,)         GroupID; 'stages' attr like the dust product

Reads every cgm_profiles_part_*.hdf5 in DUST_OUTDIR and scatters each partial's (ri, si)
entries into the (nrec, nst[, nbins]) arrays. rmax/nbins are taken from the partials (they were
set from CGM_RMAX_KPC / CGM_NBINS_R at build time, not from the plan).

Env: DUST_PLAN (plan file), DUST_OUTDIR (partials dir), CGMPROF_FINAL (output path).
"""
import os
import glob
import numpy as np
import h5py

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
OUT_DIR = os.environ.get("DUST_OUTDIR", os.path.dirname(PLAN_PATH))
FINAL = os.environ.get("CGMPROF_FINAL", os.path.join(OUT_DIR, "cgm_profiles_allcrit.hdf5"))

SIG_KEYS = ("gas", "dust", "HI", "H2")
SCAL_KEYS = ("R50_gas", "R50_dust", "M_cgm", "Ngas")


def main():
    with h5py.File(PLAN_PATH, "r") as f:
        nrec = int(f.attrs["nrec"])
        nst = int(f.attrs["nst"])
        stages = f.attrs["stages"]
        gid = f["gid"][:]

    parts = sorted(glob.glob(os.path.join(OUT_DIR, "cgm_profiles_part_*.hdf5")))
    if not parts:
        raise SystemExit("no cgm_profiles_part_*.hdf5 partials found in " + OUT_DIR)
    with h5py.File(parts[0], "r") as f:
        rmax = float(f.attrs["rmax_kpc"])
        nbins = int(f.attrs["nbins_r"])
    edges = np.linspace(0.0, rmax, nbins + 1)
    rmid = 0.5 * (edges[:-1] + edges[1:])

    SIG = {c: np.full((nrec, nst, nbins), np.nan) for c in SIG_KEYS}
    Q = {c: np.full((nrec, nst), np.nan) for c in SCAL_KEYS}

    print(f"merging {len(parts)} partial files  (rmax={rmax:g} kpc, nbins={nbins})")
    tot = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            ri = f["ri"][:]
            si = f["si"][:]
            tot += len(ri)
            for c in SCAL_KEYS:
                Q[c][ri, si] = f[c][:]
            for c in SIG_KEYS:
                if f"sigma_{c}" in f and len(ri):
                    SIG[c][ri, si] = f[f"sigma_{c}"][:]

    with h5py.File(FINAL, "w") as out:
        out.attrs["stages"] = stages
        out.attrs["rmax_kpc"] = rmax
        out.attrs["nbins_r"] = nbins
        out.create_dataset("rmid", data=rmid)
        out.create_dataset("gid", data=gid)
        for c in SCAL_KEYS:
            out.create_dataset(c, data=Q[c])
        for c in SIG_KEYS:
            out.create_dataset(f"sigma_{c}", data=SIG[c], compression="gzip")
    print(f"merged {tot} entries from {len(parts)} parts -> {FINAL}")


if __name__ == "__main__":
    main()
