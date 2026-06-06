#!/usr/bin/env python
"""Merge dust-profile partials into the final reduced product.

Output schema matches notebook §5a (``dust_profiles_allcrit.hdf5``), so §5b loads it
unchanged. Reads every ``dust_profiles_part_*.hdf5`` in DUST_OUTDIR and scatters each
partial's (ri, si) entries into the (nrec, nst) arrays.

Env overrides: DUST_PLAN (plan file), DUST_OUTDIR (partials dir), DUST_FINAL (output).
"""
import os
import glob
import numpy as np
import h5py

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
OUT_DIR = os.environ.get("DUST_OUTDIR", os.path.dirname(PLAN_PATH))
FINAL = os.environ.get("DUST_FINAL", os.path.join(OUT_DIR, "dust_profiles_allcrit.hdf5"))


def main():
    with h5py.File(PLAN_PATH, "r") as f:
        nrec = int(f.attrs["nrec"])
        nst = int(f.attrs["nst"])
        rmax = float(f.attrs["rmax_kpc"])
        nbins = int(f.attrs["nbins_r"])
        store_sigma = bool(int(f.attrs["store_sigma"]))
        stages = f.attrs["stages"]
        gid = f["gid"][:]
    edges = np.linspace(0.0, rmax, nbins + 1)
    rmid = 0.5 * (edges[:-1] + edges[1:])

    R50 = {c: np.full((nrec, nst), np.nan) for c in ("dust", "HI", "H2", "star")}
    R90 = np.full((nrec, nst), np.nan)
    SIG = {c: np.full((nrec, nst, nbins), np.nan) for c in ("dust", "HI", "H2", "star")} if store_sigma else None

    parts = sorted(glob.glob(os.path.join(OUT_DIR, "dust_profiles_part_*.hdf5")))
    print(f"merging {len(parts)} partial files")
    tot = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            ri = f["ri"][:]
            si = f["si"][:]
            tot += len(ri)
            for c in ("dust", "HI", "H2", "star"):
                R50[c][ri, si] = f[f"R50_{c}"][:]
            R90[ri, si] = f["R90_dust"][:]
            if store_sigma and "sigma_dust" in f:
                for c in ("dust", "HI", "H2", "star"):
                    SIG[c][ri, si] = f[f"sigma_{c}"][:]

    with h5py.File(FINAL, "w") as out:
        out.create_dataset("rmid", data=rmid)
        out.create_dataset("gid", data=gid)
        out.attrs["stages"] = stages
        for c in ("dust", "HI", "H2", "star"):
            out.create_dataset(f"R50_{c}", data=R50[c])
        out.create_dataset("R90_dust", data=R90)
        if store_sigma:
            for c in ("dust", "HI", "H2", "star"):
                out.create_dataset(f"sigma_{c}", data=SIG[c], compression="gzip")
    print(f"merged {tot} entries from {len(parts)} parts -> {FINAL}")


if __name__ == "__main__":
    main()
