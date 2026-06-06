#!/usr/bin/env python
"""Merge CGM-temperature partials into the final product (cgm_temperature_allcrit.hdf5).

Schema: (nrec, nst) arrays T_mw, T_med, M_cgm, f_hot, f_warm, f_cold, Ngas; plus gid and a
``stages`` attribute. Loaded by the notebook §7 CGM-temperature cell.

Env: DUST_PLAN (plan file), DUST_OUTDIR (partials dir), CGMT_FINAL (output path).
"""
import os
import glob
import numpy as np
import h5py

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
OUT_DIR = os.environ.get("DUST_OUTDIR", os.path.dirname(PLAN_PATH))
FINAL = os.environ.get("CGMT_FINAL", os.path.join(OUT_DIR, "cgm_temperature_allcrit.hdf5"))

KEYS = ["T_mw", "T_med", "M_cgm", "f_hot", "f_warm", "f_cold", "Ngas"]


def main():
    with h5py.File(PLAN_PATH, "r") as f:
        nrec = int(f.attrs["nrec"])
        nst = int(f.attrs["nst"])
        stages = f.attrs["stages"]
        gid = f["gid"][:]

    A = {k: np.full((nrec, nst), np.nan) for k in KEYS}
    parts = sorted(glob.glob(os.path.join(OUT_DIR, "cgm_temperature_part_*.hdf5")))
    print(f"merging {len(parts)} partial files")
    tot = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            ri = f["ri"][:]
            si = f["si"][:]
            tot += len(ri)
            for k in KEYS:
                A[k][ri, si] = f[k][:]

    with h5py.File(FINAL, "w") as out:
        out.attrs["stages"] = stages
        out.create_dataset("gid", data=gid)
        for k in KEYS:
            out.create_dataset(k, data=A[k])
    print(f"merged {tot} entries from {len(parts)} parts -> {FINAL}")


if __name__ == "__main__":
    main()
