#!/usr/bin/env python
"""Merge unified profile partials (profiles_part_*.hdf5 from build_profiles_job.py) into the
THREE original product files, with their ORIGINAL schemas so the notebook loaders work unchanged:

  dust_profiles_allcrit.hdf5   §5b   rmid, gid, stages; R20/R50/R80_{dust,HI,H2,star}, R90_dust,
                                     sigma_{dust,HI,H2,star}                       (nrec, nst[, nbins])
  cgm_profiles_allcrit.hdf5    §7p   rmid, gid, stages, rmax/nbins; sigma_{gas,dust,HI,H2},
                                     R50_gas, R50_dust, M_cgm, Ngas               (nrec, nst[, nbins])
  cgm_temperature_allcrit.hdf5 §7g   gid, stages; T_mw, T_med, M_cgm, f_hot, f_warm, f_cold, Ngas,
                                     PLUS the new mass-weighted T(R): rmid_cgm, cgm_Tprof

Env: DUST_PLAN (plan), DUST_OUTDIR (partials dir),
     DUST_FINAL / CGMPROF_FINAL / CGMT_FINAL (output paths).
"""
import os
import glob
import numpy as np
import h5py

PLAN_PATH = os.environ.get(
    "DUST_PLAN", os.path.join("output", "cis100", "caesar_sfh", "dust_profile_plan.hdf5"))
OUT_DIR = os.environ.get("DUST_OUTDIR", os.path.dirname(PLAN_PATH))
DUST_FINAL = os.environ.get("DUST_FINAL", os.path.join(OUT_DIR, "dust_profiles_allcrit.hdf5"))
CGMPROF_FINAL = os.environ.get("CGMPROF_FINAL", os.path.join(OUT_DIR, "cgm_profiles_allcrit.hdf5"))
CGMT_FINAL = os.environ.get("CGMT_FINAL", os.path.join(OUT_DIR, "cgm_temperature_allcrit.hdf5"))

ISM_COMPS = ("dust", "HI", "H2", "star")
CGM_SIG = ("gas", "dust", "HI", "H2")


def main():
    with h5py.File(PLAN_PATH, "r") as f:
        nrec = int(f.attrs["nrec"])
        nst = int(f.attrs["nst"])
        stages = f.attrs["stages"]
        gid = f["gid"][:]

    parts = sorted(glob.glob(os.path.join(OUT_DIR, "profiles_part_*.hdf5")))
    if not parts:
        raise SystemExit("no profiles_part_*.hdf5 partials found in " + OUT_DIR)
    with h5py.File(parts[0], "r") as f:
        inb = int(f.attrs["ism_nbins_r"]); irmax = float(f.attrs["ism_rmax_kpc"])
        cnb = int(f.attrs["cgm_nbins_r"]); crmax = float(f.attrs["cgm_rmax_kpc"])
    irmid = 0.5 * (np.linspace(0, irmax, inb + 1)[:-1] + np.linspace(0, irmax, inb + 1)[1:])
    crmid = 0.5 * (np.linspace(0, crmax, cnb + 1)[:-1] + np.linspace(0, crmax, cnb + 1)[1:])

    # allocate (nrec, nst[, nbins]) targets
    iscal = {f"{q}_{c}": np.full((nrec, nst), np.nan)
             for c in ISM_COMPS for q in ("R20", "R50", "R80")}
    iscal["R90_dust"] = np.full((nrec, nst), np.nan)
    isig = {c: np.full((nrec, nst, inb), np.nan) for c in ISM_COMPS}
    cscal = {k: np.full((nrec, nst), np.nan) for k in ("cgm_R50_gas", "cgm_R50_dust", "M_cgm", "cgm_Ngas")}
    csig = {c: np.full((nrec, nst, cnb), np.nan) for c in CGM_SIG}
    tscal = {k: np.full((nrec, nst), np.nan) for k in ("T_mw", "T_med", "M_cgm_T", "f_hot", "f_warm", "f_cold", "T_Ngas")}
    tprof = np.full((nrec, nst, cnb), np.nan)

    print(f"merging {len(parts)} partials  (ISM {irmax:g}/{inb}, CGM {crmax:g}/{cnb})")
    tot = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            ri = f["ri"][:]; si = f["si"][:]; tot += len(ri)
            if not len(ri):
                continue
            for k in iscal:
                iscal[k][ri, si] = f[k][:]
            for c in ISM_COMPS:
                isig[c][ri, si] = f[f"ism_sigma_{c}"][:]
            for k in cscal:
                cscal[k][ri, si] = f[k][:]
            for c in CGM_SIG:
                csig[c][ri, si] = f[f"cgm_sigma_{c}"][:]
            for k in tscal:
                tscal[k][ri, si] = f[k][:]
            tprof[ri, si] = f["cgm_Tprof"][:]

    # ---- dust_profiles_allcrit.hdf5 (§5b schema) ----
    with h5py.File(DUST_FINAL, "w") as out:
        out.attrs["stages"] = stages
        out.create_dataset("rmid", data=irmid)
        out.create_dataset("gid", data=gid)
        for c in ISM_COMPS:
            for q in ("R20", "R50", "R80"):
                out.create_dataset(f"{q}_{c}", data=iscal[f"{q}_{c}"])
            out.create_dataset(f"sigma_{c}", data=isig[c], compression="gzip")
        out.create_dataset("R90_dust", data=iscal["R90_dust"])
    print("wrote", DUST_FINAL)

    # ---- cgm_profiles_allcrit.hdf5 (§7p schema) ----
    with h5py.File(CGMPROF_FINAL, "w") as out:
        out.attrs["stages"] = stages
        out.attrs["rmax_kpc"] = crmax
        out.attrs["nbins_r"] = cnb
        out.create_dataset("rmid", data=crmid)
        out.create_dataset("gid", data=gid)
        out.create_dataset("R50_gas", data=cscal["cgm_R50_gas"])
        out.create_dataset("R50_dust", data=cscal["cgm_R50_dust"])
        out.create_dataset("M_cgm", data=cscal["M_cgm"])
        out.create_dataset("Ngas", data=cscal["cgm_Ngas"])
        for c in CGM_SIG:
            out.create_dataset(f"sigma_{c}", data=csig[c], compression="gzip")
    print("wrote", CGMPROF_FINAL)

    # ---- cgm_temperature_allcrit.hdf5 (§7g schema + new T(R)) ----
    with h5py.File(CGMT_FINAL, "w") as out:
        out.attrs["stages"] = stages
        out.create_dataset("gid", data=gid)
        out.create_dataset("T_mw", data=tscal["T_mw"])
        out.create_dataset("T_med", data=tscal["T_med"])
        out.create_dataset("M_cgm", data=tscal["M_cgm_T"])     # valid-T diffuse mass (original semantics)
        out.create_dataset("f_hot", data=tscal["f_hot"])
        out.create_dataset("f_warm", data=tscal["f_warm"])
        out.create_dataset("f_cold", data=tscal["f_cold"])
        out.create_dataset("Ngas", data=tscal["T_Ngas"])
        out.create_dataset("rmid_cgm", data=crmid)             # NEW: radial grid for T(R)
        out.create_dataset("cgm_Tprof", data=tprof, compression="gzip")  # NEW: mass-weighted T(R)
    print("wrote", CGMT_FINAL)
    print(f"merged {tot} entries from {len(parts)} parts")


if __name__ == "__main__":
    main()
