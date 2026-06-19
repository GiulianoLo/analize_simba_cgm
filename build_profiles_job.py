#!/usr/bin/env python
"""Unified SLURM array worker — ISM dust profiles + CGM Sigma(R) profiles + CGM temperature
in a SINGLE pass over each snapshot.

Replaces build_dust_profiles_job.py + build_cgm_profiles_job.py + build_cgm_temperature_job.py:
the snapshot file is opened ONCE per snapshot and the parent-halo diffuse gas is read ONCE
(shared by the CGM profile and the CGM temperature reductions) instead of three independent
passes -> ~3x less I/O, which is where the wall-clock goes.

For each (record, stage, galaxy, snapshot) entry of the shared plan (dust_profile_plan.hdf5):
  ISM  : galaxy gas (gal.glist), face-on cylindrical Sigma(R) in the stellar principal frame out
         to the plan's rmax (~50 kpc) -> R20/R50/R80 (dust/HI/H2/star), R90_dust, Sigma(R).
  CGM  : parent-halo diffuse gas (halo.glist & SFR==0), the SAME face-on frame, out to
         CGM_RMAX_KPC (~300 kpc) -> Sigma_gas/dust/HI/H2(R), R50_gas, R50_dust, M_cgm, Ngas,
         and a mass-weighted T(R) profile.
  CGMT : the same diffuse gas -> mass-weighted T_mw, T_med, phase fractions f_hot/warm/cold, Ngas.

Each task writes ONE partial (profiles_part_*.hdf5); run merge_profiles.py to assemble the three
product files (dust_profiles_allcrit.hdf5, cgm_profiles_allcrit.hdf5, cgm_temperature_allcrit.hdf5)
with their ORIGINAL schemas, so the notebook loaders (§5b, §7p, §7g) work unchanged.

Env: DUST_PLAN (plan), DUST_OUTDIR (partials dir), CGM_RMAX_KPC (default 300), CGM_NBINS_R (30).
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
CGM_RMAX = float(os.environ.get("CGM_RMAX_KPC", 300.0))   # CGM extent [physical kpc]
CGM_NBINS = int(os.environ.get("CGM_NBINS_R", 30))

GAMMA, KB, MP = 5.0 / 3.0, 1.380649e-16, 1.672622e-24
T_HOT, T_COLD = 1.0e5, 1.0e4


# ── unit + field helpers ────────────────────────────────────────────────────
def header_units(f):
    h = f["Header"].attrs if "Header" in f else {}
    return float(h.get("Time", 1.0)), float(h.get("HubbleParam", 0.68))


def _to_kpc(x, a, hub):
    return x * a / hub


def _to_msun(m, hub):
    return m * 1e10 / hub


def _detect(f):
    have = set(f["PartType0"].keys())
    return dict(
        dust=next((k for k in ("Dust_Masses", "DustMasses") if k in have), None),
        fmol=next((k for k in ("FractionH2", "fH2", "GrackleH2", "f_H2") if k in have), None),
        Z="Metallicity" if "Metallicity" in have else None,
        fneut="NeutralHydrogenAbundance" if "NeutralHydrogenAbundance" in have else None,
        sfr=next((k for k in ("StarFormationRate", "Sfr", "SFR") if k in have), None),
        Tdir="Temperature" if "Temperature" in have else None,
        u=next((k for k in ("InternalEnergy", "internal_energy") if k in have), None),
        ne=next((k for k in ("ElectronAbundance", "electron_abundance") if k in have), None),
    )


def _XH(Zarr, n):
    if Zarr is not None and Zarr.ndim == 2 and Zarr.shape[1] >= 2:
        return np.clip(1.0 - Zarr[:, 0] - Zarr[:, 1], 0.0, 1.0)
    return np.full(n, 0.76)


def _components(mgas, dust_msun, Zarr, fneutarr, fmolarr):
    """mgas already in Msun; dust already in Msun (or None). Returns (m_dust, m_HI, m_H2)."""
    m_dust = dust_msun if dust_msun is not None else np.full_like(mgas, np.nan)
    XH = _XH(Zarr, len(mgas))
    m_H = mgas * XH
    fneut = fneutarr if fneutarr is not None else np.full_like(mgas, np.nan)
    if fmolarr is not None:
        m_H2 = m_H * fneut * fmolarr
        m_HI = m_H * fneut * (1.0 - fmolarr)
    else:
        m_H2 = np.full_like(mgas, np.nan)
        m_HI = m_H * fneut
    return m_dust, m_HI, m_H2


def _temperature(u_code, ne, XH):
    u_cgs = u_code * 1e10
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * ne)
    return (GAMMA - 1.0) * u_cgs * mu * MP / KB


def _make_bins(rmax, nbins):
    edges = np.linspace(0.0, rmax, nbins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:]), np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


def _renc(R, mass, frac):
    if mass is None or len(R) == 0:
        return np.nan
    m = np.nan_to_num(mass)
    o = np.argsort(R)
    c = np.cumsum(m[o])
    if c[-1] <= 0:
        return np.nan
    return float(R[o][np.searchsorted(c, frac * c[-1])])


def _sig(R, mass, edges, area, nbins):
    if mass is None or len(R) == 0 or not np.any(np.isfinite(mass)):
        return np.full(nbins, np.nan)
    s, _ = np.histogram(R, bins=edges, weights=np.nan_to_num(mass))
    return s / area


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


def _cylR(pos, center, evecs):
    return (np.sqrt(np.sum(rotate_to_frame(pos, center, evecs)[:, :2] ** 2, axis=1))
            if len(pos) else np.array([]))


def _halo_of(gal, cs):
    halo = getattr(gal, "halo", None)
    if halo is None:
        hi = getattr(gal, "parent_halo_index", -1)
        if hi is None or hi < 0:
            return None
        halo = cs.halos[hi]
    return halo


# ── per-snapshot reduction ──────────────────────────────────────────────────
def process_snapshot(sim, snap, entries, ism_bins, cgm_bins):
    iedges, _irmid, iarea, inb = ism_bins
    cedges, _crmid, carea, cnb = cgm_bins
    out = []
    cs = sim.load_catalog(snap=snap)
    with h5py.File(sim.get_snapshot_file(snap), "r") as f:
        a, hub = header_units(f)
        fld = _detect(f)
        g = f["PartType0"]
        s = f["PartType4"] if "PartType4" in f else None
        for (ri, si, gx) in entries:
            try:
                gal = cs.galaxies[gx]
            except (IndexError, KeyError):
                continue
            # stellar principal frame, shared by ISM and CGM projections
            sl = np.unique(np.asarray(getattr(gal, "slist", []), dtype=np.int64))
            if s is not None and len(sl):
                spos = _to_kpc(s["Coordinates"][sl], a, hub)
                smass = _to_msun(s["Masses"][sl], hub)
            else:
                spos, smass = np.empty((0, 3)), np.empty(0)
            gl = np.unique(np.asarray(gal.glist, dtype=np.int64))
            gpos = _to_kpc(g["Coordinates"][gl], a, hub) if len(gl) else np.empty((0, 3))
            center, evecs = _frame(spos, smass, gpos)
            if center is None:
                continue
            rec = dict(ri=ri, si=si)

            # ---- ISM: galaxy gas (gal.glist), 50 kpc disc profile ----
            if len(gl):
                R = _cylR(gpos, center, evecs)
                Rs = _cylR(spos, center, evecs)
                m_dust, m_HI, m_H2 = _components(
                    _to_msun(g["Masses"][gl], hub),
                    _to_msun(g[fld["dust"]][gl], hub) if fld["dust"] else None,
                    g[fld["Z"]][gl] if fld["Z"] else None,
                    g[fld["fneut"]][gl] if fld["fneut"] else None,
                    g[fld["fmol"]][gl] if fld["fmol"] else None)
                comps = {"dust": (R, m_dust), "HI": (R, m_HI), "H2": (R, m_H2), "star": (Rs, smass)}
                for c, (Ra, m) in comps.items():
                    rec[f"R20_{c}"] = _renc(Ra, m, 0.2)
                    rec[f"R50_{c}"] = _renc(Ra, m, 0.5)
                    rec[f"R80_{c}"] = _renc(Ra, m, 0.8)
                    rec[f"ism_sigma_{c}"] = _sig(Ra, m, iedges, iarea, inb)
                rec["R90_dust"] = _renc(R, m_dust, 0.9)

            # ---- CGM: parent-halo diffuse gas (halo.glist & SFR==0), 300 kpc ----
            halo = _halo_of(gal, cs)
            if halo is not None:
                hg = np.unique(np.asarray(getattr(halo, "glist", []), dtype=np.int64))
                if len(hg):
                    sfr = np.asarray(g[fld["sfr"]][hg]) if fld["sfr"] else np.zeros(len(hg))
                    idx = hg[np.asarray(sfr) == 0]               # non-star-forming = diffuse CGM
                    if len(idx):
                        cpos = _to_kpc(g["Coordinates"][idx], a, hub)
                        Rc = _cylR(cpos, center, evecs)
                        mgas = _to_msun(g["Masses"][idx], hub)
                        Zc = g[fld["Z"]][idx] if fld["Z"] else None
                        m_dc, m_HIc, m_H2c = _components(
                            mgas,
                            _to_msun(g[fld["dust"]][idx], hub) if fld["dust"] else None,
                            Zc,
                            g[fld["fneut"]][idx] if fld["fneut"] else None,
                            g[fld["fmol"]][idx] if fld["fmol"] else None)
                        rec["cgm_sigma_gas"] = _sig(Rc, mgas, cedges, carea, cnb)
                        rec["cgm_sigma_dust"] = _sig(Rc, m_dc, cedges, carea, cnb)
                        rec["cgm_sigma_HI"] = _sig(Rc, m_HIc, cedges, carea, cnb)
                        rec["cgm_sigma_H2"] = _sig(Rc, m_H2c, cedges, carea, cnb)
                        rec["cgm_R50_gas"] = _renc(Rc, mgas, 0.5)
                        rec["cgm_R50_dust"] = _renc(Rc, m_dc, 0.5)
                        rec["M_cgm"] = float(np.nansum(mgas))
                        rec["cgm_Ngas"] = float(len(Rc))
                        # temperature (per-particle), shared selection
                        XHc = _XH(Zc, len(idx))
                        if fld["Tdir"] is not None:
                            T = np.asarray(g[fld["Tdir"]][idx], dtype=np.float64)
                        elif fld["u"] is not None and fld["ne"] is not None:
                            T = _temperature(np.asarray(g[fld["u"]][idx], dtype=np.float64),
                                             np.asarray(g[fld["ne"]][idx], dtype=np.float64), XHc)
                        else:
                            T = np.full(len(idx), np.nan)
                        good = np.isfinite(T) & np.isfinite(mgas) & (mgas > 0) & (T > 0)
                        if np.any(good):
                            Tg, mg, Rg = T[good], mgas[good], Rc[good]
                            M = float(mg.sum())
                            w = mg / M
                            rec["T_mw"] = float(10.0 ** np.sum(w * np.log10(Tg)))
                            o = np.argsort(Tg)
                            cw = np.cumsum(mg[o])
                            rec["T_med"] = float(Tg[o][np.searchsorted(cw, 0.5 * cw[-1])])
                            rec["M_cgm_T"] = M
                            rec["f_hot"] = float(mg[Tg > T_HOT].sum() / M)
                            rec["f_cold"] = float(mg[Tg < T_COLD].sum() / M)
                            rec["f_warm"] = float(max(0.0, 1.0 - rec["f_hot"] - rec["f_cold"]))
                            rec["T_Ngas"] = float(len(Tg))
                            num, _ = np.histogram(Rg, bins=cedges, weights=mg * np.log10(Tg))
                            den, _ = np.histogram(Rg, bins=cedges, weights=mg)
                            with np.errstate(all="ignore"):
                                rec["cgm_Tprof"] = np.where(den > 0, 10.0 ** (num / den), np.nan)
            out.append(rec)
    del cs
    gc.collect()
    return out


# ── partial-file schema ─────────────────────────────────────────────────────
ISM_COMPS = ("dust", "HI", "H2", "star")
ISM_SCAL = [f"{q}_{c}" for c in ISM_COMPS for q in ("R20", "R50", "R80")] + ["R90_dust"]
ISM_SIG = [f"ism_sigma_{c}" for c in ISM_COMPS]
CGM_SCAL = ["cgm_R50_gas", "cgm_R50_dust", "M_cgm", "cgm_Ngas"]
CGM_SIG = ["cgm_sigma_gas", "cgm_sigma_dust", "cgm_sigma_HI", "cgm_sigma_H2"]
T_SCAL = ["T_mw", "T_med", "M_cgm_T", "f_hot", "f_warm", "f_cold", "T_Ngas"]
T_PROF = ["cgm_Tprof"]


def _scol(results, key):
    return np.array([r.get(key, np.nan) for r in results], np.float64) if results else np.empty(0)


def _pcol(results, key, nbins):
    if not results:
        return np.empty((0, nbins))
    return np.array([np.asarray(r.get(key, np.full(nbins, np.nan)), np.float64) for r in results])


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_task = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    with h5py.File(PLAN_PATH, "r") as f:
        sim_name = f.attrs["sim_name"]
        ism_rmax = float(f.attrs["rmax_kpc"])
        ism_nbins = int(f.attrs["nbins_r"])
        ri = f["entry_ri"][:]
        si = f["entry_si"][:]
        gx = f["entry_gx"][:]
        snap = f["entry_snap"][:]

    sim = Simulation(str(sim_name))
    ism_bins = _make_bins(ism_rmax, ism_nbins)
    cgm_bins = _make_bins(CGM_RMAX, CGM_NBINS)

    snaps_all = np.sort(np.unique(snap))[::-1]                 # newest first
    my_snaps = np.array_split(snaps_all, n_task)[task_id] if n_task > 1 else snaps_all
    print(f"[task {task_id}/{n_task}] {len(my_snaps)} snapshots; ISM rmax={ism_rmax:g}/{ism_nbins} "
          f"CGM rmax={CGM_RMAX:g}/{CGM_NBINS}: {list(map(int, my_snaps))}", flush=True)

    results = []
    for k, sn in enumerate(my_snaps):
        m = snap == sn
        entries = list(zip(ri[m].tolist(), si[m].tolist(), gx[m].tolist()))
        try:
            results += process_snapshot(sim, int(sn), entries, ism_bins, cgm_bins)
        except OSError as e:
            print(f"  [skip] snapshot {sn}: {e}", flush=True)
            continue
        print(f"  [task {task_id}] {k + 1}/{len(my_snaps)} snap {sn}: {len(entries)} entries", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    part = os.path.join(OUT_DIR, f"profiles_part_{task_id:03d}.hdf5")
    with h5py.File(part, "w") as out:
        out.attrs["ism_rmax_kpc"] = ism_rmax
        out.attrs["ism_nbins_r"] = ism_nbins
        out.attrs["cgm_rmax_kpc"] = CGM_RMAX
        out.attrs["cgm_nbins_r"] = CGM_NBINS
        out.create_dataset("ri", data=np.array([r["ri"] for r in results], np.int32))
        out.create_dataset("si", data=np.array([r["si"] for r in results], np.int32))
        for kk in ISM_SCAL + CGM_SCAL + T_SCAL:
            out.create_dataset(kk, data=_scol(results, kk))
        for kk in ISM_SIG:
            out.create_dataset(kk, data=_pcol(results, kk, ism_nbins), compression="gzip")
        for kk in CGM_SIG + T_PROF:
            out.create_dataset(kk, data=_pcol(results, kk, CGM_NBINS), compression="gzip")
    print(f"[task {task_id}] wrote {len(results)} entries -> {part}", flush=True)


if __name__ == "__main__":
    main()
