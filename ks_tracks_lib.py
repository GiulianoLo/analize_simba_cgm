"""ks_tracks_lib — pure numpy/h5py helpers for ks_tracks_quenched_m25.ipynb.

Kennicutt–Schmidt (KS) evolution tracks of the cis25 quenched sample: surface densities of
molecular gas and star formation measured from the CAESAR member particles (glist/slist) stored in
the reduced particle files written by ``build_reduced_particles_job.py``, at each galaxy's critical
epochs (sf_peak → sft → qt → post_quench → gas_min → anchor, plus ``end`` = the anchor or, when the
anchor holds no measurable H2, the last snapshot with M_H2/M_star > ``fh2_min``; plus the AGN-feedback
epochs ``agn_ign`` / ``jet_on`` attached from the m25 Part 2d windows table by ``attach_bh_stages``).

The module deliberately imports NOTHING from ``simbanator`` (its ``analysis`` package pulls in yt
through sfh_fsps and is not importable off the cluster), so every measurement here can be unit-tested
locally on synthetic reduced files (``tests/test_ks_tracks.py``). The notebook injects the two
cluster-only pieces it needs: the cosmology (as an ``a -> t`` callable) and
``simbanator.analysis.quenching.find_quenching_times``.

Conventions (all documented in the notebook's closing cell as well)
-------------------------------------------------------------------
* Face-on projection: ``principal_axes`` (simbanator.utils.geometry) returns the principal axes as the
  COLUMNS of ``evecs`` and ``rotate_to_frame`` uses ``pos @ evecs``; the reduced job stores that
  matrix verbatim, so the face-on cylindrical radius is ``hypot(*(pos @ evecs)[:, :2].T)``.
  (``pos @ evecs.T`` — used by two older notebooks — projects onto the wrong frame.)
* Simulated ``m_H2`` is hydrogen-only (``build_profiles_job._components``); the observed M_H2
  (alpha_CO = 4.36) includes helium, hence ``HE_FACTOR = 1.36`` on the simulated H2 for the comparison
  columns only (raw masses are kept in the tables).
* Half-mass apertures: the mean surface density inside the face-on half-mass radius,
  ``M(<R50)/(pi R50^2)``, is IDENTICALLY ``0.5 M_tot/(pi R50^2)`` — the observed convention — so no
  separate 0.5 factor is ever multiplied in. For the SFR the observed convention assumes the star
  formation follows the CO (``0.5 SFR_tot / (pi R_CO^2)``); ``ks_columns`` provides both that
  (``logSigmaSFR_obs``, used on the R50_H2 rows) and the literal SFR(<R)/area (``logSigmaSFR_inside``).
  Fixed apertures never carry a 0.5.
* SFR = 0 inside an aperture over a window is a censored value: the one-particle floor
  ``m_star_particle / window / area`` is stored as an upper limit (``is_ul``).
* Rotation support: ``kappa_rot`` is the Sales+12 K_rot/K estimator of the m25 notebook's Part 8j0
  (``_kin``), verbatim: velocities about the weighted mean of the subset (its bulk motion), spin
  axis = the subset's angular momentum, so a constant velocity unit cancels. ``measure_zone_kinematics``
  evaluates it for all gas / H2-weighted gas / stars in SPHERICAL zones about the stored centre
  (``KIN_ZONES`` = the m25 ladder's core ``ap3kpc`` (r < 3.16 kpc), outskirt shell ``ann10kpc``
  (3.16 < r < 10 kpc) and the ``ap10kpc`` rung) together with the mass-weighted stellar age of the
  same zone — the reduced files must carry ``vel`` (build_reduced_particles_job, 2026-08-28+;
  older files are backfilled by re-running the plan); without it the kappas are NaN, the ages not.
"""
import os

import numpy as np
import h5py

__all__ = [
    "STAGES_KS", "STAGES_PLOT", "FH2_MIN_END", "FIXED_AP_KPC", "FIXED_AP_LABELS", "ADAPTIVE_AP", "AP_LABELS",
    "HE_FACTOR", "RELATIONS", "tdep_ms_gyr", "relation_y",
    "reduced_path", "load_reduced", "face_on_R", "half_mass_radius", "make_a_to_t", "sfr_window",
    "measure_ks", "MEASURE_COLUMNS", "ks_columns", "build_stage_records", "stage_time_order",
    "STAGES_BH", "nearest_row", "attach_bh_stages",
    "interp_track", "grid_stats", "ecdf",
    "KIN_ZONES", "KIN_COMPONENTS", "ZONE_KIN_COLUMNS", "kappa_rot", "measure_zone_kinematics",
]

# ── stages / apertures ────────────────────────────────────────────────────────────────────────────
STAGES_BH = ["agn_ign", "jet_on"]   # AGN ignition / jet-mode onset (m25 Part 2d windows table; attach_bh_stages), 2026-08-28
STAGES_KS = ["sf_peak", "ssfr_min", "sft", "qt", "post_quench", "gas_min", "anchor", "end"] + STAGES_BH
STAGES_PLOT = ["sf_peak", "sft", "qt", "post_quench", "gas_min", "anchor"]   # time order (summary / t_dep clock)
FH2_MIN_END = 1e-4        # `end` stage: the anchor when M_H2/M_star (history) > this, else the last snapshot above it
FIXED_AP_KPC = (1.0, 3.162, 10.0)                 # the m25 ladder rungs (ap3kpc is really 3.16 kpc)
FIXED_AP_LABELS = ("ap1kpc", "ap3kpc", "ap10kpc")
ADAPTIVE_AP = ("R50_H2", "R50_star", "R50_SFR")   # face-on member half-mass radii
AP_LABELS = tuple(FIXED_AP_LABELS) + tuple(ADAPTIVE_AP)
HE_FACTOR = 1.36                                  # He correction on the hydrogen-only simulated H2

# ── reference relations (copied from pilot_specphot/scripts/plot_ks.py) ──────────────────────────
# log Sigma_SFR [Msun/yr/kpc^2] = A + N log Sigma_x [Msun/pc^2]; sig = published 1-sigma scatter [dex]
RELATIONS = {
    "K98": dict(A=float(np.log10(2.5e-4 * 0.63)), N=1.4, sig=0.30, color="#d95f02", ls="-",
                label="Kennicutt 98 (total gas, Chabrier)", scat_label="0.30 dex rms"),
    "B08": dict(A=-2.06 - 0.96 * 1.0, N=0.96, sig=0.20, color="#1b9e77", ls="--",
                label=r"Bigiel+08 ($\Sigma_{\rm H_2}$, 750 pc)", scat_label="0.20 dex rms"),
    "RK19": dict(A=-3.84, N=1.41, sig=0.28, color="#7570b3", ls="-.",
                 label="de los Reyes & Kennicutt 19 (spirals)",
                 scat_label=r"0.28 dex $\sigma_{\rm int}$"),
}


def relation_y(key, log_sigma_h2):
    """log Sigma_SFR predicted by relation `key` at log Sigma_H2 [Msun/pc^2]."""
    rel = RELATIONS[key]
    return rel["A"] + rel["N"] * np.asarray(log_sigma_h2, float)


def tdep_ms_gyr(z):
    """Tacconi+18 main-sequence depletion time: log t_dep [Gyr] = 0.09 - 0.62 log10(1+z)."""
    return 10.0 ** (0.09 - 0.62 * np.log10(1.0 + np.asarray(z, float)))


# ── reduced particle files ────────────────────────────────────────────────────────────────────────
def reduced_path(reduced_dir, prefix, snap, gx):
    """output/<sim>/reduced_particles/snap_NNN/<prefix>_snapNNN_galGGGGGG.h5"""
    return os.path.join(reduced_dir, "snap_%03d" % int(snap),
                        "%s_snap%03d_gal%06d.h5" % (prefix, int(snap), int(gx)))


def load_reduced(path, keys=None, bad=None):
    """Read one reduced file -> {attrs..., 'gas': {...}, 'star': {...}}; None if absent/corrupt.

    keys=None reads every dataset; keys={'gas': (...), 'star': (...)} reads only those. A corrupt
    (truncated) file is reported once into the optional `bad` list and skipped.
    """
    if not os.path.exists(path):
        return None
    try:
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.attrs:
                out[k] = f.attrs[k]
            for grp in ("gas", "star"):
                if grp in f:
                    if keys is None:
                        want = list(f[grp].keys())
                    else:
                        want = [k for k in keys.get(grp, ()) if k in f[grp]]
                    out[grp] = {k: f[grp][k][:] for k in want}
        return out
    except (OSError, RuntimeError) as e:            # inflate() failed etc.
        if bad is not None and path not in bad:
            bad.append(path)
        print("  [load_reduced] SKIP corrupt file %s: %s" % (os.path.basename(path), e))
        return None


def face_on_R(pos, evecs):
    """Face-on cylindrical radius [kpc] in the stored stellar principal frame.

    `evecs` columns are the principal axes (descending eigenvalue) exactly as
    simbanator.utils.geometry.principal_axes returns them; columns 0-1 span the disc plane.
    """
    pos = np.asarray(pos, float)
    if pos.size == 0:
        return np.zeros(0)
    proj = pos @ np.asarray(evecs, float)
    return np.hypot(proj[:, 0], proj[:, 1])


def half_mass_radius(R, w, n_min=1):
    """Radius enclosing half of sum(w), interpolated on the cumulative-mass curve.

    NaN when sum(w) <= 0 or fewer than `n_min` particles carry positive weight.
    """
    R = np.asarray(R, float)
    w = np.asarray(w, float)
    ok = np.isfinite(R) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < max(int(n_min), 1):
        return np.nan
    R, w = R[ok], w[ok]
    o = np.argsort(R)
    R, w = R[o], w[o]
    cum = np.cumsum(w)
    half = 0.5 * cum[-1]
    i = int(np.searchsorted(cum, half))            # first index with cum >= half
    if i == 0:
        return float(R[0])
    c0, c1 = cum[i - 1], cum[i]
    f = (half - c0) / (c1 - c0) if c1 > c0 else 0.0
    return float(R[i - 1] + f * (R[i] - R[i - 1]))


def make_a_to_t(cosmo, n=4096, a_min=0.02):
    """Callable a_form (scale factor) -> cosmic time [Gyr] on an interpolation grid
    (the Part-7e recipe of powderday_flux_quenched_m25: np.interp on 4096 nodes)."""
    a_grid = np.linspace(a_min, 1.0, int(n))
    t_grid = np.asarray(cosmo.age(1.0 / a_grid - 1.0).value, float)

    def a_to_t(a):
        a = np.asarray(a, float)
        return np.interp(np.clip(a, a_grid[0], 1.0), a_grid, t_grid)
    a_to_t.a_grid, a_to_t.t_grid = a_grid, t_grid
    return a_to_t


def sfr_window(tform_gyr, m_star, t_obs_gyr, window_myr):
    """Archaeological SFR [Msun/yr] = mass of stars formed within `window_myr` of t_obs / window.

    Current particle masses (no mass-loss correction; ~5-10 % over 100 Myr). Returns (sfr, n_young).
    """
    tform_gyr = np.asarray(tform_gyr, float)
    m_star = np.asarray(m_star, float)
    young = np.isfinite(tform_gyr) & (tform_gyr >= t_obs_gyr - window_myr / 1e3)
    return float(np.sum(m_star[young]) / (window_myr * 1e6)), int(young.sum())


MEASURE_COLUMNS = [
    "ap_label", "ap_kind", "ap_kpc", "area_kpc2",
    "n_gas", "n_H2", "n_star", "n_young100",
    "m_H2", "m_HI", "m_gas", "m_dust", "m_star", "sfr_inst", "sfr25", "sfr100",
    "m_H2_tot", "m_gas_tot", "m_star_tot", "sfr_inst_tot", "sfr25_tot", "sfr100_tot",
    "n_gas_tot", "n_H2_tot", "n_star_tot",
    "r50_H2", "r50_star", "r50_sfr", "r50_sfr_src", "m_star_particle", "has_tform",
]


def _f64(x, n=None):
    if x is None:
        return np.zeros(0 if n is None else n)
    return np.asarray(x, float)


def measure_ks(red, t_obs_gyr, a_to_t, fixed_kpc=FIXED_AP_KPC, fixed_labels=FIXED_AP_LABELS,
               member_only=True, ngas_min=10, nh2_min=5, nstar_min=10,
               sfr_windows=(25.0, 100.0)):
    """KS ingredients of one reduced file in fixed + adaptive face-on apertures.

    Returns a list of dict rows (one per aperture label in FIXED_AP_LABELS + ADAPTIVE_AP; MEASURE_COLUMNS).
    Masses in Msun, SFRs in Msun/yr, radii/areas in kpc. `member_only` restricts everything to the
    CAESAR member particles (glist/slist); gas-based quantities in an aperture are NaN when it holds
    fewer than `ngas_min` gas particles (counts are always kept). Adaptive radii: R50_H2 from the
    member H2 mass (NaN below `nh2_min` H2-bearing particles), R50_star from the member stars
    (`nstar_min`), R50_SFR from the stars formed in the last 100 Myr (falls back to the gas SFR
    weights when the file has no `tform`; recorded in r50_sfr_src).
    """
    evecs = np.asarray(red["evecs"], float)
    g = red.get("gas") or {}
    s = red.get("star") or {}

    # ── gas ──
    if "pos" in g and len(g["pos"]):
        Rg = face_on_R(g["pos"], evecs)
        ng = len(Rg)
        m_gas = _f64(g.get("m_gas"), ng)
        m_H2 = _f64(g.get("m_H2"), ng)
        m_HI = _f64(g.get("m_HI"), ng)
        m_dust = _f64(g.get("m_dust"), ng)
        sfr_g = _f64(g.get("sfr"), ng)
        if member_only and "member" in g:
            selg = np.asarray(g["member"], bool)
        else:
            selg = np.ones(ng, bool)
    else:
        Rg = np.zeros(0)
        m_gas = m_H2 = m_HI = m_dust = sfr_g = np.zeros(0)
        selg = np.zeros(0, bool)

    # ── stars ──
    if "pos" in s and len(s["pos"]):
        Rs = face_on_R(s["pos"], evecs)
        ns = len(Rs)
        m_star = _f64(s.get("m_star"), ns)
        if member_only and "member" in s:
            sels = np.asarray(s["member"], bool)
        else:
            sels = np.ones(ns, bool)
        tform = _f64(s.get("tform")) if ("tform" in s and len(s["tform"]) == ns) else None
        has_tform = tform is not None and bool(np.isfinite(tform).any())
        tform_gyr = a_to_t(tform) if has_tform else np.full(ns, np.nan)
    else:
        Rs = np.zeros(0)
        m_star = np.zeros(0)
        sels = np.zeros(0, bool)
        has_tform = False
        tform_gyr = np.zeros(0)

    # ── member totals + adaptive radii ──
    m_H2_tot = float(np.nansum(m_H2[selg]))
    m_gas_tot = float(np.nansum(m_gas[selg]))
    m_star_tot = float(np.nansum(m_star[sels]))
    sfr_inst_tot = float(np.nansum(sfr_g[selg]))
    if has_tform:
        sfr25_tot = sfr_window(tform_gyr[sels], m_star[sels], t_obs_gyr, sfr_windows[0])[0]
        sfr100_tot = sfr_window(tform_gyr[sels], m_star[sels], t_obs_gyr, sfr_windows[1])[0]
    else:
        sfr25_tot = sfr100_tot = np.nan
    n_gas_tot = int(selg.sum())
    n_H2_tot = int((selg & (m_H2 > 0)).sum())
    n_star_tot = int(sels.sum())
    r50_H2 = half_mass_radius(Rg[selg], m_H2[selg], nh2_min)
    r50_star = half_mass_radius(Rs[sels], m_star[sels], nstar_min)
    if has_tform:
        young = sels & np.isfinite(tform_gyr) & (tform_gyr >= t_obs_gyr - sfr_windows[1] / 1e3)
        r50_sfr = half_mass_radius(Rs[young], m_star[young], 1)
        r50_sfr_src = "stars"
    else:
        r50_sfr = half_mass_radius(Rg[selg], sfr_g[selg], 1)
        r50_sfr_src = "gas"
    pos_mass = m_star[sels & (m_star > 0)]
    m_star_particle = float(pos_mass.min()) if pos_mass.size else np.nan

    common = dict(m_H2_tot=m_H2_tot, m_gas_tot=m_gas_tot, m_star_tot=m_star_tot,
                  sfr_inst_tot=sfr_inst_tot, sfr25_tot=sfr25_tot, sfr100_tot=sfr100_tot,
                  n_gas_tot=n_gas_tot, n_H2_tot=n_H2_tot, n_star_tot=n_star_tot,
                  r50_H2=r50_H2, r50_star=r50_star, r50_sfr=r50_sfr, r50_sfr_src=r50_sfr_src,
                  m_star_particle=m_star_particle, has_tform=bool(has_tform))

    apertures = [(lab, "fixed", float(r)) for lab, r in zip(fixed_labels, fixed_kpc)]
    apertures += [("R50_H2", "R50_H2", r50_H2), ("R50_star", "R50_star", r50_star),
                  ("R50_SFR", "R50_SFR", r50_sfr)]

    rows = []
    for lab, kind, r in apertures:
        row = dict(ap_label=lab, ap_kind=kind, ap_kpc=float(r) if np.isfinite(r) else np.nan,
                   area_kpc2=float(np.pi * r * r) if np.isfinite(r) else np.nan,
                   n_gas=0, n_H2=0, n_star=0, n_young100=0,
                   m_H2=np.nan, m_HI=np.nan, m_gas=np.nan, m_dust=np.nan, m_star=np.nan,
                   sfr_inst=np.nan, sfr25=np.nan, sfr100=np.nan)
        row.update(common)
        if not np.isfinite(r):
            rows.append(row)
            continue
        ing = selg & (Rg <= r)
        ins = sels & (Rs <= r)
        row["n_gas"] = int(ing.sum())
        row["n_H2"] = int((ing & (m_H2 > 0)).sum())
        row["n_star"] = int(ins.sum())
        if row["n_gas"] >= int(ngas_min):
            row["m_H2"] = float(np.nansum(m_H2[ing]))
            row["m_HI"] = float(np.nansum(m_HI[ing]))
            row["m_gas"] = float(np.nansum(m_gas[ing]))
            row["m_dust"] = float(np.nansum(m_dust[ing]))
            row["sfr_inst"] = float(np.nansum(sfr_g[ing]))
        row["m_star"] = float(np.nansum(m_star[ins]))
        if has_tform:
            row["sfr25"], _ = sfr_window(tform_gyr[ins], m_star[ins], t_obs_gyr, sfr_windows[0])
            row["sfr100"], row["n_young100"] = sfr_window(tform_gyr[ins], m_star[ins],
                                                          t_obs_gyr, sfr_windows[1])
        rows.append(row)
    return rows


def ks_columns(tab, sfr_key="sfr100", sfr_tot_key="sfr100_tot", window_myr=100.0,
               he_factor=HE_FACTOR, suffix=""):
    """Observationally scaled KS columns from a measurement table (dict-like of arrays).

    Returns a dict of new arrays (append them to the table):
      logSigmaH2        [Msun/pc^2]  log10(he_factor * m_H2 / area / 1e6)
      logSigmaSFR_inside[Msun/yr/kpc^2] log10(SFR(<R) / area) — literal aperture value
      logSigmaSFR_obs   same, but on R50_H2 rows 0.5*SFR_tot/area (observed 'SFR follows CO' convention)
      is_ul             SFR == 0 inside the aperture -> censored (one-particle floor below)
      logSigmaSFR_ul    log10(m_star_particle / (window_myr*1e6) / area)  (NaN unless is_ul)
      logSigmaSFR       fiducial: logSigmaSFR_obs where finite, the floor where is_ul
      tdep_gyr          10**(logSigmaH2 + 6 - logSigmaSFR) / 1e9 (NaN where censored)
    `suffix` is appended to every key (e.g. '_inst' for the instantaneous gas SFR, window ignored).
    """
    area = np.asarray(tab["area_kpc2"], float)
    m_H2 = np.asarray(tab["m_H2"], float)
    sfr = np.asarray(tab[sfr_key], float)
    sfr_tot = np.asarray(tab[sfr_tot_key], float)
    kind = np.char.strip(np.asarray(tab["ap_kind"]).astype(str))
    mp = np.asarray(tab["m_star_particle"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok_area = np.isfinite(area) & (area > 0)
        lsh2 = np.where(ok_area & (m_H2 > 0), np.log10(he_factor * m_H2 / area / 1e6), np.nan)
        ls_in = np.where(ok_area & (sfr > 0), np.log10(sfr / area), np.nan)
        ls_obs = np.where((kind == "R50_H2") & ok_area & (sfr_tot > 0),
                          np.log10(0.5 * sfr_tot / area), ls_in)
        is_ul = ok_area & np.isfinite(sfr) & (sfr <= 0)
        if kind.size:
            is_ul = np.where(kind == "R50_H2", ok_area & np.isfinite(sfr_tot) & (sfr_tot <= 0), is_ul)
        floor = np.where(is_ul & np.isfinite(mp), np.log10(mp / (window_myr * 1e6) / area), np.nan)
        ls_fid = np.where(np.isfinite(ls_obs), ls_obs, floor)
        tdep = np.where(np.isfinite(lsh2) & np.isfinite(ls_obs),
                        10.0 ** (lsh2 + 6.0 - ls_obs) / 1e9, np.nan)
    out = {
        "logSigmaH2": lsh2, "logSigmaSFR_inside": ls_in, "logSigmaSFR_obs": ls_obs,
        "is_ul": is_ul, "logSigmaSFR_ul": floor, "logSigmaSFR": ls_fid, "tdep_gyr": tdep,
    }
    if suffix:
        out = {k + suffix: v for k, v in out.items()}
        out["logSigmaH2"] = lsh2                    # Sigma_H2 does not depend on the SFR choice
    return out



# ── rotation support in spherical zones (kappa_rot of gas / H2 / stars + stellar age per zone) ───
KIN_ZONES = (("ap3kpc", 0.0, 3.162), ("ann10kpc", 3.162, 10.0), ("ap10kpc", 0.0, 10.0))
KIN_COMPONENTS = ("gas", "H2", "star")
ZONE_KIN_COLUMNS = [
    "zone", "r_in_kpc", "r_out_kpc",
    "kappa_gas", "n_gas", "kappa_H2", "n_H2", "kappa_star", "n_star", "cos_gas_star", "cos_H2_star",
    "m_gas", "m_H2", "m_star", "n_star_age", "age_mw_gyr", "has_vel",
]


def kappa_rot(r, v, w, n_min=10):
    """Sales+12 kappa_rot = K_rot / K of a weighted particle subset -> (kappa, unit spin axis, n_used).

    r: positions about the centre [kpc]; v: velocities (any constant unit — the weighted mean of the
    subset is subtracted, so only the internal motions count and the unit cancels in the ratio);
    w: weights (only finite w > 0 count). NaN (axis None) below `n_min` usable particles or without a
    net angular momentum. Verbatim the m25 notebook's Part 8j0 estimator (`_kin`).
    """
    r, v, w = np.asarray(r, float), np.asarray(v, float), np.asarray(w, float)
    if r.ndim != 2 or v.shape != r.shape or len(w) != len(r):
        return np.nan, None, 0
    ok = np.isfinite(w) & (w > 0) & np.isfinite(r).all(axis=1) & np.isfinite(v).all(axis=1)
    n = int(ok.sum())
    if n < max(int(n_min), 1):
        return np.nan, None, n
    r, v, w = r[ok], v[ok], w[ok]
    v = v - np.average(v, axis=0, weights=w)              # bulk motion of the subset
    j = np.cross(r, v)
    L = np.sum(w[:, None] * j, axis=0)
    Ln = float(np.linalg.norm(L))
    if not (np.isfinite(Ln) and Ln > 0):
        return np.nan, None, n
    zh = L / Ln
    jz = j @ zh
    Rc = np.sqrt(np.maximum(np.sum(r ** 2, axis=1) - (r @ zh) ** 2, 0.0))
    okR = Rc > 1e-3
    K_rot = 0.5 * np.sum(w[okR] * (jz[okR] / Rc[okR]) ** 2)
    K_tot = 0.5 * np.sum(w * np.sum(v ** 2, axis=1))
    return (float(K_rot / K_tot) if K_tot > 0 else np.nan), zh, n


def measure_zone_kinematics(red, t_obs_gyr, a_to_t, zones=KIN_ZONES, member_only=False,
                            nkin_min=10, nstar_min=20):
    """kappa_rot of all gas / H2-weighted gas / stars and the stellar age in spherical zones of one reduced file.

    Returns one dict row per zone (ZONE_KIN_COLUMNS). Geometry is spherical about the stored centre
    (``r = |pos|``; the m25 Part 8j0 convention — a 3-D quantity, sightline-independent), so ``zone``
    labels are those of the m25 ladder rungs / shells. Per component: ``kappa_<c>`` and ``n_<c>`` = the
    number of weighted particles in the zone (counted even when the file has no ``vel``, so the consumer
    can tell "too few particles" from "no velocities": ``has_vel``). ``cos_*`` = alignment of the spin
    axes. ``age_mw_gyr`` = m_star-weighted mean of ``t_obs - t(tform)`` over the zone's stars (NaN below
    ``nstar_min`` stars with a formation epoch). ``member_only`` restricts every set to the CAESAR
    members (the KS-plane measurements' convention); the default takes every particle in the zone
    (the Part 8j0 convention, which the anchor rows reproduce).
    """
    gas, star = red.get("gas", {}), red.get("star", {})
    has_vel = ("vel" in gas) and ("vel" in star)

    def _set(grp, mass_key):
        pos = np.asarray(grp.get("pos", np.zeros((0, 3))), float)
        n = len(pos)
        keep = np.asarray(grp["member"], bool) if (member_only and "member" in grp) else np.ones(n, bool)
        pos = pos[keep]
        vel = np.asarray(grp["vel"], float)[keep] if "vel" in grp else np.full((len(pos), 3), np.nan)
        m = np.asarray(grp[mass_key], float)[keep] if mass_key in grp else np.zeros(len(pos))
        return pos, vel, m, keep

    gpos, gvel, m_gas, gkeep = _set(gas, "m_gas")
    m_H2 = np.asarray(gas["m_H2"], float)[gkeep] if "m_H2" in gas else np.zeros(len(gpos))
    spos, svel, m_star, skeep = _set(star, "m_star")
    tform = np.asarray(star["tform"], float)[skeep] if "tform" in star else np.full(len(spos), np.nan)
    with np.errstate(invalid="ignore"):
        t_form = a_to_t(tform) if len(tform) else np.zeros(0)
    age = np.where(np.isfinite(t_form), np.maximum(float(t_obs_gyr) - t_form, 0.0), np.nan)
    rg = np.sqrt(np.sum(gpos ** 2, axis=1)) if len(gpos) else np.zeros(0)
    rs = np.sqrt(np.sum(spos ** 2, axis=1)) if len(spos) else np.zeros(0)

    rows = []
    for label, r_in, r_out in zones:
        ing = (rg <= r_out) & ((rg > r_in) if r_in > 0 else True)
        ins = (rs <= r_out) & ((rs > r_in) if r_in > 0 else True)
        row = dict(zone=str(label), r_in_kpc=float(r_in), r_out_kpc=float(r_out), has_vel=bool(has_vel))
        axes = {}
        for comp, pos, vel, w, sel in (("gas", gpos, gvel, m_gas, ing), ("H2", gpos, gvel, m_H2, ing),
                                       ("star", spos, svel, m_star, ins)):
            ww = w[sel]
            n_w = int(np.sum(np.isfinite(ww) & (ww > 0)))
            if has_vel:
                kap, ax, n_w = kappa_rot(pos[sel], vel[sel], ww, n_min=nkin_min)
            else:
                kap, ax = np.nan, None
            row["kappa_%s" % comp] = kap
            row["n_%s" % comp] = int(n_w)
            axes[comp] = ax
        for a, b in (("gas", "star"), ("H2", "star")):
            row["cos_%s_%s" % (a, b)] = (float(axes[a] @ axes[b])
                                         if (axes.get(a) is not None and axes.get(b) is not None) else np.nan)
        row["m_gas"] = float(np.nansum(m_gas[ing]))
        row["m_H2"] = float(np.nansum(m_H2[ing]))
        row["m_star"] = float(np.nansum(m_star[ins]))
        oka = ins & np.isfinite(age) & np.isfinite(m_star) & (m_star > 0)
        row["n_star_age"] = int(oka.sum())
        row["age_mw_gyr"] = (float(np.average(age[oka], weights=m_star[oka]))
                             if oka.sum() >= max(int(nstar_min), 1) else np.nan)
        rows.append(row)
    return rows


# ── critical epochs (port of quench_mode_vs_sigma_gas build_records, trimmed to the KS stages) ────
def _smooth(y, w=3):
    n = len(y)
    h = w // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = y[max(0, i - h):min(n, i + h + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _ordered_positive(q, valid, t):
    d = np.where(valid, q, np.nan).astype(float)
    pos = np.isfinite(d) & (d > 0)
    if pos.sum() < 2:
        return None, None
    order = np.where(pos)[0]
    order = order[np.argsort(t[order])]
    return order, _smooth(d[order])


def _peak_time(q, valid, t):
    order, ds = _ordered_positive(q, valid, t)
    if order is None or not np.isfinite(ds).any():
        return np.nan
    return t[order[np.nanargmax(ds)]]


def _trough_before_floor(q, valid, t, floor_frac=1e-6):
    order, ds = _ordered_positive(q, valid, t)
    if order is None or not np.isfinite(ds).any():
        return np.nan
    floor = floor_frac * np.nanmax(ds)
    below = np.isfinite(ds) & (ds <= floor)
    cut = int(np.argmax(below)) if below.any() else len(ds)
    cut = max(cut, 2)
    seg = ds[:cut]
    if not np.isfinite(seg).any():
        return np.nan
    return t[order[np.nanargmin(seg)]]


def build_stage_records(P, t_cosmic_yr, redshift, galaxy_ids, cols, find_quenching_times,
                        age_of_z_gyr=None, stages=STAGES_KS, min_valid=5, gas_key="masses.H2",
                        fh2_min=FH2_MIN_END):
    """Critical epochs of every selected history column (quench_mode_vs_sigma_gas §3, KS stages).

    P : {property: (n_snap, n_gal) array} with row 0 = anchor; t_cosmic_yr / redshift per row;
    cols : column indices of the sample galaxies; find_quenching_times : the simbanator finder
    (injected); age_of_z_gyr(z) -> t_H [Gyr] (for tau_q / t_H; optional).
    Every galaxy gets a record (unlike the original, which skipped event-less galaxies): stages
    without a definition carry t_<stage>=NaN and row_<stage>=-1. The LAST quench event is used
    (k = argmax(qts)), the same rule as the m25 selection table.

    Departures from the residual_dust/quench_mode definitions, specific to a KS track:
    * ``gas_min`` is the trough (before any floor) of ``gas_key`` — the H2 mass when the history has
      it, else ``masses.gas`` — searched only AFTER t_QT (after the sSFR peak for event-less
      galaxies), i.e. the post-quench molecular depletion, not the formation-epoch minimum.
    * a stage whose time lies beyond the anchor (``post_quench`` = QT + persistence often does)
      keeps its t_<stage> but gets row_<stage> = -1: it is not observable inside the history and
      must not alias onto the anchor row.
    * ``end`` (the drawn track endpoint) is the anchor row when the anchor's catalogue molecular
      fraction ``masses.H2 / masses.stellar`` exceeds ``fh2_min``; otherwise the LATEST history row
      whose fraction does (``end_is_anchor`` False, ``t_end`` / ``row_end`` of that snapshot; -1 when no
      row qualifies). Without an H2 column in ``P`` the end is always the anchor.
    """
    t_cosmic_yr = np.asarray(t_cosmic_yr, float)
    redshift = np.asarray(redshift, float)
    galaxy_ids = np.asarray(galaxy_ids)
    cols = np.asarray(cols, int)

    def _nearest_row(t_target):
        return nearest_row(t_cosmic_yr, t_target)

    records = []
    for col in cols:
        gid = galaxy_ids[col]
        mstar = np.asarray(P["masses.stellar"][:, col], float)
        sfr = np.asarray(P["sfr"][:, col], float)
        gas = np.asarray(P["masses.gas"][:, col], float)
        with np.errstate(all="ignore"):
            ssfr = np.where(mstar > 0, sfr / mstar, np.nan)
        valid = np.isfinite(ssfr) & (ssfr > 0) & np.isfinite(t_cosmic_yr)
        rec = dict(gid=int(gid), col=int(col), n_valid=int(valid.sum()), n_events=0,
                   t_sft=np.nan, t_qt=np.nan, t_post_quench=np.nan, tau_q=np.nan,
                   tau_q_over_tH=np.nan, z_qt=np.nan, t_qt_first=np.nan, t_sft_first=np.nan)
        if valid.sum() >= min_valid:
            t = t_cosmic_yr[valid]
            sv = ssfr[valid]
            o = np.argsort(t)
            t, sv = t[o], sv[o]
            tu, ui = np.unique(t, return_index=True)
            su = sv[ui]
            if len(tu) >= min_valid:
                qts, sfts, _, dbg = find_quenching_times(
                    tu, su, galaxy_id=int(gid), plot=False, save_fits_path=None, return_debug=True)
                qts = np.asarray(qts, float)
                sfts = np.asarray(sfts, float)
                rec["n_events"] = int(len(qts))
                if len(qts):
                    k = int(np.argmax(qts))
                    rec["t_qt"], rec["t_sft"] = float(qts[k]), float(sfts[k])
                    pe = dbg.get("persistence_end_times") if isinstance(dbg, dict) else None
                    if pe is not None and len(pe) > k and np.isfinite(pe[k]):
                        rec["t_post_quench"] = float(pe[k])
                    kf = int(np.argmin(qts))
                    rec["t_qt_first"], rec["t_sft_first"] = float(qts[kf]), float(sfts[kf])
                    rec["tau_q"] = rec["t_qt"] - rec["t_sft"]
                    rec["z_qt"] = float(np.interp(rec["t_qt"], t_cosmic_yr[::-1], redshift[::-1]))
                    if age_of_z_gyr is not None:
                        rec["tau_q_over_tH"] = rec["tau_q"] / (float(age_of_z_gyr(rec["z_qt"])) * 1e9)
        t_sfpeak = _peak_time(ssfr, valid, t_cosmic_yr)
        gq = np.asarray(P[gas_key][:, col], float) if gas_key in P else gas
        t_start = rec["t_qt"] if np.isfinite(rec["t_qt"]) else t_sfpeak
        after = valid & (t_cosmic_yr >= t_start) if np.isfinite(t_start) else valid
        t_gasmin = _trough_before_floor(gq, after, t_cosmic_yr)
        if not np.isfinite(t_gasmin):                      # e.g. gas already at the floor: fall back
            t_gasmin = _trough_before_floor(gas, after, t_cosmic_yr)
        t_times = {
            "sf_peak": t_sfpeak,
            "ssfr_min": _trough_before_floor(ssfr, valid, t_cosmic_yr),
            "sft": rec["t_sft"], "qt": rec["t_qt"], "post_quench": rec["t_post_quench"],
            "gas_min": t_gasmin,
            "anchor": float(t_cosmic_yr[0]),
        }
        t_anchor = float(t_cosmic_yr[0])
        # end = anchor, or the last snapshot with a measurable H2 fraction (history level)
        with np.errstate(all="ignore"):
            fh2 = np.where(mstar > 0, np.asarray(P[gas_key][:, col], float) / mstar, np.nan) \
                if gas_key in P else np.full(len(mstar), np.nan)
        rec["fh2_anchor"] = float(fh2[0]) if np.isfinite(fh2[0]) else np.nan
        if gas_key not in P or (np.isfinite(fh2[0]) and fh2[0] > fh2_min):
            row_end, end_is_anchor = 0, True
        else:
            ok_h2 = np.isfinite(fh2) & (fh2 > fh2_min) & np.isfinite(t_cosmic_yr)
            row_end = int(np.where(ok_h2)[0][np.argmax(t_cosmic_yr[ok_h2])]) if ok_h2.any() else -1
            end_is_anchor = False
        rec["end_is_anchor"] = bool(end_is_anchor)
        t_times["end"] = float(t_cosmic_yr[row_end]) if row_end >= 0 else np.nan
        for st in stages:
            tt = t_times.get(st, np.nan)
            rec["t_%s" % st] = float(tt) if np.isfinite(tt) else np.nan
            if st == "anchor":
                rec["row_%s" % st] = 0
            elif st == "end":
                rec["row_%s" % st] = int(row_end)
            elif np.isfinite(tt) and tt > t_anchor + 1.0:      # beyond the history (1 yr tolerance)
                rec["row_%s" % st] = -1
            else:
                rec["row_%s" % st] = _nearest_row(tt)
        records.append(rec)
    return records


def stage_time_order(rec, stages=STAGES_PLOT):
    """Stages of one record that are defined, sorted by their cosmic time (for drawing a track)."""
    have = [(rec["t_%s" % st], st) for st in stages
            if np.isfinite(rec.get("t_%s" % st, np.nan)) and rec.get("row_%s" % st, -1) >= 0]
    return [st for _, st in sorted(have, key=lambda x: x[0])]


def nearest_row(t_cosmic_yr, t_target, t_anchor=None, tol_yr=1.0):
    """Index of the history row nearest to `t_target` [yr]; -1 when the time is not finite or (with `t_anchor`)
    lies more than `tol_yr` beyond the anchor, i.e. outside the history."""
    if not np.isfinite(t_target):
        return -1
    if t_anchor is not None and t_target > float(t_anchor) + tol_yr:
        return -1
    return int(np.argmin(np.abs(np.asarray(t_cosmic_yr, float) - float(t_target))))


def attach_bh_stages(recs, t_cosmic_yr, dt_sft_gyr, stages=STAGES_BH):
    """Add the AGN-feedback epochs to the stage records of `build_stage_records` (in place; returns `recs`).

    dt_sft_gyr : {gid: {stage: dt}} — the epoch of each stage in Gyr RELATIVE TO SFT, the convention of the m25
    notebook's Part 2d windows table (`agn_classifier_windows.fits`: t_agn = ignition, first crossing of half the
    pre-QT peak BHAR; t_jet = first snapshot with the ungated w_jet >= 0.5; both (t - t_SFT)/Gyr).
    A stage gets t_<stage> = t_sft + dt [yr] and row_<stage> = the nearest history row (-1 beyond the anchor,
    row 0 = anchor); it stays undefined (NaN / -1) when the galaxy has no SFT, no entry, or a NaN dt.
    """
    t_cosmic_yr = np.asarray(t_cosmic_yr, float)
    t_anchor = float(t_cosmic_yr[0])
    for rec in recs:
        d = dt_sft_gyr.get(int(rec["gid"]), {}) or {}
        t_sft = float(rec.get("t_sft", np.nan))
        for st in stages:
            dt = d.get(st, np.nan)
            dt = float(dt) if dt is not None else np.nan
            t = t_sft + dt * 1e9 if (np.isfinite(t_sft) and np.isfinite(dt)) else np.nan
            rec["t_%s" % st] = float(t) if np.isfinite(t) else np.nan
            rec["row_%s" % st] = nearest_row(t_cosmic_yr, t, t_anchor=t_anchor)
    return recs


# ── histories on a common clock (notebook Part 5: evolutionary properties of the KS regions) ─────
def interp_track(t, y, grid):
    """One galaxy's y(t) linearly interpolated onto `grid`; NaN outside the coverage of its finite samples
    (no extrapolation), non-finite samples dropped, duplicate times collapsed; all-NaN with < 2 samples."""
    t, y, grid = np.asarray(t, float), np.asarray(y, float), np.asarray(grid, float)
    out = np.full(len(grid), np.nan)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return out
    tt, ui = np.unique(t[ok], return_index=True)
    yy = y[ok][ui]
    if len(tt) < 2:
        return out
    inside = (grid >= tt[0]) & (grid <= tt[-1])
    out[inside] = np.interp(grid[inside], tt, yy)
    return out


def grid_stats(tracks, nmin=5):
    """tracks: (n_gal, n_grid) array, NaN = no coverage -> dict(n, med, p16, p84) per grid point; the statistics
    are NaN wherever fewer than `nmin` galaxies contribute (n is always the true count)."""
    tr = np.atleast_2d(np.asarray(tracks, float))
    fin = np.isfinite(tr)
    n = fin.sum(axis=0)
    med, p16, p84 = (np.full(tr.shape[1], np.nan) for _ in range(3))
    for j in range(tr.shape[1]):
        if n[j] >= nmin:
            v = tr[fin[:, j], j]
            med[j], p16[j], p84[j] = np.median(v), np.percentile(v, 16), np.percentile(v, 84)
    return dict(n=n, med=med, p16=p16, p84=p84)


def ecdf(x):
    """(sorted finite values, empirical CDF 1/n ... 1); empty arrays when nothing is finite."""
    v = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])
    return v, (np.arange(1, len(v) + 1) / len(v) if len(v) else np.zeros(0))
