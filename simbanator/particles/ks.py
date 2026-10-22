"""The Kennicutt-Schmidt measurements of the reduced particle files (moved from ks_tracks_lib.py).

See ks_tracks_lib.py for the conventions (face-on frame, He factor, half-mass apertures, censored SFRs).
"""
import os

import numpy as np
import h5py


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════
from .frames import face_on_R
from .profiles import half_mass_radius
from .cutouts import load_reduced

FIXED_AP_KPC = (1.0, 3.162, 10.0)                 # the m25 ladder rungs (ap3kpc is really 3.16 kpc)

FIXED_AP_LABELS = ("ap1kpc", "ap3kpc", "ap10kpc")

ADAPTIVE_AP = ("R50_H2", "R50_star", "R50_SFR")   # face-on member half-mass radii

AP_LABELS = tuple(FIXED_AP_LABELS) + tuple(ADAPTIVE_AP)

HE_FACTOR = 1.36                                  # He correction on the hydrogen-only simulated H2

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


# ═════════════════════════════════════════════════════════════════════════════
# Regions on the K-S plane, track selection, file loops, the SF-law depletion time
# ═════════════════════════════════════════════════════════════════════════════

def ks_region(x, y, upper_limit, ref="B08", band=None):
    """Where points sit relative to a K-S relation: ``'below'`` / ``'within'`` / ``'above'`` / ``'undef'``.

    ``ref`` is a key of :data:`simbanator.obs.RELATIONS`; ``band`` is the half-width [dex]
    (default: that relation's published scatter). A censored point (``upper_limit``: y is the
    one-particle floor) is ``'below'`` only when its floor is already below the band, otherwise
    ``'undef'``. Returns ``(regions, Δ = y - relation(x))``.
    """
    from ..obs import relation, relation_y
    band = float(relation(ref)["sig"]) if band is None else float(band)
    x, y, ul = np.asarray(x, float), np.asarray(y, float), np.asarray(upper_limit, bool)
    with np.errstate(invalid="ignore"):
        off = y - relation_y(ref, x)
    fin = np.isfinite(off)
    reg = np.full(len(x), "undef", dtype=object)
    reg[fin & ~ul & (off < -band)] = "below"
    reg[fin & ~ul & (np.abs(off) <= band)] = "within"
    reg[fin & ~ul & (off > band)] = "above"
    reg[fin & ul & (off < -band)] = "below"
    return reg.astype(str), off


def select_tracks(tracks, aperture, population="Q"):
    """Rows of a ks_tracks table for one aperture label and population (``Q`` quenched, ``SF`` controls)."""
    from ..io.tables import as_str
    return tracks[(as_str(tracks["ap_label"]) == aperture) & (as_str(tracks["pop"]) == population)]


def measure_files(pairs, path_of, measure, keys=None, progress_every=200):
    """Run ``measure(reduced dict, (snap, gx)) -> list of row dicts`` over the reduced files of ``pairs``.

    ``path_of(snap, gx)`` gives the file. Each row gets ``snap`` and ``gx``. Returns
    ``(rows, n_missing, unreadable paths)``.
    """
    rows, bad = [], []
    n_missing = 0
    for k, (snap, gx) in enumerate(pairs):
        red = load_reduced(path_of(snap, gx), keys=keys, bad=bad)
        if red is None:
            n_missing += 1
            continue
        for r in measure(red, (snap, gx)):
            r.update(snap=int(snap), gx=int(gx))
            rows.append(r)
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  {k + 1}/{len(pairs)} files")
    return rows, n_missing, bad


def sf_law_depletion_time_gyr(n_h, eps_ff=0.02):
    """``t_ff / eps_ff`` [Gyr] at hydrogen density ``n_h`` [cm^-3] (ρ = n_H m_p / X_H, X_H = 0.76); SIMBA uses eps_ff = 0.02."""
    rho = np.asarray(n_h, float) * 1.6726e-24 / 0.76
    return np.sqrt(3 * np.pi / (32 * 6.674e-8 * rho)) / 3.156e7 / eps_ff / 1e9
