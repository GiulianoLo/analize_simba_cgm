"""Rotation support of particle subsets: kappa_rot (Sales+12) and zone kinematics (moved from ks_tracks_lib.py)."""
import os

import numpy as np
import h5py


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════

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
