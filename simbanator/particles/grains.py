"""Unit conversion helpers for astrophysical quantities."""

import numpy as np


def Z_to_OH12(Z):
    """Convert metallicity *Z* (mass fraction) to 12 + log(O/H).

    Parameters
    ----------
    Z : float or array-like
        Metal mass fraction.

    Returns
    -------
    float or np.ndarray
        Oxygen abundance in solar-normalised logarithmic scale.
    """
    return np.log10(Z / 0.0127) + 8.69


def Dust_to_Metal(M_dust, M_h2, abundance):
    """Compute the dust-to-metal ratio following De Vis (2019).

    Parameters
    ----------
    M_dust : float or array-like
        Dust mass.
    M_h2 : float or array-like
        Molecular hydrogen mass.
    abundance : float or array-like
        12 + log(O/H) oxygen abundance.

    Returns
    -------
    float or np.ndarray
        Dust-to-metal ratio.
    """
    f_z = 27.36 * (10 ** (abundance - 12))
    M_z = f_z * M_h2 + M_dust
    return M_dust / M_z


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════
from .ks import _f64

# ── the mass budget and the radial flows of dust / H2 / gas in spherical shells out to the CGM (2026-09-04) ──────
BUDGET_ZONES = (("core", 0.0, 1.0), ("ring", 1.0, 3.162), ("outskirt", 3.162, 10.0),
                ("cgm_in", 10.0, 30.0), ("cgm_out", 30.0, 100.0))   # [proper kpc]; the reduced files reach rmax_kpc = 100

BUDGET_COLUMNS = [
    "zone", "r_in_kpc", "r_out_kpc", "has_vel", "n_gas", "n_member",
    "m_gas", "m_HI", "m_H2", "m_dust", "m_dust_member", "m_H2_member", "m_dust_hot",
    "vr_gas", "vr_H2", "vr_dust", "m_dust_out", "m_H2_out", "m_dust_in", "m_H2_in",
    "m_dust_esc", "m_H2_esc", "vesc_kms", "sigma_r_dust", "v_sys_n",
]

G_KPC_KMS2 = 4.30091e-6                            # G in kpc (km/s)^2 / Msun

def nfw_concentration(m200c_msun, z, h=0.68):
    """Dutton & Maccio 2014 c200c(M200c, z) (Planck cosmology): log c = a(z) + b(z) log(M200c / 1e12 h^-1 Msun),
    a = 0.520 + 0.385 exp(-0.617 z^1.21), b = -0.101 + 0.026 z."""
    m = np.asarray(m200c_msun, float); z = np.asarray(z, float)
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
    b = -0.101 + 0.026 * z
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10.0 ** (a + b * np.log10(m * h / 1e12))

def nfw_vesc_kms(r_kpc, m200c_msun, r200c_kpc, c):
    """Escape speed of an NFW halo at r [proper kpc]: v_esc^2 = 2 |Phi|, Phi(r) = -G M200 ln(1 + r/r_s) / (r f(c)),
    r_s = r200/c, f(c) = ln(1 + c) - c/(1 + c) (Phi -> 0 at infinity)."""
    r = np.maximum(np.asarray(r_kpc, float), 1e-3)
    rs = float(r200c_kpc) / float(c)
    fc = np.log1p(float(c)) - float(c) / (1.0 + float(c))
    phi = -G_KPC_KMS2 * float(m200c_msun) / (r * fc) * np.log1p(r / rs)
    return np.sqrt(2.0 * np.abs(phi))

def measure_zone_budget(red, zones=BUDGET_ZONES, t_hot=1e5, v_sys_kpc=10.0, hubble_kms_per_kpc=0.0,
                        m200c_msun=None, r200c_kpc=None, conc=None, z=None):
    """Dust / H2 / gas masses and their radial flows in spherical shells of one reduced file, EVERY particle.

    Returns one dict row per zone (BUDGET_COLUMNS). Geometry is spherical about the stored centre
    (``r = |pos|``). Unlike :func:`measure_ks` nothing is restricted to the CAESAR members — the
    question is where the dust and the molecular gas ARE, members or not — but the member masses are
    carried alongside (``m_dust_member``, ``m_H2_member``) so the ISM / CGM split can be read either
    way. Velocities: the systemic velocity is the m_star-weighted mean of the member stars inside
    ``v_sys_kpc`` (fallback: the member gas, then all gas there); ``vr_*`` are the mass-weighted mean
    radial velocities [km/s] of the zone's gas / H2 / dust about it (positive = outflow), with the
    Hubble flow ``hubble_kms_per_kpc * r`` added when given (peculiar velocities are stored);
    ``m_dust_out`` / ``m_H2_out`` = the mass carried by particles with v_r > 0 (outward), ``m_dust_in`` /
    ``m_H2_in`` the mass on v_r < 0 (inward); ``m_dust_esc`` / ``m_H2_esc`` = the mass on particles ABOVE THE
    LOCAL ESCAPE SPEED of the host halo (an NFW potential from ``m200c_msun`` / ``r200c_kpc`` [proper kpc], with
    ``conc`` or the Dutton & Maccio 2014 c(M, z) when ``z`` is given; NaN when the halo is not given) and
    ``vesc_kms`` that escape speed at the shell's mid radius; ``sigma_r_dust`` = the dust-weighted spread of
    v_r; ``m_dust_hot`` = the dust sitting in gas above ``t_hot`` K (where thermal sputtering acts). Without
    ``vel`` (files built before 2026-08-28) the flow columns are NaN and ``has_vel`` is False; the masses are
    always measured.
    """
    gas, star = red.get("gas", {}), red.get("star", {})
    has_vel = ("vel" in gas)
    gpos = np.asarray(gas.get("pos", np.zeros((0, 3))), float)
    n = len(gpos)
    m_gas = _f64(gas.get("m_gas"), n)
    m_H2 = _f64(gas.get("m_H2"), n)
    m_HI = _f64(gas.get("m_HI"), n)
    m_dust = _f64(gas.get("m_dust"), n)
    temp = _f64(gas.get("temp"), n) if "temp" in gas else np.full(n, np.nan)
    gmem = np.asarray(gas["member"], bool) if "member" in gas else np.zeros(n, bool)
    rg = np.sqrt(np.sum(gpos ** 2, axis=1)) if n else np.zeros(0)
    # the systemic velocity
    v_sys, v_sys_n = np.zeros(3), 0
    if has_vel:
        gvel = np.asarray(gas["vel"], float)
        spos = np.asarray(star.get("pos", np.zeros((0, 3))), float)
        if "vel" in star and len(spos):
            svel = np.asarray(star["vel"], float)
            ms = _f64(star.get("m_star"), len(spos))
            smem = np.asarray(star["member"], bool) if "member" in star else np.ones(len(spos), bool)
            rs = np.sqrt(np.sum(spos ** 2, axis=1))
            sel = smem & (rs <= v_sys_kpc) & np.isfinite(ms) & (ms > 0)
            if sel.sum() >= 10:
                v_sys, v_sys_n = np.average(svel[sel], axis=0, weights=ms[sel]), int(sel.sum())
        if v_sys_n == 0:
            for sel in (gmem & (rg <= v_sys_kpc), (rg <= v_sys_kpc)):
                sel = sel & np.isfinite(m_gas) & (m_gas > 0)
                if sel.sum() >= 10:
                    v_sys, v_sys_n = np.average(gvel[sel], axis=0, weights=m_gas[sel]), int(sel.sum())
                    break
        with np.errstate(invalid="ignore", divide="ignore"):
            rhat = gpos / np.where(rg > 0, rg, np.nan)[:, None]
            vr = np.sum((gvel - v_sys) * rhat, axis=1) + float(hubble_kms_per_kpc) * rg
    else:
        vr = np.full(n, np.nan)
    have_halo = (m200c_msun is not None and r200c_kpc is not None and np.isfinite(m200c_msun) and np.isfinite(r200c_kpc)
                 and m200c_msun > 0 and r200c_kpc > 0)
    if have_halo:
        c_ = float(conc) if (conc is not None and np.isfinite(conc)) else (float(nfw_concentration(m200c_msun, z)) if z is not None else np.nan)
        have_halo = np.isfinite(c_) and c_ > 0
    vesc = nfw_vesc_kms(rg, m200c_msun, r200c_kpc, c_) if (have_halo and n) else np.full(n, np.nan)

    def _wmean(v, w):
        w = np.where(np.isfinite(v) & np.isfinite(w) & (w > 0), w, 0.0)
        return float(np.sum(v * w) / np.sum(w)) if np.sum(w) > 0 else np.nan

    rows = []
    for label, r_in, r_out in zones:
        sel = (rg <= r_out) & ((rg > r_in) if r_in > 0 else True)
        row = dict(zone=str(label), r_in_kpc=float(r_in), r_out_kpc=float(r_out), has_vel=bool(has_vel),
                   n_gas=int(sel.sum()), n_member=int((sel & gmem).sum()),
                   m_gas=float(np.nansum(m_gas[sel])), m_HI=float(np.nansum(m_HI[sel])),
                   m_H2=float(np.nansum(m_H2[sel])), m_dust=float(np.nansum(m_dust[sel])),
                   m_dust_member=float(np.nansum(m_dust[sel & gmem])), m_H2_member=float(np.nansum(m_H2[sel & gmem])),
                   m_dust_hot=float(np.nansum(m_dust[sel & np.isfinite(temp) & (temp > t_hot)])),
                   vr_gas=np.nan, vr_H2=np.nan, vr_dust=np.nan, m_dust_out=np.nan, m_H2_out=np.nan,
                   m_dust_in=np.nan, m_H2_in=np.nan, m_dust_esc=np.nan, m_H2_esc=np.nan,
                   vesc_kms=float(nfw_vesc_kms(0.5 * (r_in + r_out), m200c_msun, r200c_kpc, c_)) if have_halo else np.nan,
                   sigma_r_dust=np.nan, v_sys_n=int(v_sys_n))
        if has_vel and sel.any():
            vz = vr[sel]
            row["vr_gas"] = _wmean(vz, m_gas[sel])
            row["vr_H2"] = _wmean(vz, m_H2[sel])
            row["vr_dust"] = _wmean(vz, m_dust[sel])
            out = sel & np.isfinite(vr) & (vr > 0)
            row["m_dust_out"] = float(np.nansum(m_dust[out]))
            row["m_H2_out"] = float(np.nansum(m_H2[out]))
            inn = sel & np.isfinite(vr) & (vr < 0)
            row["m_dust_in"] = float(np.nansum(m_dust[inn]))
            row["m_H2_in"] = float(np.nansum(m_H2[inn]))
            if have_halo:
                esc = sel & np.isfinite(vr) & (vr > vesc)
                row["m_dust_esc"] = float(np.nansum(m_dust[esc]))
                row["m_H2_esc"] = float(np.nansum(m_H2[esc]))
            if np.isfinite(row["vr_dust"]):
                row["sigma_r_dust"] = float(np.sqrt(_wmean((vz - row["vr_dust"]) ** 2, m_dust[sel])))
        rows.append(row)
    return rows



# ═════════════════════════════════════════════════════════════════════════════
# Gas phases, grain growth, dust budgets
# ═════════════════════════════════════════════════════════════════════════════

XH_CAESAR = 0.76            # caesar's constant hydrogen fraction (so particle sums match the catalogue)
NH_H2_MIN = 0.13            # n_H [cm^-3] below which caesar counts no H2


def gas_components(mgas_msun, dust_msun, fneut, fmol, nH=None, XH=XH_CAESAR, nh_h2_min=NH_H2_MIN):
    """Dust, HI and H2 masses [Msun] of gas particles with caesar's split (the 2026-08-27 recipe).

    ``H2 = XH m fH2`` (0 where ``n_H < nh_h2_min`` when ``nH`` [cm^-3] is given),
    ``HI = XH m min(f_neut, 1 - fH2)``. Unlike the older notebook recipe, the H2 is not multiplied by
    Grackle's neutral fraction (which is ~0 in star-forming gas and put particle H2 up to 30x below
    caesar). Returns ``(m_dust, m_HI, m_H2)``; m_dust NaN without dust, m_H2 NaN without fH2.
    """
    m_dust = dust_msun if dust_msun is not None else np.full_like(np.asarray(mgas_msun, float), np.nan)
    m_H = np.asarray(mgas_msun, np.float64) * XH
    fneut = np.asarray(fneut, np.float64) if fneut is not None else np.full_like(m_H, np.nan)
    if fmol is not None:
        fm = np.asarray(fmol, np.float64)
        if nH is not None:
            fm = np.where(np.asarray(nH, np.float64) >= nh_h2_min, fm, 0.0)
        m_H2 = m_H * fm
        m_HI = m_H * np.minimum(fneut, 1.0 - fm)
    else:
        m_H2 = np.full_like(m_H, np.nan)
        m_HI = m_H * fneut
    return m_dust, m_HI, m_H2


def growth_timescale(nh, T, Z, tau_ref_gyr=0.0254, n_ref=100.0, T_ref=20.0, a_um=0.1, a_ref_um=0.1, Z_sun=0.0134):
    """Grain-growth (metal accretion) time [Gyr]: ``tau_ref (a/a_ref) (n_H/n_ref)^-1 (T/T_ref)^-1/2 (Z/Z_sun)^-1``.

    Depends only on the gas state, not on dust already present. NaN where n_H, T or Z is not
    positive. The default normalisation is the one of dusty_qgs_kappa_selection Part 17:
    25.4 Myr at n_H = 100 cm^-3, T = 20 K, solar metallicity (Asplund+09) and a = 0.1 um.
    """
    nh, T, Z = (np.asarray(v, float) for v in (nh, T, Z))
    out = np.full(np.broadcast(nh, T, Z).shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        zrel = Z / Z_sun
        ok = np.isfinite(nh) & np.isfinite(T) & np.isfinite(zrel) & (nh > 0) & (T > 0) & (zrel > 0)
        nh_, T_, z_ = np.broadcast_arrays(nh, T, zrel)
        out[ok] = (a_um / a_ref_um) * tau_ref_gyr * (n_ref / nh_[ok]) * np.sqrt(T_ref / T_[ok]) / z_[ok]
    return out


def stellar_dust_budget(sfr, dt_gyr, dust_yield):
    """Largest dust mass stars and SNe can make in ``dt_gyr`` at constant SFR [Msun/yr], with no destruction."""
    return dust_yield * np.asarray(sfr, float) * dt_gyr * 1e9


def weighted_fraction(weight, valid, good, nmin=5):
    """Weighted fraction of the ``valid`` reservoir that is ``good`` (NaN below ``nmin`` valid particles or zero weight)."""
    weight = np.asarray(weight, float)
    valid = np.asarray(valid, bool) & np.isfinite(weight) & (weight > 0)
    if valid.sum() < nmin:
        return np.nan
    denom = np.sum(weight[valid])
    return float(np.sum(weight[valid & np.asarray(good, bool)]) / denom) if denom > 0 else np.nan


def number_fraction(valid, good, nmin=5):
    """Particle-number fraction of the ``valid`` set that is ``good`` (NaN below ``nmin``)."""
    valid = np.asarray(valid, bool)
    n = int(valid.sum())
    return float(np.sum(valid & np.asarray(good, bool)) / n) if n >= nmin else np.nan
