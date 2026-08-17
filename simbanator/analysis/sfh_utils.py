"""Star-formation-history track utilities.

SIMBA's per-snapshot ``sfr`` is the instantaneous gas SFR sampled at the
snapshot cadence (~100-300 Myr), so the tracks carry burst-to-burst scatter
that smooth parametric SFH forms (and SED-derived SFR estimates) cannot and
should not follow. These helpers turn such a track into a smooth, uniformly
sampled SFH suitable for fitting and for truth comparisons.

The module also holds the particle-archaeology builders promoted from the
powderday mock-observation notebooks: an FSPS surviving-mass-fraction lookup
(powderday-consistent formed-mass correction) and fixed-grid archaeological
SFHs of projected radial regions, per sightline.
"""

import numpy as np

from ..utils.geometry import projected_radius

__all__ = ["smooth_resample_sfh", "recent_sfr",
           "sfr_delayed_bq", "fit_delayed_bq",
           "build_mfrac_lookup", "mfrac_of",
           "archaeological_sfh", "projected_region_sfh"]


def smooth_resample_sfh(t_gyr, sfr, dt_myr=25.0, kernel_myr=None):
    """Gaussian-kernel smooth + uniform resample of a snapshot-cadence SFH.

    Parameters
    ----------
    t_gyr : array
        Cosmic times of the samples [Gyr] (any order).
    sfr : array
        SFR at those times [Msun/yr]; negatives are clipped to 0, non-finite
        samples dropped (zeros are kept — they carry the quench).
    dt_myr : float
        Output grid step [Myr].
    kernel_myr : float, optional
        Gaussian kernel sigma [Myr]. Default (None): the median input spacing,
        which suppresses the snapshot-level stochasticity while preserving the
        rise/quench shape on longer timescales. Floored at ``dt_myr``.

    Returns
    -------
    t_grid_gyr, sfr_smooth : ndarray, ndarray
        Uniform time grid spanning the input range and the Nadaraya-Watson
        (Gaussian-kernel) regression of the SFR on it. Smoothing is done in
        linear SFR, so the long-timescale mean is conserved.
    """
    t = np.asarray(t_gyr, float)
    s = np.asarray(sfr, float)
    ok = np.isfinite(t) & np.isfinite(s)
    t, s = t[ok], np.clip(s[ok], 0.0, None)
    if t.size < 2:
        raise ValueError("smooth_resample_sfh needs >= 2 finite SFH samples")
    o = np.argsort(t)
    t, s = t[o], s[o]

    dt = float(dt_myr) / 1e3                                       # Gyr
    sig = (float(kernel_myr) / 1e3 if kernel_myr
           else float(np.median(np.diff(t))))
    sig = max(sig, dt)

    t_grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    w = np.exp(-0.5 * ((t_grid[:, None] - t[None, :]) / sig) ** 2)
    sfr_smooth = (w @ s) / np.clip(w.sum(axis=1), 1e-300, None)
    return t_grid, sfr_smooth


def recent_sfr(t_gyr, sfr, avg_myr=100.0, t_obs_gyr=None, **kwargs):
    """<SFR> over the ``avg_myr`` before ``t_obs`` of the smoothed SFH.

    The SIMBA truth comparable to SED-fit SFR estimates: ``avg_myr=100``
    matches CIGALE's ``sfh.sfr100Myrs``; a small ``avg_myr`` (~one grid step)
    approximates the model-instantaneous ``sfh.sfr`` without the snapshot
    burst noise. ``t_obs_gyr`` defaults to the last sample; extra keyword
    arguments are passed to :func:`smooth_resample_sfh`.
    """
    t_grid, sm = smooth_resample_sfh(t_gyr, sfr, **kwargs)
    t1 = float(t_obs_gyr) if t_obs_gyr is not None else t_grid[-1]
    m = (t_grid >= t1 - avg_myr / 1e3) & (t_grid <= t1)
    return float(sm[m].mean()) if m.any() else float(sm[-1])


def build_mfrac_lookup(cache_path, overwrite=False, imf_type=1, pagb=1,
                       add_agb_dust_model=True):
    """FSPS surviving-mass-fraction lookup, powderday-consistent.

    ``get_spectrum(tage=0)`` returns the whole SSP age grid, so
    ``sp.stellar_mass`` comes back as an ``(nage,)`` array: one FSPS call per
    metallicity node instead of one per particle. The result is cached to
    ``cache_path`` (``.npz``); fsps is only imported on a cache miss.

    The defaults match powderday's active parameters_master (imf_type=1
    Chabrier, pagb=1, add_agb_dust_model=True); ``add_stellar_remnants`` is
    left at the FSPS default (1), so the fraction INCLUDES remnants — exactly
    the ``mass/mfrac`` scaling powderday applies in source_creation.py.

    Parameters
    ----------
    cache_path : str
        ``.npz`` cache file (keys ``zlegend, log_age_yr, mfrac, source``).
    overwrite : bool
        Rebuild even if the cache exists.

    Returns
    -------
    zlegend, log_age_yr, mfrac, source : ndarray, ndarray, ndarray, str
        Metallicity nodes, log10(age/yr) grid, ``(nz, nage)`` surviving
        fractions, and a provenance string (fsps version, libraries, flags).
    """
    import os
    if os.path.exists(cache_path) and not overwrite:
        d = np.load(cache_path)
        return (d["zlegend"], d["log_age_yr"], d["mfrac"], str(d["source"]))
    import fsps
    sp = fsps.StellarPopulation(imf_type=imf_type, pagb=pagb, sfh=0,
                                add_agb_dust_model=add_agb_dust_model,
                                add_neb_emission=False)  # no effect on m_star
    zleg, rows, log_age = np.asarray(sp.zlegend, float), [], None
    for iz in range(1, len(zleg) + 1):            # FSPS zmet is 1-based
        sp.params["zmet"] = iz
        sp.get_spectrum(tage=0)
        log_age = np.asarray(sp.log_age, float)          # log10(yr)
        rows.append(np.asarray(sp.stellar_mass, float))  # (nage,)
    mfrac = np.vstack(rows)
    src = (f"fsps {fsps.__version__} libs={sp.libraries} "
           f"imf_type={imf_type} pagb={pagb} agb_dust={add_agb_dust_model} "
           f"remnants={sp.params['add_stellar_remnants']}")
    np.savez(cache_path, zlegend=zleg, log_age_yr=log_age, mfrac=mfrac,
             source=src)
    return zleg, log_age, mfrac, src


def mfrac_of(age_gyr, zstar, lookup, clip=(0.05, 1.0)):
    """Surviving-mass fraction powderday used for these particles.

    Nearest metallicity node in LINEAR space — powderday's find_nearest_zmet
    (SED_gen.py) is ``argmin(|zlegend - Z|)``, not a log-space snap — then
    linear interpolation in log10(age/yr) clipped to the tabulated range.

    Parameters
    ----------
    age_gyr, zstar : arrays
        Stellar ages [Gyr] and total metallicities (mass fractions).
    lookup : tuple
        ``(zlegend, log_age_yr, mfrac[, source])`` from
        :func:`build_mfrac_lookup`.
    clip : (float, float)
        Bounds on the returned fraction.
    """
    zleg, log_age_yr, mfrac = lookup[0], lookup[1], lookup[2]
    iz = np.argmin(np.abs(np.asarray(zstar, float)[:, None] - zleg[None, :]),
                   axis=1)
    la = np.log10(np.clip(np.asarray(age_gyr, float), 1e-4, None) * 1e9)
    la = np.clip(la, log_age_yr[0], log_age_yr[-1])
    out = np.empty(la.shape, float)
    for k in np.unique(iz):
        m = iz == k
        out[m] = np.interp(la[m], log_age_yr, mfrac[k])
    return np.clip(out, clip[0], clip[1])


def archaeological_sfh(tform_gyr, mass_msun, t_obs_gyr, bin_myr=100.0,
                       dt_myr=25.0, kernel_myr=150.0):
    """Smoothed archaeological SFH from star-particle formation times.

    Histograms ``mass_msun`` (formed mass — divide current masses by
    :func:`mfrac_of` first) on a uniform ``bin_myr`` grid spanning
    ``[0, t_obs_gyr]`` in cosmic time, converts to SFR [Msun/yr], and smooths
    with :func:`smooth_resample_sfh`. The fixed 0-based edges make SFHs of
    different subsets of the same snapshot directly comparable, and the whole
    pipeline is linear in the weights: SFHs of disjoint particle sets sum to
    the SFH of their union.

    Returns
    -------
    t_grid_gyr, sfr : ndarray, ndarray
        Uniform cosmic-time grid (``dt_myr`` step) and smoothed SFR [Msun/yr].
    """
    edges = np.arange(0.0, float(t_obs_gyr) + bin_myr / 1e3, bin_myr / 1e3)
    centres = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(np.asarray(tform_gyr, float), bins=edges,
                        weights=np.asarray(mass_msun, float))
    return smooth_resample_sfh(centres, h / (bin_myr * 1e6),
                               dt_myr=dt_myr, kernel_myr=kernel_myr)


def projected_region_sfh(tform_gyr, mass_msun, pos_kpc, nhat, r_in_kpc,
                         r_out_kpc, t_obs_gyr, nstar_min=20, bin_myr=100.0,
                         dt_myr=25.0, kernel_myr=150.0):
    """Archaeological SFH of the stars in a projected radial region.

    Selects ``r_in < R <= r_out`` with ``R`` the image-plane radius along the
    sightline ``nhat`` (:func:`~simbanator.utils.geometry.projected_radius`) —
    the geometry of a Hyperion SED aperture; ``r_in_kpc <= 0`` degrades to the
    cumulative disc ``R <= r_out``. Positions must already be relative to the
    aperture centre, in the same (proper) units as the radii.

    Returns
    -------
    t_grid_gyr, sfr, nstar : ndarray, ndarray, int
        The smoothed SFH from :func:`archaeological_sfh` and the number of
        selected star particles — or ``(None, None, nstar)`` when
        ``nstar < nstar_min`` (shot-noise, not an SFH; the caller records the
        skip).
    """
    rp = projected_radius(pos_kpc, nhat)
    msk = (rp <= float(r_out_kpc)) if float(r_in_kpc) <= 0.0 else \
          ((rp > float(r_in_kpc)) & (rp <= float(r_out_kpc)))
    nstar = int(msk.sum())
    if nstar < int(nstar_min):
        return None, None, nstar
    t, s = archaeological_sfh(np.asarray(tform_gyr, float)[msk],
                              np.asarray(mass_msun, float)[msk],
                              t_obs_gyr, bin_myr=bin_myr, dt_myr=dt_myr,
                              kernel_myr=kernel_myr)
    return t, s, nstar


def sfr_delayed_bq(t, A, tau, age_main, age_bq, r_sfr, t_obs):
    """Delayed-tau main population + burst/quench (CIGALE sfhdelayedbq), SFR(t).
    t, t_obs, tau, age_main, age_bq all in Gyr; formation at t_obs-age_main,
    quench at t_obs-age_bq to a constant r_sfr*SFR(t_q)."""
    t = np.asarray(t, float)
    t0 = t_obs - age_main
    x = np.clip(t - t0, 0.0, None)
    main = A * (x / tau) * np.exp(-x / tau)
    tq = t_obs - age_bq
    xq = max(tq - t0, 0.0)
    sfr_tq = A * (xq / tau) * np.exp(-xq / tau)
    return np.where(t >= tq, r_sfr * sfr_tq, main)


def fit_delayed_bq(t, sfr, t_obs, age_bq0=None):
    """Bounded, truth-seeded fit of :func:`sfr_delayed_bq`; dict (Myr) or None."""
    from scipy.optimize import curve_fit
    t = np.asarray(t, float); sfr = np.asarray(sfr, float)
    ok = np.isfinite(t) & np.isfinite(sfr) & (sfr >= 0)
    t, sfr = t[ok], sfr[ok]
    if t.size < 6:
        return None
    o = np.argsort(t); t, sfr = t[o], sfr[o]
    span = t[-1] - t[0]; peak = max(sfr.max(), 1e-6)
    age_main0 = min(max(span * 1.1, 1.0), t_obs)
    age_bq0 = age_bq0 if (age_bq0 and np.isfinite(age_bq0)) else max(span * 0.2, 0.1)
    age_bq0 = min(age_bq0, 0.9 * age_main0)
    p0 = [peak * 3, 2.0, age_main0, age_bq0, 0.1]
    lo = [0.0, 0.1, 1.0, 0.03, 0.0]
    hi = [peak * 1e3, 12.0, t_obs, 0.8 * t_obs, 3.0]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
    try:
        popt, _ = curve_fit(lambda tt, A, tau, am, ab, r:
                            sfr_delayed_bq(tt, A, tau, am, ab, r, t_obs),
                            t, sfr, p0=p0, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None
    A, tau, am, ab, r = popt
    resid = sfr - sfr_delayed_bq(t, *popt, t_obs)
    r2 = 1.0 - np.sum(resid**2) / max(np.sum((sfr - sfr.mean())**2), 1e-30)
    return dict(A=A, tau_main_myr=tau * 1e3, age_main_myr=am * 1e3,
                age_bq_myr=ab * 1e3, r_sfr=r, r2=r2, t_obs=t_obs)
