"""Star-formation-history track utilities.

SIMBA's per-snapshot ``sfr`` is the instantaneous gas SFR sampled at the
snapshot cadence (~100-300 Myr), so the tracks carry burst-to-burst scatter
that smooth parametric SFH forms (and SED-derived SFR estimates) cannot and
should not follow. These helpers turn such a track into a smooth, uniformly
sampled SFH suitable for fitting and for truth comparisons.
"""

import numpy as np

__all__ = ["smooth_resample_sfh", "recent_sfr",
           "sfr_delayed_bq", "fit_delayed_bq"]


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
