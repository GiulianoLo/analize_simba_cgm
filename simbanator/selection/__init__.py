"""Generic sample rules and the dust label.

The candidate divisions of the quenched population (coupling class, X-ray class, kappa_gas, peel, ...)
belong to projects/divisions, not here.

- :func:`quenched_sample`  the quenched selection (the m25 rule by default) -> mask + the individual cuts
- :func:`dust_label`       dusty / dust-poor on log M_dust/M* at a threshold
- :func:`dusty_from_av`    1 / 0 / -1 (not measured) from an attenuation
- :func:`mass_bins`        bin index of log M* on fixed edges
- :func:`sf_controls`      star-forming partners matched in stellar mass to a quenched sample
"""
import numpy as np

__all__ = ["M25_RULE", "quenched_sample", "dust_label", "dusty_from_av", "mass_bins", "sf_controls"]

# The m25 rule (powderday_flux_quenched_m25 Part 2, paper_ism_prediction_boxes Part 1).
M25_RULE = dict(mass_floor=10.0, passive_factor=0.2, ngas_min=21, nstar_min=20, dust_to_h2_min=1e-4)


def quenched_sample(mstar, sfr, t_hubble_yr, ngas=None, nstar=None, mdust=None, mh2=None, central=None,
                    mass_floor=10.0, passive_factor=0.2, ngas_min=21, nstar_min=20, dust_to_h2_min=1e-4,
                    centrals_only=False):
    """The quenched selection at one epoch -> ``(mask, {cut name: mask})``.

    Cuts: ``massive`` (log M* > ``mass_floor``), ``passive`` (sSFR < ``passive_factor`` / t_H, the QT
    threshold of the quench finder), ``ngas`` / ``nstar`` (at least that many particles), ``dust/H2``
    (``M_dust >= dust_to_h2_min x M_H2``, written as a product so M_H2 = 0 rows pass) and, with
    ``centrals_only``, ``central``. A cut whose input is None is skipped (it passes everything).
    The defaults are the m25 rule; the older residual_dust / quench_mode samples used
    ``mass_floor=9.5, ngas_min=5 / 10`` and no dust/H2 cut.
    """
    mstar = np.asarray(mstar, float); sfr = np.asarray(sfr, float)
    with np.errstate(all="ignore"):
        ssfr = np.where(mstar > 0, sfr / mstar, np.nan)
        lm = np.log10(np.where(mstar > 0, mstar, np.nan))
    cuts = {"massive": lm > mass_floor, "passive": ssfr < passive_factor / np.asarray(t_hubble_yr, float)}
    if ngas is not None and ngas_min is not None:
        cuts["ngas"] = np.asarray(ngas, float) >= ngas_min
    if nstar is not None and nstar_min is not None:
        cuts["nstar"] = np.asarray(nstar, float) >= nstar_min
    if mdust is not None and mh2 is not None and dust_to_h2_min is not None:
        with np.errstate(all="ignore"):
            cuts["dust/H2"] = np.asarray(mdust, float) >= dust_to_h2_min * np.asarray(mh2, float)
    if centrals_only and central is not None:
        cuts["central"] = np.asarray(central, float) > 0
    mask = np.ones(len(mstar), bool)
    for m in cuts.values():
        mask &= m
    return mask, cuts


def dust_label(mdust, mstar, log_threshold=-5.0):
    """``(dusty, dust_poor)`` masks on log10(M_dust/M*) > threshold; galaxies with an undefined fraction are in neither."""
    with np.errstate(all="ignore"):
        f = np.where(np.asarray(mstar, float) > 0, np.asarray(mdust, float) / np.asarray(mstar, float), np.nan)
    ok = np.isfinite(f)
    dusty = ok & (f > 10.0 ** log_threshold)
    return dusty, ok & ~dusty


def dusty_from_av(av, threshold=0.1):
    """1 where A_V > threshold, 0 where it is not, -1 where A_V is not measured (NaN)."""
    av = np.asarray(av, float)
    return np.where(np.isnan(av), -1, (av > threshold).astype(int))


def mass_bins(log_mstar, edges):
    """Bin index of each log M* on ``edges`` (``[lo, hi)``), -1 outside or NaN."""
    lm = np.asarray(log_mstar, float)
    b = np.full(len(lm), -1, int)
    for k in range(len(edges) - 1):
        b[(lm >= edges[k]) & (lm < edges[k + 1])] = k
    return b


def sf_controls(log_mstar, eligible, quenched_ids, quenched_logm, ratio=1):
    """Star-forming partners for a quenched sample: for each quenched galaxy (most massive first) the ``ratio``
    eligible galaxies nearest in log M*, without replacement.

    ``log_mstar`` / ``eligible`` run over the whole catalogue (``eligible``: the star-forming cuts);
    galaxies already in ``quenched_ids`` are never picked. Returns a list of
    ``(control index, matched quenched id, Δlog M*)``.
    """
    lm = np.asarray(log_mstar, float)
    cand = np.where(np.asarray(eligible, bool) & ~np.isin(np.arange(len(lm)), np.asarray(quenched_ids, int)))[0]
    taken, out = set(), []
    qlm = np.asarray(quenched_logm, float)
    for qi in np.argsort(qlm)[::-1]:
        if not np.isfinite(qlm[qi]):
            continue
        for _ in range(ratio):
            free = np.array([c for c in cand if c not in taken], int)
            if not free.size:
                break
            c = int(free[np.argmin(np.abs(lm[free] - qlm[qi]))])
            taken.add(c)
            out.append((c, int(np.asarray(quenched_ids)[qi]), float(lm[c] - qlm[qi])))
    return out
