"""Attenuation, SED conversions and dust emission (numpy + astropy; no hyperion)."""
import numpy as np

__all__ = ["av_from_pair", "attenuation_curve", "av_in_window", "bolometric_from_nulnu", "nulnu_to_fnu_factor",
           "fnu_to_nulnu", "mbb_flux_mjy", "anchor_av", "attenuation_table"]


def av_from_pair(dusty, dust_free, magnitudes=False):
    """A_V (or A_λ) from a dusty / dust-free pair of fluxes (``-2.5 log10(F_dust / F_free)``) or magnitudes (difference).

    Positive = attenuated; non-finite -> NaN.
    """
    a, b = np.asarray(dusty, float), np.asarray(dust_free, float)
    with np.errstate(all="ignore"):
        av = a - b if magnitudes else -2.5 * np.log10(a / b)
    av = np.array(av, float)
    av[~np.isfinite(av)] = np.nan
    return av


def attenuation_curve(dusty_sed, free_sed, grid):
    """A_λ on ``grid`` from a dust_on / dust_off pair of SEDs ``(wl, nuLnu)``; None when either is missing."""
    from ..rt.outputs import on_log_grid
    if dusty_sed is None or free_sed is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return -2.5 * np.log10(on_log_grid(*dusty_sed, grid) / on_log_grid(*free_sed, grid))


def av_in_window(dusty_sed, free_sed, window_um=np.linspace(0.54, 0.56, 25)):
    """A_V from the median flux of a dust_on / dust_off pair in a rest-frame window (Bessell V core by default); None when not positive."""
    f_on = np.median(np.interp(window_um, *dusty_sed))
    f_off = np.median(np.interp(window_um, *free_sed))
    if not (f_on > 0 and f_off > 0):
        return None
    return float(-2.5 * np.log10(f_on / f_off))


def bolometric_from_nulnu(wav_um, nuLnu):
    """∫ L_nu dnu [erg/s] of a ``nu L_nu`` curve on a wavelength grid [µm]."""
    nu = 2.99792458e14 / np.asarray(wav_um, float)
    o = np.argsort(nu)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.asarray(nuLnu, float)[o] / nu[o], nu[o]))


def nulnu_to_fnu_factor(grid_rest_um, z, cosmology=None):
    """``(observed wavelengths [µm], factor)`` such that ``nu L_nu [erg/s] x factor = F_nu [mJy]`` at redshift ``z`` (Planck13 default)."""
    if cosmology is None:
        from astropy.cosmology import Planck13 as cosmology
    import astropy.units as u
    wav_obs = np.asarray(grid_rest_um, float) * (1.0 + z)
    dl = cosmology.luminosity_distance(z).to(u.cm).value
    return wav_obs, 1.0 / (4.0 * np.pi * dl ** 2 * (2.99792458e14 / wav_obs)) / 1e-26


def fnu_to_nulnu(wl_obs_um, fnu_mjy, z, cosmology=None):
    """Observed ``(wavelength [µm], F_nu [mJy])`` -> rest-frame ``(wavelength [µm], nu L_nu [erg/s])``."""
    if cosmology is None:
        from astropy.cosmology import Planck13 as cosmology
    import astropy.units as u
    wl = np.asarray(wl_obs_um, float)
    dl = cosmology.luminosity_distance(z).to(u.cm).value
    nu = 2.99792458e14 / wl
    return wl / (1.0 + z), nu * np.asarray(fnu_mjy, float) * 1e-26 * 4 * np.pi * dl ** 2


def mbb_flux_mjy(m_dust_msun, z, lam_obs_um, t_dust_k, kappa0_m2_kg=0.077, lam0_um=850.0, beta=1.8, cosmology=None):
    """Optically thin modified-blackbody flux density [mJy]: ``(1+z) κ(ν_rest) B_ν(ν_rest, T_d) M_dust / d_L²``.

    κ = κ0 (ν/ν0)^β with James+02 κ0 = 0.077 m²/kg at 850 µm and β = 1.8 by default; Planck15 by default.
    """
    import astropy.constants as const
    import astropy.units as u
    if cosmology is None:
        from astropy.cosmology import Planck15 as cosmology
    nu_rest = (const.c / (np.asarray(lam_obs_um, float) * u.um)).to(u.Hz) * (1.0 + z)
    nu0 = (const.c / (lam0_um * u.um)).to(u.Hz)
    kappa = kappa0_m2_kg * u.m ** 2 / u.kg * (nu_rest / nu0) ** beta
    x = (const.h * nu_rest / (const.k_B * t_dust_k * u.K)).to(u.dimensionless_unscaled)
    bnu = (2.0 * const.h * nu_rest ** 3 / const.c ** 2) / np.expm1(x.value)
    dl = cosmology.luminosity_distance(z)
    return ((1.0 + z) * (np.asarray(m_dust_msun) * u.Msun) * kappa * bnu / dl ** 2).to(u.mJy).value


def anchor_av(aperture, annulus_av_path=None, attenuation_path=None):
    """A_V of every galaxy at its anchor in ``aperture`` -> ``({(snap, gal_id): A_V}, description)``.

    The median over the sightlines of the annulus A_V table (powderday_flux Part 8b) when given and
    present, else the fiducial-sightline column ``A_V_<aperture>`` of the attenuation table (Part 7a).
    """
    import os
    from ..io.tables import read_table
    from ..io.products import need
    if annulus_av_path and os.path.exists(annulus_av_path):
        A = read_table(annulus_av_path)
        A = A[np.asarray(A["aperture"]) == aperture]
        if not len(A):
            raise KeyError(f"{annulus_av_path} has no aperture {aperture!r}")
        from collections import defaultdict
        groups = defaultdict(list)
        for s_, g_, a_ in zip(np.asarray(A["snap"], int), np.asarray(A["gal_id"], int), np.asarray(A["A_V"], float)):
            groups[(int(s_), int(g_))].append(a_)
        out = {k: float(np.median(v)) for k, v in groups.items()}
        return out, f"median over {len(np.unique(np.asarray(A['incl'])))} sightlines"
    A = read_table(need(attenuation_path, "powderday_flux_quenched_m25 Part 7a (or Part 8b for the annulus table)"))
    col = f"A_V_{aperture}"
    if col not in A.colnames:
        raise KeyError(f"{attenuation_path} has no column {col!r}")
    return {(int(s), int(i)): float(v) for s, i, v in zip(A["snap"], A["gal_id"], np.asarray(A[col], float))}, "fiducial sightline"


def attenuation_table(dusty_catalog, free_catalog, bands, keys=("snap", "gal_id")):
    """Join a dust_on and a dust_off flux catalogue and add ``A_<name>`` columns -> Table; None when either is missing.

    ``bands`` maps output names to the flux column shared by both catalogues (``{"A_V": "V_mJy"}``).
    """
    import os
    from astropy.table import Table, join
    from .fluxes import attenuation_mag
    if not (os.path.exists(dusty_catalog) and os.path.exists(free_catalog)):
        return None
    on, off = Table.read(dusty_catalog), Table.read(free_catalog)
    off_cols = list(keys) + [c for c in bands.values() if c in off.colnames]
    m = join(on, off[off_cols], keys=list(keys), table_names=["on", "off"], metadata_conflicts="silent")
    for name, col in bands.items():
        m[name] = (attenuation_mag(m[f"{col}_on"], m[f"{col}_off"])
                   if f"{col}_on" in m.colnames and f"{col}_off" in m.colnames else np.full(len(m), np.nan))
    return m
