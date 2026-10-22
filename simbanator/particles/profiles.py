"""Compute radial profiles of galaxy properties from simulation snapshots."""

import numpy as np


def radial_profile(snapfile, catfile, galaxy_id, properties_dict,
                   radii=None, region=False, dens=False, dim=None, norm=True):
    """Compute radial profiles of specified particle properties.

    Parameters
    ----------
    snapfile : str
        Path to the simulation snapshot file.
    catfile : str
        Path to the Caesar catalog file.
    galaxy_id : int
        Galaxy GroupID in the Caesar catalog.
    properties_dict : dict
        ``{particle_type: [property_names]}``
        e.g. ``{'PartType0': ['Masses', 'Metallicity']}``.
    radii : array-like, optional
        Radial bin edges in kpc (default ``np.arange(0, 100, 1)``).
    region : bool
        If *True*, use all particles; otherwise use only those
        belonging to the galaxy.
    dens : bool
        If *True*, compute surface density; otherwise compute mean.
    dim : dict, optional
        Unit conversion mapping,
        e.g. ``{'PartType0': ['code_mass', 'Msun']}``.
    norm : bool
        Normalise profiles to [0, 1].

    Returns
    -------
    tuple
        ``(radii, profiles)`` where *profiles* is a nested dict
        ``{ptype: {property: list}}``.
    """
    if radii is None:
        radii = np.arange(0, 100, 1)

    import yt       # heavy dep – lazy import
    import caesar   # heavy dep – lazy import

    ds = yt.load(snapfile)
    obj = caesar.load(catfile)
    ad = ds.all_data()
    gal = [i for i in obj.galaxies if i.GroupID == galaxy_id][0]
    center = gal.pos.in_units('kpc').value

    def get_data(particle_type, props, dim_info, indices=None):
        pos = ad[particle_type, 'Coordinates'].in_units('kpc').value
        if indices is not None:
            pos = pos[indices]
        data_list = []
        for prop in props:
            data = ad[particle_type, prop[:-2] if '_s' in prop else prop]
            if indices is not None:
                data = data[indices]
            data = ds.arr(data, dim_info[particle_type][0]).in_units(dim_info[particle_type][1]).value
            data_list.append(data)
        return pos, data_list

    profiles = {
        ptype: {prop: [] for prop in properties_dict[ptype]}
        for ptype in properties_dict
    }

    for ptype, props in properties_dict.items():
        indices = None
        if not region:
            if ptype == 'PartType0':
                indices = gal.glist
            elif ptype == 'PartType4':
                indices = gal.slist
        pos, data = get_data(ptype, props, dim, indices=indices)

        radial_distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

        for r in radii:
            mask = radial_distances < r
            A = np.pi * r ** 2

            for i, prop in enumerate(props):
                if dens:
                    profiles[ptype][prop].append(np.sum(data[i][mask]) / A)
                else:
                    profiles[ptype][prop].append(np.mean(data[i][mask]))

    if norm:
        for ptype in profiles:
            for prop in profiles[ptype]:
                if len(profiles[ptype][prop]) > 0:
                    temp = np.array(profiles[ptype][prop])
                    ptp = np.nanmax(temp) - np.nanmin(temp)
                    if ptp > 0:
                        profiles[ptype][prop] = (temp - np.nanmin(temp)) / ptp
                    else:
                        profiles[ptype][prop] = temp
    else:
        for ptype in profiles:
            for prop in profiles[ptype]:
                profiles[ptype][prop] = np.array(profiles[ptype][prop])

    return radii, profiles


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# Profiles of reduced particle files (the face-on frame stored in the file)
# ═════════════════════════════════════════════════════════════════════════════

from .frames import reduced_radius, select_gas   # noqa: E402

COMPONENT_DATASETS = {"gas": "m_gas", "dust": "m_dust", "HI": "m_HI", "H2": "m_H2"}
T_COLD, T_HOT = 1.0e4, 1.0e5                       # gas phase cuts [K] (as in build_profiles_job)
PHASES = {"cold": (0.0, T_COLD), "warm": (T_COLD, T_HOT), "hot": (T_HOT, np.inf)}


def _default_edges(red, n=26):
    return np.linspace(0.0, float(red.get("rmax_kpc", 100.0)) if red else 100.0, n)


def _annulus_area(edges):
    edges = np.asarray(edges, float)
    return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


def reduced_sigma_profile(red, comp="gas", edges=None, geom="cyl", region="all", member_only=False):
    """Σ(R) [Msun/kpc^2] of a gas component (``gas``, ``dust``, ``HI``, ``H2``) on the bin ``edges`` [kpc] -> ``(rmid, Σ)``.

    ``region`` ``"ism"`` / ``"cgm"`` keeps star-forming / non-star-forming gas. NaN when the file
    lacks the component.
    """
    edges = _default_edges(red) if edges is None else np.asarray(edges, float)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    R, m, reg = select_gas(red, COMPONENT_DATASETS[comp], geom, region, member_only)
    if R is None:
        return rmid, np.full(len(rmid), np.nan)
    s, _ = np.histogram(R[reg], bins=edges, weights=np.nan_to_num(m[reg]))
    return rmid, s / _annulus_area(edges)


def reduced_sigma_scalar(red, comp="H2", r_aper_kpc=10.0, geom="cyl", region="all", member_only=False):
    """Mean Σ [Msun/kpc^2] of a gas component inside a circular (``cyl``) or spherical (``3d``) aperture.

    0 for a genuinely empty aperture (the galaxy has the gas, just not inside ``r``); NaN when
    the file has no data for the component.
    """
    if red is None:
        return np.nan
    R, m, reg = select_gas(red, COMPONENT_DATASETS[comp], geom, region, member_only)
    if R is None:
        return np.nan
    sel = reg & (R <= r_aper_kpc)
    if not np.any(sel):
        return 0.0 if np.any(reg) else np.nan
    return float(np.nansum(m[sel]) / (np.pi * r_aper_kpc ** 2))


def reduced_aperture_mass(red, comp="H2", r_kpc=10.0, geom="3d", member_only=False):
    """Mass [Msun] of a gas component inside the aperture (:func:`reduced_sigma_scalar` x the aperture area)."""
    return reduced_sigma_scalar(red, comp, r_aper_kpc=r_kpc, geom=geom, member_only=member_only) * np.pi * r_kpc ** 2


def reduced_phase_profile(red, phase="cold", edges=None, geom="cyl"):
    """Σ(R) [Msun/kpc^2] of the gas in one temperature phase (``cold`` < 1e4 K, ``warm``, ``hot`` > 1e5 K); NaN without ``temp``."""
    edges = _default_edges(red) if edges is None else np.asarray(edges, float)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    g = red.get("gas") if red else None
    if g is None or "temp" not in g or "m_gas" not in g:
        return rmid, np.full(len(rmid), np.nan)
    R = reduced_radius(g["pos"], red["evecs"], geom)
    m = np.asarray(g["m_gas"], float); T = np.asarray(g["temp"], float)
    lo, hi = PHASES[phase]
    sel = np.isfinite(T) & (T >= lo) & (T < hi)
    s, _ = np.histogram(R[sel], bins=edges, weights=np.nan_to_num(m[sel]))
    return rmid, s / _annulus_area(edges)


def reduced_temperature_profile(red, edges=None, geom="cyl", region="all"):
    """Mass-weighted gas temperature T(R) [K]; NaN in empty bins or without ``temp``."""
    edges = _default_edges(red) if edges is None else np.asarray(edges, float)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    g = red.get("gas") if red else None
    if g is None or "temp" not in g:
        return rmid, np.full(len(rmid), np.nan)
    R, m, reg = select_gas(red, "m_gas", geom, region)
    if R is None:
        return rmid, np.full(len(rmid), np.nan)
    T = np.asarray(g["temp"], float)
    num, _ = np.histogram(R[reg], bins=edges, weights=np.nan_to_num(m[reg] * T[reg]))
    den, _ = np.histogram(R[reg], bins=edges, weights=np.nan_to_num(m[reg]))
    with np.errstate(all="ignore"):
        return rmid, np.where(den > 0, num / den, np.nan)


def reduced_star_sigma_profile(red, edges=None, geom="cyl", member_only=False):
    """Σ_*(R) [Msun/kpc^2] of the star particles of a reduced file; NaN without stars."""
    edges = _default_edges(red) if edges is None else np.asarray(edges, float)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    s = red.get("star") if red else None
    if not s or "pos" not in s or "m_star" not in s:
        return rmid, np.full(len(rmid), np.nan)
    R = reduced_radius(s["pos"], red["evecs"], geom)
    m = np.asarray(s["m_star"], float)
    if member_only and "member" in s:
        keep = np.asarray(s["member"], bool); R, m = R[keep], m[keep]
    h, _ = np.histogram(R, bins=edges, weights=np.nan_to_num(m))
    return rmid, h / _annulus_area(edges)


def member_radii(red, comp="star", geom="3d", fracs=(0.2, 0.5, 0.8), nmin=5):
    """Radii [kpc] enclosing ``fracs`` of the CAESAR-member mass of ``comp`` (``star`` or a gas component); NaN below ``nmin`` members."""
    grp_name = "star" if comp == "star" else "gas"
    mkey = "m_star" if comp == "star" else COMPONENT_DATASETS[comp]
    grp = red.get(grp_name) if red else None
    nanret = {f: np.nan for f in fracs}
    if not grp or not all(k in grp for k in ("member", "pos", mkey)):
        return nanret
    memb = np.asarray(grp["member"], bool)
    if memb.sum() < nmin:
        return nanret
    R = reduced_radius(np.asarray(grp["pos"], float)[memb], red["evecs"], geom)
    m = np.asarray(grp[mkey], float)[memb]
    o = np.argsort(R); cm = np.cumsum(m[o]); tot = float(cm[-1])
    if tot <= 0:
        return nanret
    return {f: float(np.interp(f * tot, cm, R[o])) for f in fracs}


def enclosed_radii(profile, edges, fracs=(0.2, 0.5, 0.8, 0.9)):
    """Radii enclosing ``fracs`` of the mass of a binned Σ(R) profile (linear Σ on bin ``edges``); NaNs when empty."""
    if profile is None:
        return (np.nan,) * len(fracs)
    edges = np.asarray(edges, float)
    M = np.nancumsum(np.nan_to_num(np.asarray(profile, float) * _annulus_area(edges)))
    tot = float(M[-1])
    if not np.isfinite(tot) or tot <= 0:
        return (np.nan,) * len(fracs)
    return tuple(float(np.interp(f * tot, M, edges[1:])) for f in fracs)


def mean_sigma_inside(profile, rmid, R):
    """Mean of a binned Σ(R) over the bins whose centre is within ``R`` (NaN when R is invalid or nothing is inside)."""
    if profile is None or not np.isfinite(R) or R <= 0:
        return np.nan
    inb = np.asarray(rmid, float) <= R
    if not inb.any():
        return np.nan
    v = np.asarray(profile, float)[inb]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def radius_from_curve_of_growth(M, R):
    """Half-mass radius from a cumulative curve of growth ``M(<R)`` on the aperture radii ``R``.

    Log-log interpolation between apertures; inside the innermost aperture Σ is taken flat
    (M ∝ R²). NaN when the curve is not finite or not positive.
    """
    M, R = np.asarray(M, float), np.asarray(R, float)
    if not np.all(np.isfinite(M)) or M[-1] <= 0:
        return np.nan
    f = M / M[-1]
    if f[0] >= 0.5:
        return float(R[0] * np.sqrt(0.5 / max(f[0], 1e-9)))
    lf = np.log10(np.maximum(f, 1e-9))
    return float(10 ** np.interp(np.log10(0.5), lf, np.log10(R)))


def annulus_surface_densities(path, annuli_kpc, apertures_kpc=(), star_member_only=False):
    """Face-on Σ_*, Σ_dust, Σ_H2 [Msun/pc^2] per annulus, star counts per annulus, and star / gas counts
    inside each aperture, from one reduced file -> dict (None when the file is missing or unreadable).
    """
    import os
    import h5py
    if not os.path.exists(path):
        return None
    edges = np.asarray(annuli_kpc, float); area_pc2 = _annulus_area(edges) * 1e6
    try:
        with h5py.File(path, "r") as f:
            ev = np.asarray(f.attrs["evecs"], float)
            out = {}
            for grp, mkeys in (("star", ("m_star",)), ("gas", ("m_dust", "m_H2"))):
                g = f.get(grp)
                if g is None or "pos" not in g or len(g["pos"]) == 0:
                    out[grp] = (np.zeros(0), {k: np.zeros(0) for k in mkeys}, np.zeros(0, bool)); continue
                R = reduced_radius(np.asarray(g["pos"][:], float), ev, "cyl")
                m = {k: (np.asarray(g[k][:], float) if k in g else np.zeros(len(R))) for k in mkeys}
                mem = np.asarray(g["member"][:], bool) if "member" in g else np.ones(len(R), bool)
                out[grp] = (R, m, mem)
    except (OSError, RuntimeError):
        return None
    Rs, ms, mems = out["star"]; m_star = ms["m_star"]
    if star_member_only:
        Rs, m_star = Rs[mems], m_star[mems]
    Rg, mg, _ = out["gas"]
    return dict(sig_star=np.histogram(Rs, bins=edges, weights=m_star)[0] / area_pc2,
                sig_dust=np.histogram(Rg, bins=edges, weights=mg["m_dust"])[0] / area_pc2,
                sig_h2=np.histogram(Rg, bins=edges, weights=mg["m_H2"])[0] / area_pc2,
                n_star=np.histogram(Rs, bins=edges)[0].astype(float),
                nstar_ap=np.array([np.sum(Rs < ap) for ap in apertures_kpc], float),
                ngas_ap=np.array([np.sum(Rg < ap) for ap in apertures_kpc], float))


def detect_gas_fields(f):
    """Names of the dust, H2-fraction, metallicity and neutral-fraction datasets in a snapshot's ``PartType0`` (None if absent)."""
    have = set(f["PartType0"].keys())
    dust = next((k for k in ("Dust_Masses", "DustMasses") if k in have), None)
    fmol = next((k for k in ("FractionH2", "fH2", "GrackleH2", "f_H2") if k in have), None)
    return dust, fmol, ("Metallicity" if "Metallicity" in have else None), \
        ("NeutralHydrogenAbundance" if "NeutralHydrogenAbundance" in have else None)


# ── profile cache: Σ(R) of every reduced file, computed once per binning ─────────────────────────

def profile_cache_path(table_dir, nbins, rmin, rmax, geom, region):
    """Cache file named after its binning, so a different binning writes a new file instead of overwriting."""
    import os
    return os.path.join(table_dir, f"reduced_profiles_cache_n{nbins}_r{rmin:g}-{rmax:g}_{geom}_{region}.hdf5")


def log_edges(rmin, rmax, nbins):
    """``[0, rmin, ..., rmax]``: a central disc plus ``nbins`` log-spaced bins (the profile-cache binning)."""
    return np.concatenate([[0.0], np.logspace(np.log10(rmin), np.log10(rmax), nbins)])


def file_profiles(red, edges, comps=("gas", "H2", "HI", "dust", "star"), geom="cyl", region="all"):
    """All cached profiles of one reduced file -> ``{name: Σ(R) float32}`` (gas components, stars, and, with
    ``temp``, the cold / warm / hot phases and ``Tprof``)."""
    out = {}
    for comp in comps:
        if comp == "star":
            _, sig = reduced_star_sigma_profile(red, edges=edges, geom=geom)
        else:
            _, sig = reduced_sigma_profile(red, comp=comp, edges=edges, geom=geom, region=region)
        out[comp] = sig.astype("f4")
    if "temp" in red.get("gas", {}):
        for ph in ("cold", "warm", "hot"):
            out[f"phase_{ph}"] = reduced_phase_profile(red, ph, edges=edges, geom=geom)[1].astype("f4")
        out["Tprof"] = reduced_temperature_profile(red, edges=edges, geom=geom, region=region)[1].astype("f4")
    return out


def build_profile_cache(cache_path, reduced_files, load, edges, attrs=None, rebuild=False, **profile_kwargs):
    """Cache :func:`file_profiles` of every reduced file into ``cache_path`` (groups ``sNNN_gGGGGGG``).

    ``reduced_files``: paths; ``load(path) -> reduced dict or None``. Incremental: files already in
    the cache are skipped unless ``rebuild``. Returns the number of profiles written.
    """
    import os
    import h5py
    from .cutouts import parse_reduced_name
    have = set()
    if os.path.exists(cache_path) and not rebuild:
        with h5py.File(cache_path, "r") as cf:
            have = set(cf.keys())
    todo = []
    for p in reduced_files:
        sg = parse_reduced_name(os.path.basename(p))
        if sg is not None and (rebuild or f"s{sg[0]:03d}_g{sg[1]:06d}" not in have):
            todo.append((sg, p))
    if not todo:
        return 0
    results = []
    for (snap, gx), p in todo:
        red = load(p)
        if red is not None and "gas" in red:
            results.append((f"s{snap:03d}_g{gx:06d}", file_profiles(red, edges, **profile_kwargs)))
    with h5py.File(cache_path, "w" if rebuild else "a") as cf:
        cf.attrs.update(attrs or {})
        if "edges" not in cf:
            cf.create_dataset("edges", data=np.asarray(edges, float))
        for key, prof in results:
            grp = cf.require_group(key)
            for name, arr in prof.items():
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=arr)
    return len(results)


def load_profile_cache(cache_path, nbins=None):
    """``{(snap, gx): {name: Σ(R)}}`` from a profile cache; entries whose length is not ``nbins`` are skipped."""
    import os
    import h5py
    out = {}
    if not os.path.exists(cache_path):
        return out
    with h5py.File(cache_path, "r") as cf:
        for key in cf.keys():
            if key == "edges":
                continue
            d = {c: cf[key][c][:] for c in cf[key].keys()}
            if nbins is not None and any(np.ndim(v) != 1 or len(v) != nbins for v in d.values()):
                continue
            out[(int(key[1:4]), int(key[6:]))] = d
    return out
