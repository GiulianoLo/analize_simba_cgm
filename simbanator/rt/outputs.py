"""Powderday outputs: tree layout, completeness and job status, staged masters, SED readers.

numpy + h5py only; hyperion is imported inside the SED readers.

Tree layout: ``<sed_out>/<tag>/powderday_sed_out/snap_NNN/gal_<gid>/snapNNN.galaxyGGGGGG.rtout.sed``.
"""
import collections
import glob
import os
import re
import time

import numpy as np
import h5py

__all__ = [
    "rtout_path", "tree_rtout_path", "sed_is_complete", "log_phase", "source_types", "count_bh_particles",
    "parse_master", "staged_master_values", "rt_census", "check_bh_field_names",
    "read_rt_sed", "on_log_grid", "stack_seds", "sed_to_flux_density", "dust_opacity", "grid_centres",
]


# ═════════════════════════════════════════════════════════════════════════════
# Paths, completeness, status
# ═════════════════════════════════════════════════════════════════════════════

def rtout_path(sed_dir, snap, gid):
    """``<sed_dir>/snap_NNN/gal_<gid>/snapNNN.galaxyGGGGGG.rtout.sed`` (``sed_dir`` = a run's ``powderday_sed_out``)."""
    snap, gid = int(snap), int(gid)
    return os.path.join(sed_dir, f"snap_{snap:03d}", f"gal_{gid}", f"snap{snap:03d}.galaxy{gid:06d}.rtout.sed")


def tree_rtout_path(sed_out, tag, snap, gid):
    """The .rtout.sed of one galaxy in the run tree ``tag`` under ``sed_out``."""
    return rtout_path(os.path.join(sed_out, tag, "powderday_sed_out"), snap, gid)


def sed_is_complete(path):
    """True only when the RT finished: hyperion creates the .rtout.sed when it starts and writes the
    ``Peeled`` SED groups last, so the file existing is not enough."""
    if not (path and os.path.exists(path)):
        return False
    try:
        with h5py.File(path, "r") as f:
            g = f.get("Peeled")
            return bool(g) and any("seds" in g[k] for k in g.keys())
    except OSError:
        return False


def log_phase(galaxy_dir):
    """Last milestone of a job's powderday / hyperion LOG and its age in minutes.

    A live LOG means the job runs; a stale LOG on an incomplete SED means it died.
    ``('no LOG (never started)', nan)`` without a LOG.
    """
    logs = glob.glob(os.path.join(galaxy_dir, "*.LOG"))
    if not logs:
        return "no LOG (never started)", float("nan")
    txt = open(logs[0], errors="replace").read()
    if "Total CPU time elapsed" in txt or "Ending on" in txt:
        phase = "hyperion finished"
    else:
        phase = "python front end"
        for key in ("starting raytracing iteration", "starting final iteration", "starting Lucy iteration",
                    "Adding Black Holes to Source List"):
            if key in txt:
                phase = key
                break
    return phase, (time.time() - os.path.getmtime(logs[0])) / 60.0


def source_types(rtout):
    """Counter of the hyperion source types in an RT output (stars ``point_collection``, an AGN ``point``); attributes only."""
    with h5py.File(rtout, "r") as f:
        src = f["Input/Sources"]
        out = []
        for k in src.keys():
            t = src[k].attrs.get("type", b"?")
            out.append(t.decode() if isinstance(t, bytes) else str(t))
    return collections.Counter(out)


def count_bh_particles(cutout):
    """Number of black-hole particles in a cutout (-1 when the file is missing)."""
    if not os.path.exists(cutout):
        return -1
    with h5py.File(cutout, "r") as f:
        return int(f["PartType5/BH_Mass"].shape[0]) if "PartType5" in f else 0


def rt_census(sed_out, tag, snaps):
    """Every staged galaxy of a run tree -> list of dicts ``snap, gid, exists, complete, phase, log_age_min``."""
    rows = []
    for snap in snaps:
        jdir = os.path.join(sed_out, tag, "powderday_sed_out", f"snap_{int(snap):03d}")
        for gdir in sorted(glob.glob(os.path.join(jdir, "gal_*"))):
            gid = int(os.path.basename(gdir).split("_")[1])
            p = tree_rtout_path(sed_out, tag, snap, gid)
            ph, age = log_phase(gdir)
            rows.append(dict(snap=int(snap), gid=gid, exists=os.path.exists(p), complete=sed_is_complete(p),
                             phase=ph, log_age_min=age))
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Parameter masters
# ═════════════════════════════════════════════════════════════════════════════

MASTER_KEYS = ("BH_SED", "BH_var", "dust_grid_type", "BH_model", "dusttometals_ratio", "n_photons_raytracing_dust")


def parse_master(path, keys=MASTER_KEYS):
    """The knobs of a powderday parameter master -> ``(values, source text)``.

    Values are the raw right-hand sides (quotes stripped; None when absent), plus ``incl`` (the
    Nenkova torus inclination, ``nenkova_params[2]``) and ``n_theta`` (number of sightlines).
    """
    src = open(path).read()
    v = {}
    for key in keys:
        m = re.search(rf"(?m)^{key}\s*=\s*(.+?)\s*(#.*)?$", src)
        v[key] = m.group(1).strip().strip("\"'") if m else None
    m = re.search(r"(?m)^nenkova_params\s*=\s*\[([^\]]+)\]", src)
    v["incl"] = int(float(m.group(1).split(",")[2])) if m else None
    m = re.search(r"(?m)^THETA\s*=\s*\[([^\]]+)\]", src)
    v["n_theta"] = len(m.group(1).split(",")) if m else None
    return v, src


def staged_master_values(sed_out, tag, snaps, keys=("BH_SED", "BH_model", "BH_var", "dust_grid_type")):
    """The knobs of the parameter master staged in every snapshot of a run tree, and the number of ids.

    Returns ``{snap: dict(values or None, n_ids)}``; ``values`` is None when the snapshot was not
    staged or has no ``parameters_master.py``.
    """
    out = {}
    for s in sorted(set(int(x) for x in snaps)):
        jdir = os.path.join(sed_out, tag, "powderday_sed_out", f"snap_{s:03d}")
        if not os.path.isdir(jdir):
            jdir = os.path.join(sed_out, tag, "powderday_sed_out", f"snap_{s}")
        pm = os.path.join(jdir, "parameters_master.py")
        if not os.path.exists(pm):
            out[s] = dict(values=None, n_ids=0, staged=os.path.isdir(jdir)); continue
        vals = {}
        for line in open(pm):
            m = re.match(rf"\s*({'|'.join(keys)})\s*=\s*(\S+)", line)
            if m:
                vals[m.group(1)] = m.group(2)
        ids = os.path.join(jdir, "ids.txt")
        out[s] = dict(values=vals, n_ids=sum(1 for _ in open(ids)) if os.path.exists(ids) else 0, staged=True)
    return out


def check_bh_field_names(source_text, bad_marker, expected):
    """Preflight of the gadget2pd black-hole field fix -> ``(ok, n malformed, fields found)``."""
    bad = source_text.count(bad_marker)
    good = sorted(set(re.findall(r'add_field\(\("bh","(\w+)"\)', source_text)))
    return (bad == 0 and good == sorted(expected)), bad, good


# ═════════════════════════════════════════════════════════════════════════════
# SEDs
# ═════════════════════════════════════════════════════════════════════════════

def read_rt_sed(path, aperture=-1, sightline="mean"):
    """An .rtout.sed -> ``(wavelength [µm, rest frame, sorted], nu L_nu [erg/s])``; None when missing.

    ``sightline``: ``"mean"`` over the inclinations, ``"all"`` (array ``(n_incl, n_wav)``) or an
    index (``0`` = the first, e.g. the fiducial ``findx=0``). Needs hyperion.
    """
    if path is None or not os.path.exists(path):
        return None
    from hyperion.model import ModelOutput
    sed = ModelOutput(path).get_sed(inclination="all", aperture=aperture)
    wav = np.asarray(sed.wav, float)
    val = np.atleast_2d(np.asarray(sed.val, float))
    s = np.argsort(wav)
    if sightline == "all":
        return wav[s], val[:, s]
    v = val.mean(axis=0) if sightline == "mean" else val[int(sightline) if val.shape[0] > 1 else 0]
    return wav[s], v[s]


def on_log_grid(wl, nuLnu, grid):
    """An SED interpolated in log-log onto ``grid`` (NaN outside its range)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 ** np.interp(np.log10(grid), np.log10(wl), np.log10(nuLnu), left=np.nan, right=np.nan)


def stack_seds(seds, grid):
    """Median and 16-84 % band of SEDs ``[(wl, nuLnu), ...]`` on ``grid`` -> dict ``med, lo, hi, n, curves``; None when empty."""
    curves = [on_log_grid(w, v, grid) for w, v in seds if w is not None]
    C = np.asarray(curves)
    if C.size == 0:
        return None
    return dict(med=np.nanmedian(C, axis=0), lo=np.nanpercentile(C, 16, axis=0), hi=np.nanpercentile(C, 84, axis=0),
                n=len(C), curves=C)


def sed_to_flux_density(wav_rest_um, nuLnu_erg_s, z, cosmology=None):
    """Rest-frame ``nu L_nu`` [erg/s] -> observed ``(wavelength [µm], F_nu [mJy])`` at redshift ``z`` (Planck13 by default)."""
    if cosmology is None:
        from astropy.cosmology import Planck13 as cosmology
    import astropy.units as u
    wav_obs = np.asarray(wav_rest_um, float) * (1.0 + z)
    dl = cosmology.luminosity_distance(z).to(u.cm).value
    nu_obs = 2.99792458e14 / wav_obs
    return wav_obs, np.asarray(nuLnu_erg_s, float) / (4.0 * np.pi * dl ** 2 * nu_obs) / 1e-26


def dust_opacity(bands_um, dust_file=None, fallback_kappa_v=3.0e4):
    """Extinction opacity χ [cm² per g of dust] of the RT dust model at the given wavelengths -> ``({band: χ}, source)``.

    The dust file defaults to the one the parameter masters use:
    ``$POWDERDAY_ROOT (else ~)/hyperion-dust/dust_files/kmh94_3.1_hg.hdf5``. If unreadable, a MW-like
    fallback (κ_V = ``fallback_kappa_v``, A_U/A_V = 1.55, A_J/A_V = 0.28) for the U / V / J keys.
    """
    if dust_file is None:
        dust_file = os.path.join(os.environ.get("POWDERDAY_ROOT", os.path.expanduser("~")),
                                 "hyperion-dust", "dust_files", "kmh94_3.1_hg.hdf5")
    try:
        with h5py.File(dust_file, "r") as f:
            nu = np.asarray(f["optical_properties"]["nu"][:], float)
            chi = np.asarray(f["optical_properties"]["chi"][:], float)
        lam = 2.99792458e14 / nu
        o = np.argsort(lam)
        return {b: float(np.interp(l, lam[o], chi[o])) for b, l in bands_um.items()}, os.path.basename(dust_file)
    except (OSError, KeyError):
        return {"U": 1.55 * fallback_kappa_v, "V": fallback_kappa_v, "J": 0.28 * fallback_kappa_v}, "fallback"


def grid_centres(selection_h5, snaps=None, gids=None, sim=None):
    """RT-grid centres (code units) by ``(snap, gid)``: from the Stage-1 selection file, else from caesar.

    The caesar fallback needs ``sim`` (and loads each catalogue once); the values are identical.
    """
    from .makesed import read_selection_centers
    cen = read_selection_centers(selection_h5)
    if cen or sim is None:
        return cen
    import gc
    snaps, gids = np.asarray(snaps), np.asarray(gids)
    for s in np.unique(snaps):
        cs = sim.load_catalog(snap=int(s))
        for g in np.unique(gids[snaps == s]):
            cen[(int(s), int(g))] = cs.galaxies[int(g)].pos.in_units("code_length").value
        del cs
        gc.collect()
    return cen
