"""End-to-end CIGALE (2025.0) interface for powderday flux tables.

Covers the full path from a ``MakeSED.extract_flux_batch`` table to a finished
fit without ever touching ``pcigale init / genconf`` by hand:

1. :func:`write_cigale_input` — flux table -> CIGALE ``data_file`` (FITS, mJy,
   database band names, ``<band>_err`` columns).
2. :func:`prepare_run` — writes a complete, validated ``pcigale.ini`` +
   ``pcigale.ini.spec`` into a run directory (band list auto-derived from the
   data file; module grids from :data:`DEFAULT_MODULE_PARAMS` unless
   overridden).
3. :func:`run` / :func:`check` — thin wrappers around the ``pcigale``
   executable (results land in ``<run_dir>/out/``).
4. :func:`describe_run` — print the genconf-style reminder of every module,
   parameter and analysis variable (the same comments a hand-made
   ``pcigale.ini`` carries; :func:`prepare_run` also writes them into the ini).
5. :func:`compare_results` — truth table vs the fitted ``out/results.fits``:
   per-galaxy print, offset/scatter stats, one-to-one figures; table + figure
   saved into that run's ``out/``.
6. :func:`plot_parameter_priors` — for every gridded parameter, the fitted
   distribution over the sample against the grid nodes (the priors), with
   printed grid-refinement advice (extend / refine / trim); saved into
   that run's ``out/``.

Only needs numpy + astropy (matplotlib just for :func:`compare_results`);
``pcigale`` itself is required just where :func:`run`/:func:`check` execute.
The config layout and the filter names are pinned to **CIGALE 2025.0**
(verified against its source); other versions may need
:data:`MODULE_REGISTRY` / the band sets updated.
"""

import os
import re
import subprocess
import textwrap
import warnings

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck13

__all__ = [
    "cigale_band", "write_cigale_input", "prepare_run", "describe_run",
    "check", "run", "read_results", "plot_seds", "compare_results",
    "plot_parameter_priors",
    "DEFAULT_SED_MODULES", "DEFAULT_MODULE_PARAMS",
    "DEFAULT_ANALYSIS_PARAMS", "CIGALE_KNOWN_BANDS",
]


# ══════════════════════════════════════════════════════════════════════════
# band-name mapping: '<Facility>.<Instrument>.<filter>' -> CIGALE names
# (sets verified against pcigale/data/filters; 2025.1 splits WFC3 UVIS into
#  per-chip uvis1/uvis2 — we map to uvis1, the primary/reference CCD)
# ══════════════════════════════════════════════════════════════════════════

WFC3_IR_FILTERS = {
    "F098M", "F105W", "F110W", "F125W", "F126N", "F127M", "F128N", "F130N",
    "F132N", "F139M", "F140W", "F153M", "F160W", "F164N", "F167N",
}
WFC3_UVIS_FILTERS = {
    "F218W", "F225W", "F275W", "F280N", "F336W", "F343N", "F373N", "F390M",
    "F390W", "F395N", "F410M", "F438W", "F467M", "F469N", "F475W", "F487N",
    "F502N", "F547M", "F555W", "F606W", "F621M", "F625W", "F631N", "F645N",
    "F656N", "F657N", "F658N", "F665N", "F673N", "F680N", "F689M", "F763M",
    "F775W", "F814W", "F845M", "F953N",
}
NIRCAM_FILTERS = {
    "F070W", "F090W", "F115W", "F140M", "F150W", "F150W2", "F162M", "F164N",
    "F182M", "F187N", "F200W", "F210M", "F212N", "F250M", "F277W", "F300M",
    "F322W2", "F323N", "F335M", "F356W", "F360M", "F405N", "F410M", "F430M",
    "F444W", "F460M", "F466N", "F470N", "F480M",
}
MIRI_FILTERS = {
    "F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W",
    "F2100W", "F2550W",
}
HSC_FILTERS = {"g", "r", "i", "z", "Y"}
MEGACAM_FILTERS = {"u", "u_1", "g", "g_1", "r", "r_1", "i", "i_1", "i_2",
                   "z", "z_1"}
# SVO lists VISTA under Paranal/VIRCAM; CIGALE names them paranal.vircam.*
VIRCAM_FILTERS = {"Z", "Y", "J", "H", "Ks", "NB118", "NB980", "NB990"}
SCUBA2_FILTERS = {"450GHz", "850GHz"}
CIGALE_KNOWN_BANDS = (
    {
        "2mass.J", "generic.johnson.U", "generic.johnson.V",
        "spitzer.irac.I1", "spitzer.irac.I2", "spitzer.irac.I3",
        "spitzer.irac.I4",
        "spitzer.mips.24mu", "spitzer.mips.70mu", "spitzer.mips.160mu",
        "herschel.pacs.blue", "herschel.pacs.green", "herschel.pacs.red",
        "herschel.spire.PSW", "herschel.spire.PMW", "herschel.spire.PLW",
        "herschel.spire.PSW_ext", "herschel.spire.PMW_ext",
        "herschel.spire.PLW_ext",
        # NOT in the stock 2025 DB — custom filter, register once per env:
        #   pcigale-filters add ALMA_band6_cigale.dat   (repo root)
        "alma.band6",
    }
    | {f"hst.wfc3.ir.{f}" for f in WFC3_IR_FILTERS}
    | {f"hst.wfc3.uvis1.{f}" for f in WFC3_UVIS_FILTERS}
    | {f"jwst.nircam.{f}" for f in NIRCAM_FILTERS}
    | {f"jwst.miri.{f}" for f in MIRI_FILTERS}
    | {f"subaru.hsc.{f}" for f in HSC_FILTERS}
    | {f"cfht.megacam.{f}" for f in MEGACAM_FILTERS}
    | {f"paranal.vircam.{f}" for f in VIRCAM_FILTERS}
    | {f"jcmt.scuba2.{f}" for f in SCUBA2_FILTERS}
)

# local filter files used by the notebooks -> CIGALE names
CIGALE_OVERRIDES = {
    "2MASS.J.J": "2mass.J",
    "Johnson.V.V": "generic.johnson.V",
    "Johnson2.U.U": "generic.johnson.U",
    "ALMA.ALMA.band6": "alma.band6",
}


def cigale_band(col):
    """Map an extract_flux_batch column name to a CIGALE 2025.0 band name.

    Columns are ``'<Facility>.<Instrument>.<filter>'``. Returns None for
    bands with no CIGALE counterpart (grisms, quad filters, ...). SVO lumps
    WFC3 UVIS+IR under one instrument while CIGALE splits them, hence the
    explicit routing.
    """
    if col in CIGALE_OVERRIDES:
        return CIGALE_OVERRIDES[col]
    try:
        fac, inst, filt = col.split(".", 2)
    except ValueError:
        return None
    if fac == "HST" and inst == "WFC3":
        sub = "ir" if filt in WFC3_IR_FILTERS else "uvis1"
        name = f"hst.wfc3.{sub}.{filt}"
    else:
        name = f"{fac.lower()}.{inst.lower()}.{filt}"
    return name if name in CIGALE_KNOWN_BANDS else None


# ══════════════════════════════════════════════════════════════════════════
# input catalog
# ══════════════════════════════════════════════════════════════════════════

def write_cigale_input(flux_table, outpath, err_floor=0.0, verbose=True):
    """Convert an ``extract_flux_batch`` table into a CIGALE ``data_file``.

    Parameters
    ----------
    flux_table : str or :class:`~astropy.table.Table`
        Table (or path to the FITS) produced by
        ``MakeSED.extract_flux_batch(..., redshift=True)``. CIGALE compares
        redshifted models to the photometry, so the fluxes MUST be
        observed-frame (``redshift=True``); rest-frame tables give
        inconsistent fits.
    outpath : str
        Destination FITS file.
    err_floor : float
        Fractional error added in quadrature to the Monte-Carlo error.
        Leave at 0: CIGALE itself adds ``additionalerror`` (10% by default,
        see :func:`prepare_run`) at fit time, so a floor here would be
        double-counted.
    verbose : bool
        Print a per-file summary.

    Returns
    -------
    str : *outpath*

    Notes
    -----
    Output columns: ``id`` (``snapNNN_galID``), ``redshift``, ``distance``
    (Mpc, Planck13 — the cosmology used to normalize the fluxes; shipped
    explicitly so CIGALE does not re-derive D_L with Planck 2018), then per
    band the flux in mJy and its ``<band>_err``. Missing fluxes are NaN
    (CIGALE's missing-data convention); set an error negative by hand to
    treat a band as an upper limit.
    """
    t = Table.read(flux_table) if isinstance(flux_table, str) else flux_table
    if len(t) == 0:
        raise ValueError("empty flux table")

    out = Table()
    out["id"] = np.array([f"snap{int(s):03d}_gal{int(g)}"
                          for s, g in zip(t["snap"], t["gal_id_at_snap"])])
    out["redshift"] = np.asarray(t["redshift"], float)
    out["distance"] = Planck13.luminosity_distance(
        out["redshift"]).to("Mpc").value

    dropped, n_neg = [], 0
    for col in t.colnames:
        if col in ("gal_id_at_snap", "snap", "redshift") or col.endswith("_err"):
            continue
        band = cigale_band(col)
        if band is None:
            dropped.append(col)
            continue
        flux = np.asarray(t[col], float)
        ecol = f"{col}_err"
        err = (np.asarray(t[ecol], float) if ecol in t.colnames
               else np.full(len(t), np.nan))
        # a negative error means 'upper limit' to CIGALE — only ever set that
        # by hand on the OUTPUT file. MC errors are magnitudes; older
        # convolveFilterWithSED builds flipped their sign (descending-λ SEDs)
        n_neg += int((err < 0).sum())
        err = np.abs(err)
        if err_floor > 0:
            floor = err_floor * np.abs(flux)
            err = np.where(np.isfinite(err), np.hypot(err, floor), floor)
        err[~np.isfinite(flux)] = np.nan
        out[band] = flux
        out[f"{band}_err"] = err

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    out.write(outpath, overwrite=True)
    if verbose:
        nband = (len(out.colnames) - 3) // 2
        print(f"[cigale] {len(out)} objects, {nband} bands -> {outpath}")
        if n_neg:
            print(f"[cigale]   {n_neg} negative input error(s) -> abs() "
                  "(negative = upper limit to CIGALE; set those by hand on "
                  "the output)")
        if dropped:
            print(f"[cigale]   dropped (not in the CIGALE 2025.0 DB): "
                  f"{sorted(set(dropped))}")
    return outpath


def grid_options(module, param):
    """Allowed values of a strict-grid CIGALE parameter, from the registry.

    Parses the ``options=a & b & ...`` clause of the configobj type string in
    :data:`MODULE_REGISTRY` (verbatim from the CIGALE sources), e.g.
    ``grid_options('bc03', 'metallicity')`` ->
    ``[0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]``. Raises for parameters
    without a fixed option list.
    """
    typ = MODULE_REGISTRY[module][param][0]
    m = re.search(r"options=([^)]*)", typ)
    if not m:
        raise ValueError(f"{module}.{param} has no fixed option list "
                         f"(type: {typ})")
    return [float(t) for t in m.group(1).split("&")]


def nearest_option(values, options, log=True):
    """Snap *values* to the nearest allowed grid *options* (NaN stays NaN).

    With ``log=True`` (default) 'nearest' is measured in log space — the
    right metric for metallicity-like grids that are geometrically spaced.
    """
    values = np.atleast_1d(np.asarray(values, float))
    opts = np.sort(np.asarray(options, float))
    space = np.log10(opts) if log else opts
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values) & ((values > 0) if log else True)
    if ok.any():
        v = np.log10(values[ok]) if log else values[ok]
        out[ok] = opts[np.argmin(np.abs(v[:, None] - space[None, :]),
                                 axis=1)]
    return out


def split_by_metallicity(data_file, zstar, zgas=None, outdir=None,
                         n_zgas_max=3, verbose=True):
    """Split a CIGALE data file into per-metallicity groups (strict priors).

    CIGALE cannot pin a grid parameter per object, so a per-galaxy
    metallicity prior needs one sub-catalog + run per metallicity group.
    Each object's SIMBA stellar metallicity is snapped to the nearest
    allowed bc03 ``metallicity`` (log-space nearest); objects sharing a node
    form a group fitted with that **single** stellar Z. Groups are keyed by
    the stellar node only — the nebular ``zgas`` grid is dense (26 nodes),
    so keying on it too would fragment the sample into ~per-object runs,
    each recomputing the full model grid (more total compute than no split).
    Instead each group's ``zgas`` grid is restricted to at most
    *n_zgas_max* values picked at quantiles of the members' snapped zgas
    distribution (re-snapped to the grid) — a range prior that cannot
    balloon the model grid: the 26-node zgas grid is dense enough that a
    large group can otherwise collect 10+ distinct values, multiplying the
    grid by that factor (the 2026-07-09 SIGBUS: 14 zgas x 10^4 SFH x 120
    dust x 5 z = 84M models in one block).

    Parameters
    ----------
    data_file : str
        A :func:`write_cigale_input` catalog.
    zstar : dict
        ``{id: Z}`` — linear stellar metallicity (mass fraction) per object
        id (``snapNNN_galID``). Objects missing from the dict (or with
        non-finite Z) go to a ``Zsfree`` group fitted with the default
        metallicity grid.
    zgas : dict, optional
        Same for the nebular gas metallicity. Omitted -> only bc03 is
        constrained.
    n_zgas_max : int
        Maximum number of zgas grid values per group (default 3:
        min/median/max quantiles of the members' snapped distribution).
    outdir : str, optional
        Where the sub-catalogs are written. Default: ``<dirname>/zbinned/``
        (a subdirectory, so the originals' glob does not re-match them).

    Returns
    -------
    list of dict
        One entry per group: ``{'tag', 'path', 'n', 'module_params'}`` —
        *tag* like ``'Zs0p008'`` (or ``'Zsfree'``), *path* the sub-catalog,
        *module_params* the override dict (``None`` for the free group) to
        merge into :func:`prepare_run`'s ``module_params``.
    """
    t = Table.read(data_file)
    ids = np.array([str(i) for i in t["id"]])
    zs_grid = grid_options("bc03", "metallicity")
    zg_grid = grid_options("nebular", "zgas")

    zs = nearest_option([zstar.get(i, np.nan) for i in ids], zs_grid)
    zg = (nearest_option([zgas.get(i, np.nan) for i in ids], zg_grid)
          if zgas is not None else np.full(len(ids), np.nan))

    def _fmt_z(v):
        return f"{v:g}".replace(".", "p")

    groups = {}
    for k in range(len(ids)):
        key = zs[k] if np.isfinite(zs[k]) else "free"
        groups.setdefault(key, []).append(k)

    if outdir is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(data_file)),
                              "zbinned")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(data_file))[0]

    out = []
    for kzs, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if kzs == "free":
            gtag, mp = "Zsfree", None
        else:
            gtag = f"Zs{_fmt_z(kzs)}"
            mp = {"bc03": {"metallicity": [kzs]}}
            zgm = zg[rows]
            zgm = zgm[np.isfinite(zgm)]
            if zgm.size:
                zg_vals = sorted(set(zgm.tolist()))
                if len(zg_vals) > n_zgas_max:
                    picks = np.quantile(zgm, np.linspace(0, 1, n_zgas_max))
                    zg_vals = sorted(set(
                        nearest_option(picks, zg_grid).tolist()))
                mp["nebular"] = {"zgas": zg_vals}
        path = os.path.join(outdir, f"{stem}_{gtag}.fits")
        t[np.array(rows)].write(path, overwrite=True)
        out.append({"tag": gtag, "path": path, "n": len(rows),
                    "module_params": mp})
    if verbose:
        n_free = sum(g["n"] for g in out if g["module_params"] is None)
        print(f"[cigale] {stem}: {len(t)} objects -> {len(out)} metallicity "
              f"group(s) " + ", ".join(f"{g['tag']}({g['n']})" for g in out)
              + (f" — {n_free} without a Z prior fit the default grid"
                 if n_free else ""))
    return out


# ══════════════════════════════════════════════════════════════════════════
# pcigale.ini generation (replaces `pcigale init` + `pcigale genconf`)
# ══════════════════════════════════════════════════════════════════════════

# (configobj type string, default) per parameter — verbatim from the CIGALE
# 2025.0 sources, so the generated .spec validates/converts exactly like one
# written by genconf. Add entries here to enable other modules.
MODULE_REGISTRY = {
    "sfhdelayedbq": {
        "tau_main": ("cigale_list()", 2000.0),
        "age_main": ("cigale_list(dtype=int, minvalue=0.)", 5000),
        "age_bq": ("cigale_list(dtype=int)", 500),
        "r_sfr": ("cigale_list(minvalue=0.)", 0.1),
        "sfr_A": ("cigale_list(minvalue=0.)", 1.0),
        "normalise": ("boolean()", True),
    },
    "sfhdelayed": {
        "tau_main": ("cigale_list()", 2000.0),
        "age_main": ("cigale_list(dtype=int, minvalue=0.)", 5000),
        "tau_burst": ("cigale_list()", 50.0),
        "age_burst": ("cigale_list(dtype=int, minvalue=1.)", 20),
        "f_burst": ("cigale_list(minvalue=0., maxvalue=0.9999)", 0.0),
        "sfr_A": ("cigale_list(minvalue=0.)", 1.0),
        "normalise": ("boolean()", True),
    },
    "bc03": {
        "imf": ("cigale_list(dtype=int, options=0. & 1.)", 0),
        "metallicity": ("cigale_list(options=0.0001 & 0.0004 & 0.004 & "
                        "0.008 & 0.02 & 0.05)", 0.02),
        "separation_age": ("cigale_list(dtype=int, minvalue=0)", 10),
    },
    "nebular": {
        "logU": ("cigale_list(options=-4.0 & -3.9 & -3.8 & -3.7 & -3.6 & "
                 "-3.5 & -3.4 & -3.3 & -3.2 & -3.1 & -3.0 & -2.9 & -2.8 & "
                 "-2.7 & -2.6 & -2.5 & -2.4 & -2.3 & -2.2 & -2.1 & -2.0 & "
                 "-1.9 & -1.8 & -1.7 & -1.6 & -1.5 & -1.4 & -1.3 & -1.2 & "
                 "-1.1 & -1.0)", -2.0),
        "zgas": ("cigale_list(options=0.0001 & 0.0004 & 0.001 & 0.002 & "
                 "0.0025 & 0.003 & 0.004 & 0.005 & 0.006 & 0.007 & 0.008 & "
                 "0.009 & 0.011 & 0.012 & 0.014 & 0.016 & 0.019 & 0.020 & "
                 "0.022 & 0.025 & 0.03 & 0.033 & 0.037 & 0.041 & 0.046 & "
                 "0.051)", 0.02),
        "ne": ("cigale_list(options=10 & 100 & 1000)", 100),
        "f_esc": ("cigale_list(minvalue=0., maxvalue=1.)", 0.0),
        "f_dust": ("cigale_list(minvalue=0., maxvalue=1.)", 0.0),
        "lines_width": ("cigale_list(minvalue=0.)", 300.0),
        "emission": ("boolean()", True),
        "line_list": ("string()", ""),   # auto-filled by CIGALE at run time
    },
    "dustatt_modified_CF00": {
        "Av_ISM": ("cigale_list(minvalue=0)", 1.0),
        "mu": ("cigale_list(minvalue=.0001, maxvalue=1.)", 0.44),
        "slope_ISM": ("cigale_list()", -0.7),
        "slope_BC": ("cigale_list()", -1.3),
        "filters": ("string()", "generic.bessell.B & generic.bessell.V"),
    },
    "dl2014": {
        "qpah": ("cigale_list(minvalue=0.47, maxvalue=7.32)", 2.5),
        "umin": ("cigale_list(options=0.10 & 0.12 & 0.15 & 0.17 & 0.20 & "
                 "0.25 & 0.30 & 0.35 & 0.40 & 0.50 & 0.60 & 0.70 & 0.80 & "
                 "1.00 & 1.20 & 1.50 & 1.70 & 2.00 & 2.50 & 3.00 & 3.50 & "
                 "4.00 & 5.00 & 6.00 & 7.00 & 8.00 & 10.00 & 12.00 & 15.00 & "
                 "17.00 & 20.00 & 25.00 & 30.00 & 35.00 & 40.00 & 50.00)",
                 1.0),
        "alpha": ("cigale_list(options=1.0 & 1.1 & 1.2 & 1.3 & 1.4 & 1.5 & "
                  "1.6 & 1.7 & 1.8 & 1.9 & 2.0 & 2.1 & 2.2 & 2.3 & 2.4 & "
                  "2.5 & 2.6 & 2.7 & 2.8 & 2.9 & 3.0)", 2.0),
        "gamma": ("cigale_list(minvalue=0., maxvalue=1.)", 0.1),
    },
    "restframe_parameters": {
        # booleans/strings, NOT grid dimensions (no model-count growth).
        # NOTE 2025.1 renamed/reworked these vs older CIGALEs: 'Dn4000' (was
        # D4000), 'EW' (was EW_lines; label + blue/line/red sideband bounds),
        # and filters use the dotted DB names (galex.FUV, not FUV).
        "beta_calz94": ("boolean()", False),
        "Dn4000": ("boolean()", False),
        "IRX": ("boolean()", False),
        "EW": ("string()",
               "HdeltaA/404.160/407.975/408.350/412.225/412.850/416.100 & "
               "Halpha/650.0/652.5/653.5/660.0/661.0/663.5"),
        "luminosity_filters": ("string()", "galex.FUV & generic.bessell.V"),
        "colours_filters": ("string()",
                            "galex.FUV-galex.NUV & galex.NUV-sloan.sdss.r"),
    },
    "redshifting": {
        # empty -> CIGALE fills the grid from the data-file redshifts at run
        "redshift": ("cigale_list(minvalue=0.)", ""),
    },
}

ANALYSIS_REGISTRY = {
    "pdf_analysis": {
        "variables": ("cigale_string_list()", []),
        "bands": ("cigale_string_list()", []),   # auto-filled from data_file
        "save_best_sed": ("boolean()", False),
        "save_chi2": ("option('all', 'none', 'properties', 'fluxes')",
                      "none"),
        "lim_flag": ("option('full', 'noscaling', 'none')", "noscaling"),
        "mock_flag": ("boolean()", False),
        "redshift_decimals": ("integer()", 2),
        "blocks": ("integer(min=1)", 1),
    },
}

# one-line descriptions, condensed from the comments `pcigale genconf` writes
# into pcigale.ini (CIGALE 2025.0). prepare_run writes them back into the ini
# and describe_run prints them as an in-notebook reminder.
TOP_DOCS = {
    "data_file": "input catalog: id, redshift[, distance Mpc], fluxes in mJy "
                 "+ <band>_err (NaN = missing, negative error = upper limit)",
    "parameters_file": "optional explicit list of models to compute; leave "
                       "empty to build the grid from the module parameters "
                       "below",
    "sed_modules": "SED-building chain, order matters: SFH -> SSP -> nebular "
                   "-> attenuation -> dust emission -> redshifting (last)",
    "analysis_method": "pdf_analysis: Bayesian (posterior mean+std) and "
                       "best-chi2 estimates of the physical properties",
    "cores": "number of CPU cores used to build the grid and fit",
    "bands": "bands (+ _err) of the data file used in the fit",
    "properties": "intensive/extensive physical properties to fit alongside "
                  "the fluxes (rarely needed)",
    "additionalerror": "relative error added in quadrature to the observed "
                       "flux errors at fit time (CIGALE default 0.1)",
}

MODULE_DOCS = {
    "sfhdelayedbq": "delayed SFH ~ t*exp(-t/tau) with a burst/quench episode: "
                    "at age_bq the SFR jumps to r_sfr x SFR(age_bq) and stays "
                    "constant",
    "sfhdelayed": "delayed SFH ~ t*exp(-t/tau), optionally with a late "
                  "exponential burst",
    "bc03": "Bruzual & Charlot (2003) single stellar populations",
    "nebular": "nebular emission (lines + continuum) from the absorbed "
               "Lyman-continuum photons",
    "dustatt_modified_CF00": "Charlot & Fall (2000) two-phase attenuation "
                             "(birth clouds + ISM) with free power-law slopes",
    "dl2014": "Draine & Li (2014) dust emission templates",
    "restframe_parameters": "rest-frame quantities computed on each model "
                            "right before redshifting: Dn4000, UV slopes, "
                            "IRX, line EWs, rest-frame luminosities/colours "
                            "(outputs: param.*)",
    "redshifting": "redshift the SED + IGM absorption (must be last)",
    "pdf_analysis": "Bayesian (bayes.* = posterior mean+std) and best-fit "
                    "(best.*) estimates for each property",
}

PARAM_DOCS = {
    "sfhdelayedbq": {
        "tau_main": "e-folding time of the main stellar population [Myr]",
        "age_main": "age of the main stellar population [Myr]",
        "age_bq": "age of the burst/quench episode [Myr before observation]",
        "r_sfr": "ratio of the SFR after / before age_bq (0 = full quench, "
                 ">1 = burst)",
        "sfr_A": "multiplicative SFR factor, used only if normalise is False",
        "normalise": "normalise the SFH to produce one solar mass",
    },
    "sfhdelayed": {
        "tau_main": "e-folding time of the main stellar population [Myr]",
        "age_main": "age of the main stellar population [Myr]",
        "tau_burst": "e-folding time of the late starburst population [Myr]",
        "age_burst": "age of the late burst [Myr]",
        "f_burst": "mass fraction of the late burst population (0 = none)",
        "sfr_A": "multiplicative SFR factor, used only if normalise is False",
        "normalise": "normalise the SFH to produce one solar mass",
    },
    "bc03": {
        "imf": "initial mass function: 0 = Salpeter, 1 = Chabrier",
        "metallicity": "stellar metallicity Z (0.02 = solar); options "
                       "0.0001/0.0004/0.004/0.008/0.02/0.05",
        "separation_age": "age [Myr] separating the young and old stellar "
                          "populations (0 = no separation)",
    },
    "nebular": {
        "logU": "ionisation parameter",
        "zgas": "gas metallicity",
        "ne": "electron density [cm^-3]",
        "f_esc": "fraction of Lyman-continuum photons escaping the galaxy",
        "f_dust": "fraction of Lyman-continuum photons absorbed by dust",
        "lines_width": "line width [km/s]",
        "emission": "include nebular emission (lines + continuum)",
        "line_list": "emission lines to consider (auto-filled by CIGALE at "
                     "run time)",
    },
    "dustatt_modified_CF00": {
        "Av_ISM": "V-band attenuation of the ISM component [mag]",
        "mu": "Av_ISM / (Av_BC + Av_ISM): ISM fraction of the total V-band "
              "attenuation (CF00 = 0.44)",
        "slope_ISM": "power-law slope of the ISM attenuation curve",
        "slope_BC": "power-law slope of the birth-cloud attenuation curve",
        "filters": "filters for which the attenuation is stored in the SED "
                   "info ('&'-separated)",
    },
    "dl2014": {
        "qpah": "PAH mass fraction [%]",
        "umin": "minimum radiation field U_min",
        "alpha": "power-law slope dU/dM ~ U^alpha of the radiation-field "
                 "distribution",
        "gamma": "dust-mass fraction illuminated from U_min to U_max (the "
                 "rest sits at U_min)",
    },
    "restframe_parameters": {
        "beta_calz94": "observed + intrinsic UV slopes beta/beta0 "
                       "(Calzetti et al. 1994) -> param.beta_calz94/"
                       "param.beta0_calz94",
        "Dn4000": "Dn4000 break, Balogh et al. (1999) definition -> "
                  "param.Dn4000",
        "IRX": "log10(L_dust/L_FUV) -> param.IRX",
        "EW": "line equivalent widths [nm]: label/blue_lo/blue_hi/line_lo/"
              "line_hi/red_lo/red_hi per line, '&'-separated (absorption "
              "positive, emission negative) -> param.EW(label)",
        "luminosity_filters": "'&'-separated DB filter names -> "
                              "param.restframe_Lnu(filter) [W/Hz]",
        "colours_filters": "'&'-separated 'f1-f2' rest-frame colours -> "
                           "param.restframe_f1-f2 [mag]",
    },
    "redshifting": {
        "redshift": "redshift grid; leave empty to take the redshifts from "
                    "the data file",
    },
    "pdf_analysis": {
        "variables": "physical properties to estimate (bayes.* + best.*); "
                     "empty = all (slow on big grids)",
        "bands": "bands whose fluxes are also estimated (independent of the "
                 "bands actually fitted)",
        "save_best_sed": "save each object's best model "
                         "(<id>_best_model.fits; needed by pcigale-plots sed)",
        "save_chi2": "save the raw chi2: 'all'/'none'/'properties'/'fluxes' "
                     "(~15 MB per 1e6 models per variable)",
        "lim_flag": "upper-limit handling: 'full' (exact), 'noscaling' "
                    "(recommended), 'none' (drop those bands)",
        "mock_flag": "also build + analyse a mock catalog (accuracy "
                     "self-test)",
        "redshift_decimals": "decimals kept when rounding the data-file "
                             "redshifts into the model grid (negative = no "
                             "rounding)",
        "blocks": "number of model blocks (keep 1 if memory allows)",
    },
}

# defaults tuned for the quenched-m25 sample: delayed SFH with a burst/quench
# episode (sfhdelayedbq), Chabrier IMF (SIMBA), two-phase CF00 attenuation,
# DL2014 dust emission. ~126k models per redshift with these grids.
DEFAULT_SED_MODULES = ("sfhdelayedbq", "bc03", "nebular",
                       "dustatt_modified_CF00", "dl2014",
                       "restframe_parameters", "redshifting")

DEFAULT_MODULE_PARAMS = {
    "sfhdelayedbq": {
        "tau_main": [500, 1000, 2000, 4000],
        "age_main": [2000, 4000, 6000, 8000, 10000],
        "age_bq": [100, 300, 500, 1000, 2000],       # Myr before observation
        "r_sfr": [0.0, 0.02, 0.05, 0.1, 0.2],        # 0 = full quench
    },
    "bc03": {"imf": 1, "metallicity": [0.008, 0.02, 0.05]},
    "dustatt_modified_CF00": {
        "Av_ISM": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
    },
    "dl2014": {"qpah": [0.47, 2.5], "umin": [1.0, 5.0, 10.0],
               "gamma": [0.02, 0.1]},
    # rest-frame quantities: Dn4000 + HdeltaA/Halpha EWs (quenched/PSB
    # diagnostics), UVJ + FUV-NUV + NUV-r rest colours, L_FUV/L_V
    "restframe_parameters": {
        "Dn4000": True,
        "colours_filters": ("generic.johnson.U-generic.johnson.V & "
                            "generic.johnson.V-generic.johnson.J & "
                            "galex.FUV-galex.NUV & galex.NUV-sloan.sdss.r"),
    },
}

DEFAULT_ANALYSIS_PARAMS = {
    "variables": ["stellar.m_star", "stellar.metallicity",
                  "sfh.sfr", "sfh.sfr10Myrs", "sfh.sfr100Myrs",
                  "sfh.tau_main", "sfh.age_main", "sfh.age_bq", "sfh.r_sfr",
                  "stellar.age_m_star", "attenuation.Av_ISM",
                  "dust.luminosity", "dust.mass",
                  "param.Dn4000", "param.EW(HdeltaA)", "param.EW(Halpha)",
                  "param.restframe_Lnu(galex.FUV)",
                  "param.restframe_Lnu(generic.bessell.V)",
                  "param.restframe_generic.johnson.U-generic.johnson.V",
                  "param.restframe_generic.johnson.V-generic.johnson.J",
                  "param.restframe_galex.FUV-galex.NUV",
                  "param.restframe_galex.NUV-sloan.sdss.r"],
    "save_best_sed": True,
}


def _fmt(value):
    """Format a python value the way configobj writes it in pcigale.ini."""
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [_fmt(v) for v in value]
        if len(vals) == 0:
            return ""
        if len(vals) == 1:
            return f"{vals[0]},"
        return ", ".join(vals)
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    s = str(value)
    if isinstance(value, str) and ("," in s or s != s.strip()):
        # configobj splits unquoted comma-containing values into lists
        # (bites e.g. gvfs paths: 'sftp:host=...,user=...'); quote them
        if '"' in s:
            raise ValueError(f"cannot represent value with both a comma and "
                             f"a double quote in pcigale.ini: {s!r}")
        s = f'"{s}"'
    return s


def _merged_params(sed_modules, module_params, analysis_params):
    """Validate + merge overrides over DEFAULT_* over the registry defaults.

    Returns ``(sed_modules, mp, ap)`` with *mp* ``{module: {param: value}}``
    complete for every registered parameter and *ap* the pdf_analysis dict.
    """
    if sed_modules is None:
        sed_modules = DEFAULT_SED_MODULES
    unknown = [m for m in sed_modules if m not in MODULE_REGISTRY]
    if unknown:
        raise KeyError(
            f"module(s) {unknown} not in MODULE_REGISTRY — add their "
            "(type-string, default) parameter entries (see the CIGALE "
            "sed_modules sources) to use them")
    if "redshifting" not in sed_modules:
        raise ValueError("'redshifting' must be the last SED module")

    mp = {mod: dict(DEFAULT_MODULE_PARAMS.get(mod, {}))
          for mod in sed_modules}
    for mod, over in (module_params or {}).items():
        if mod not in mp:
            raise KeyError(f"module_params for '{mod}' but it is not in "
                           f"sed_modules {tuple(sed_modules)}")
        bad = set(over) - set(MODULE_REGISTRY[mod])
        if bad:
            raise KeyError(f"unknown parameter(s) {sorted(bad)} for module "
                           f"'{mod}' (known: {sorted(MODULE_REGISTRY[mod])})")
        mp[mod].update(over)
    for mod in sed_modules:   # fill the registry defaults
        for par, (typ, default) in MODULE_REGISTRY[mod].items():
            mp[mod].setdefault(par, default)

    ap = dict(DEFAULT_ANALYSIS_PARAMS)
    ap.update(analysis_params or {})
    bad = set(ap) - set(ANALYSIS_REGISTRY["pdf_analysis"])
    if bad:
        raise KeyError(f"unknown analysis_params {sorted(bad)} "
                       f"(known: {sorted(ANALYSIS_REGISTRY['pdf_analysis'])})")
    for par, (typ, default) in ANALYSIS_REGISTRY["pdf_analysis"].items():
        ap.setdefault(par, default)
    return tuple(sed_modules), mp, ap


def _grid_size(sed_modules, mp):
    """Number of models per redshift implied by the grid parameters."""
    nmod = 1
    for mod in sed_modules:
        for par, (typ, _default) in MODULE_REGISTRY[mod].items():
            if not typ.startswith("cigale_list"):
                continue
            v = mp[mod].get(par)
            if v is None:
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                nmod *= max(len(v), 1)
            elif isinstance(v, str) and "," in v:
                nmod *= max(len([x for x in v.split(",") if x.strip()]), 1)
    return nmod


def prepare_run(run_dir, data_file, sed_modules=None, module_params=None,
                analysis_params=None, cores=None, additional_error=0.1,
                properties=(), max_block_models=5_000_000, verbose=True):
    """Write a complete ``pcigale.ini`` + ``pcigale.ini.spec`` into *run_dir*.

    Equivalent to ``pcigale init`` + ``pcigale genconf`` + hand-editing, in
    one deterministic step. After this, ``run(run_dir)`` (or ``pcigale run``
    inside *run_dir*) is all that is left.

    Parameters
    ----------
    run_dir : str
        Directory for this fit (created if needed); CIGALE writes its
        results to ``<run_dir>/out/``. Use one directory per data file.
    data_file : str
        CIGALE input catalog from :func:`write_cigale_input`. Band columns
        are read from it to fill the ``bands`` entry — no hand-typed lists.
    sed_modules : sequence of str, optional
        Module chain, default :data:`DEFAULT_SED_MODULES`. Every module must
        exist in :data:`MODULE_REGISTRY` (add new ones there with their
        configobj type strings).
    module_params : dict, optional
        ``{module: {param: value}}`` overrides, merged over
        :data:`DEFAULT_MODULE_PARAMS` (which is merged over the CIGALE
        defaults). Pass lists for grid dimensions.
    analysis_params : dict, optional
        Overrides for the pdf_analysis section, merged over
        :data:`DEFAULT_ANALYSIS_PARAMS`.
    cores : int, optional
        Default: all available.
    additional_error : float
        CIGALE's ``additionalerror``: relative error added in quadrature to
        the flux uncertainties at fit time (CIGALE default 0.1 — this is why
        :func:`write_cigale_input` does not apply its own floor).
    properties : sequence of str
        Intensive/extensive properties to fit (rarely needed here).
    max_block_models : int
        Unless ``analysis_params`` sets ``blocks`` explicitly, the CIGALE
        ``blocks`` count is chosen automatically as
        ``ceil(n_z * n_models / max_block_models)``. CIGALE builds each
        block's model fluxes in shared memory (``/dev/shm``, counted
        against the SLURM job cgroup — overrunning it dies with SIGBUS,
        not a clean OOM): ~93 bands x 8 B = ~750 B/model, so the default
        5M-model blocks stay under ~4 GB.

    Returns
    -------
    str : path to the written ``pcigale.ini``.
    """
    sed_modules, mp, ap = _merged_params(sed_modules, module_params,
                                         analysis_params)

    data_file = os.path.abspath(data_file)
    obs = Table.read(data_file)
    bands = [c for c in obs.colnames
             if c not in ("id", "redshift", "distance")
             and not c.endswith("_err")]
    if not bands:
        raise ValueError(f"no band columns found in {data_file}")
    band_entries = []
    for b in bands:
        band_entries.append(b)
        if f"{b}_err" in obs.colnames:
            band_entries.append(f"{b}_err")

    if cores is None:
        cores = os.cpu_count() or 1

    ap["bands"] = bands   # fluxes to predict; independent of the fit itself

    # auto-chunk the model grid unless the caller pinned `blocks` explicitly
    n_z = len(np.unique(np.round(np.asarray(obs["redshift"], float),
                                 int(ap.get("redshift_decimals", 2)))))
    n_models = _grid_size(sed_modules, mp)
    if not (analysis_params and "blocks" in analysis_params):
        ap["blocks"] = max(1, int(np.ceil(n_z * n_models
                                          / float(max_block_models))))

    ini, spec = [], []

    def comment(text, indent=0):
        pad = "  " * indent
        ini.extend(textwrap.wrap(text, width=78,
                                 initial_indent=pad + "# ",
                                 subsequent_indent=pad + "# "))

    def emit(key, value, typ, indent=0, doc=None):
        pad = "  " * indent
        if doc:
            comment(doc, indent)
        ini.append(f"{pad}{key} = {_fmt(value)}")
        spec.append(f"{pad}{key} = {typ}")

    emit("data_file", data_file, "string()", doc=TOP_DOCS["data_file"])
    emit("parameters_file", "", "string()", doc=TOP_DOCS["parameters_file"])
    emit("sed_modules", list(sed_modules), "cigale_string_list()",
         doc=TOP_DOCS["sed_modules"])
    emit("analysis_method", "pdf_analysis", "string()",
         doc=TOP_DOCS["analysis_method"])
    emit("cores", int(cores), "integer(min=1)", doc=TOP_DOCS["cores"])
    emit("bands", band_entries, "cigale_string_list()", doc=TOP_DOCS["bands"])
    emit("properties", list(properties), "cigale_string_list()",
         doc=TOP_DOCS["properties"])
    emit("additionalerror", float(additional_error), "float(min=0.0)",
         doc=TOP_DOCS["additionalerror"])

    ini.append("[sed_modules_params]")
    spec.append("[sed_modules_params]")
    for mod in sed_modules:
        comment(MODULE_DOCS.get(mod, ""), indent=1)
        ini.append(f"  [[{mod}]]")
        spec.append(f"  [[{mod}]]")
        for par, (typ, _default) in MODULE_REGISTRY[mod].items():
            emit(par, mp[mod][par], typ, indent=2,
                 doc=PARAM_DOCS.get(mod, {}).get(par))

    ini.append("[analysis_params]")
    spec.append("[analysis_params]")
    for par, (typ, _default) in ANALYSIS_REGISTRY["pdf_analysis"].items():
        emit(par, ap[par], typ, indent=1,
             doc=PARAM_DOCS["pdf_analysis"].get(par))

    os.makedirs(run_dir, exist_ok=True)
    ini_path = os.path.join(run_dir, "pcigale.ini")
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ini) + "\n")
    with open(os.path.join(run_dir, "pcigale.ini.spec"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(spec) + "\n")

    if verbose:
        print(f"[cigale] {ini_path}: {len(obs)} objects, {len(bands)} bands, "
              f"{n_z} redshift(s) x {n_models} models = {n_z * n_models} "
              f"total in {ap['blocks']} block(s)")
    return ini_path


def _read_ini(ini_path):
    """Parse a prepare_run-written pcigale.ini back into (top, modules, ap).

    Values stay display strings (exactly as written). Only understands the
    deterministic layout of :func:`prepare_run` — not arbitrary configobj.
    """
    top, modules, analysis = {}, {}, {}
    section, mod = None, None
    with open(ini_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[["):
                mod = line.strip("[]")
                modules[mod] = {}
            elif line.startswith("["):
                section, mod = line.strip("[]"), None
            elif "=" in line:
                key, val = (s.strip() for s in line.split("=", 1))
                if section == "sed_modules_params" and mod:
                    modules[mod][key] = val
                elif section == "analysis_params":
                    analysis[key] = val
                else:
                    top[key] = val
    return top, modules, analysis


def describe_run(run_dir=None, sed_modules=None, module_params=None,
                 analysis_params=None, docs=True):
    """Print a reminder of the modules, parameters and analysis variables.

    The same information ``pcigale genconf`` documents inside ``pcigale.ini``
    — so the notebook shows what is being fitted without opening the ini.

    Parameters
    ----------
    run_dir : str, optional
        With a run directory, the values are read back from its
        ``pcigale.ini`` (i.e. what will actually run). Without one, the
        effective package defaults (merged with the given overrides) are
        shown — call it before :func:`prepare_run` to preview a config.
    sed_modules, module_params, analysis_params
        Only used when *run_dir* is None; same meaning as in
        :func:`prepare_run`.
    docs : bool
        True: full listing with the per-parameter descriptions.
        False: compact — one line per module, values only.
    """
    if run_dir is not None:
        ini_path = os.path.join(run_dir, "pcigale.ini")
        top, mods, ap = _read_ini(ini_path)
        head = ini_path
        fmt = str        # values are already display strings
    else:
        modules, mp, ap = _merged_params(sed_modules, module_params,
                                         analysis_params)
        mods = {mod: {par: mp[mod][par] for par in MODULE_REGISTRY[mod]}
                for mod in modules}
        top, head, fmt = {}, "defaults (nothing written yet)", _fmt

    print(f"[cigale] configuration — {head}")
    for key in ("data_file", "cores", "additionalerror"):
        if key in top:
            print(f"  {key} = {top[key]}")

    print("  SED modules (order matters, redshifting last):")
    for mod, pars in mods.items():
        if docs:
            print(f"    [[{mod}]] — {MODULE_DOCS.get(mod, '?')}")
            for par, val in pars.items():
                doc = PARAM_DOCS.get(mod, {}).get(par, "")
                print(f"      {par} = {fmt(val)}")
                if doc:
                    print(textwrap.fill(doc, width=78,
                                        initial_indent="          # ",
                                        subsequent_indent="          # "))
        else:
            body = "; ".join(f"{p}={fmt(v)}" for p, v in pars.items())
            print(f"    {mod}: {body}")

    print("  analysis (pdf_analysis):")
    for par, val in ap.items():
        if par == "bands":       # long + already visible in the ini
            continue
        if docs:
            doc = PARAM_DOCS["pdf_analysis"].get(par, "")
            print(f"    {par} = {fmt(val)}")
            if doc:
                print(textwrap.fill(doc, width=78,
                                    initial_indent="        # ",
                                    subsequent_indent="        # "))
        else:
            print(f"    {par} = {fmt(val)}")

    nmod = _grid_size([m for m in mods if m in MODULE_REGISTRY], mods)
    print(f"  grid: {nmod} models per redshift")


# ══════════════════════════════════════════════════════════════════════════
# execution
# ══════════════════════════════════════════════════════════════════════════

def _pcigale(run_dir, command, pcigale_cmd="pcigale"):
    """Run `pcigale <command>` inside *run_dir* (streams output)."""
    if not os.path.exists(os.path.join(run_dir, "pcigale.ini")):
        raise FileNotFoundError(f"no pcigale.ini in {run_dir} — "
                                "call prepare_run first")
    try:
        proc = subprocess.run([pcigale_cmd, command], cwd=run_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{pcigale_cmd}' not found on PATH — run this on the machine "
            "with CIGALE installed, or pass pcigale_cmd='/path/to/pcigale'")
    if proc.returncode != 0:
        raise RuntimeError(f"pcigale {command} failed "
                           f"(exit {proc.returncode}) in {run_dir}")


def check(run_dir, pcigale_cmd="pcigale"):
    """`pcigale check` — sanity-check the configuration without fitting."""
    _pcigale(run_dir, "check", pcigale_cmd)


def run(run_dir, pcigale_cmd="pcigale", skip_if_done=False):
    """`pcigale run` in *run_dir*; results in ``<run_dir>/out/``.

    CIGALE renames a pre-existing ``out/`` with a timestamp instead of
    overwriting; ``skip_if_done=True`` returns immediately if
    ``out/results.fits`` already exists.
    """
    results = os.path.join(run_dir, "out", "results.fits")
    if skip_if_done and os.path.exists(results):
        print(f"[cigale] {results} exists — skipped")
        return results
    _pcigale(run_dir, "run", pcigale_cmd)
    if not os.path.exists(results):
        warnings.warn(f"pcigale run finished but {results} is missing")
    return results


def read_results(run_dir):
    """Load ``<run_dir>/out/results.fits`` (Bayesian + best-fit estimates)."""
    return Table.read(os.path.join(run_dir, "out", "results.fits"))


def find_pcigale(env_names=("cigale", "cigale-env"),
                 conda_roots=("miniforge3", "miniconda3")):
    """Absolute path of the ``pcigale`` executable from the dedicated env.

    CIGALE lives in its own conda env (it pins numpy/astropy differently from
    the powderday env), so the notebook kernel never activates it — every
    caller just needs the executable's absolute path. Searches
    ``~/{root}/envs/{env}/bin/pcigale`` for each (root, env) combination,
    then falls back to ``PATH``.

    Raises ``FileNotFoundError`` with setup instructions if not found.
    """
    import shutil
    cands = [os.path.expanduser(f"~/{r}/envs/{e}/bin/pcigale")
             for r in conda_roots for e in env_names]
    cands.append(shutil.which("pcigale"))
    found = next((p for p in cands if p and os.path.exists(p)), None)
    if found is None:
        raise FileNotFoundError(
            f"pcigale not found ({cands[:-1]} + PATH) — create the dedicated "
            "env first:\n  conda create -n cigale python=3.12 -y\n"
            "  conda activate cigale && pip install <cigale-v2025 tarball>")
    return found


def sanitize_input_errors(data_file, verbose=True):
    """Repair legacy all-negative error columns in a CIGALE input file.

    Input files written before the ``convolveFilterWithSED`` sign fix
    (2026-07-07) carry all-negative errors (descending-wavelength hyperion
    SEDs made the trapezoid norms negative). CIGALE reads a negative error
    as "upper limit" — every band an upper limit yields an all-NaN
    ``results.fits``. This ``abs()``-es any negative error in place.

    Returns the number of repaired values (0 = file untouched).
    """
    t = Table.read(data_file)
    ecols = [c for c in t.colnames if c.endswith("_err")]
    n_neg = int(sum((np.asarray(t[c], float) < 0).sum() for c in ecols))
    if n_neg:
        for c in ecols:
            t[c] = np.abs(np.asarray(t[c], float))
        t.write(data_file, overwrite=True)
        if verbose:
            print(f"[repair] {os.path.basename(data_file)}: {n_neg} negative "
                  "error(s) -> abs(), file rewritten")
    return n_neg


# bands dominated by dust emission (or with none of the fit information the
# optical side needs): dropped from the FITTED bands of an optical-only run
IR_BAND_RE = re.compile(r"^(jwst\.miri\.|spitzer\.mips\.|herschel\.|"
                        r"jcmt\.|alma\.)")


def optical_only_run(src_run_dir, dst_run_dir, ir_bands=IR_BAND_RE,
                     verbose=True):
    """Clone a prepared run into its decoupled *optical-only* variant.

    Copies ``pcigale.ini``/``.spec`` from *src_run_dir* (a
    :func:`prepare_run` product) into *dst_run_dir* with three edits — and
    nothing else, so data file, SFH/metallicity grids and analysis setup are
    identical to the coupled run **by construction**:

    - ``dl2014`` is removed from the module chain (and its parameter
      section dropped): the absorbed luminosity is NOT re-emitted, i.e. the
      energy-balance constraint is switched off. ``dust.luminosity`` (the
      absorbed power, set by the attenuation module) is still estimated —
      it is the L_abs side of the energy-balance test;
    - bands matching *ir_bands* are removed from the top-level (fitted)
      ``bands`` list. They stay in the ``[analysis_params]`` bands, so the
      run still predicts ``bayes.<band>`` there — with no dust emission in
      the model that prediction is the pure attenuated stellar continuum,
      which :mod:`simbanator.sed.dl2014_fit` subtracts before fitting the
      IR residual;
    - ``dust.mass`` is dropped from the estimated variables (it is created
      by the dl2014 module and would crash the analysis without it).

    Parameters
    ----------
    src_run_dir : str
        Prepared (coupled) run directory; must contain ``pcigale.ini``.
    dst_run_dir : str
        Destination run directory (created; existing ini overwritten,
        any existing ``out/`` left untouched).
    ir_bands : compiled regex
        Fitted bands whose name (without ``_err``) matches are unfitted.

    Returns
    -------
    str : path to the written ``pcigale.ini`` in *dst_run_dir*.
    """
    src_ini = os.path.join(src_run_dir, "pcigale.ini")
    if not os.path.exists(src_ini):
        raise FileNotFoundError(f"no pcigale.ini in {src_run_dir} — "
                                "call prepare_run there first")

    def _filter_items(value, drop):
        items = [s.strip() for s in value.split(",") if s.strip()]
        kept = [s for s in items if not drop(s)]
        return kept, len(items) - len(kept)

    n_dropped_bands = 0
    found_dl2014 = False

    def _transform(lines, is_spec):
        nonlocal n_dropped_bands, found_dl2014
        out, section, skipping = [], None, False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("["):        # any (sub)section header
                skipping = False
                if stripped == "[[dl2014]]":
                    found_dl2014 = True
                    skipping = True
                    continue
                if stripped in ("[sed_modules_params]", "[analysis_params]"):
                    section = stripped
            if skipping:
                continue
            if section is None and line.startswith("sed_modules = "):
                if not is_spec:
                    kept, _ = _filter_items(line[len("sed_modules = "):],
                                            lambda s: s == "dl2014")
                    line = "sed_modules = " + ", ".join(kept)
            elif section is None and line.startswith("bands = "):
                if not is_spec:
                    kept, n = _filter_items(
                        line[len("bands = "):],
                        lambda s: bool(ir_bands.search(
                            s[:-len("_err")] if s.endswith("_err") else s)))
                    n_dropped_bands = n
                    line = "bands = " + ", ".join(kept)
            elif (section == "[analysis_params]"
                  and line.startswith("  variables = ") and not is_spec):
                kept, _ = _filter_items(line[len("  variables = "):],
                                        lambda s: s == "dust.mass")
                line = "  variables = " + ", ".join(kept)
            out.append(line)
        return out

    os.makedirs(dst_run_dir, exist_ok=True)
    for fname, is_spec in (("pcigale.ini", False), ("pcigale.ini.spec", True)):
        src = os.path.join(src_run_dir, fname)
        with open(src, encoding="utf-8") as f:
            lines = f.read().splitlines()
        with open(os.path.join(dst_run_dir, fname), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(_transform(lines, is_spec)) + "\n")

    if not found_dl2014:
        warnings.warn(f"{src_ini} has no [[dl2014]] section — cloned an "
                      "already dust-emission-free run")
    if verbose:
        print(f"[cigale] {dst_run_dir}: optical-only clone of "
              f"{os.path.basename(src_run_dir)} "
              f"({n_dropped_bands // 2} IR bands unfitted, dl2014 removed)")
    return os.path.join(dst_run_dir, "pcigale.ini")


def write_slurm_array(run_dirs, job_file, pcigale_cmd="pcigale",
                      partition=None, cores=8, mem_per_cpu_mb=3800,
                      time="0-08:00", array_throttle=None, plots=True,
                      plots_format="png", job_name="cigale_fits",
                      skip_if_done=False, verbose=True):
    """Write a SLURM job array that runs ``pcigale run`` in every *run_dirs*.

    One array task per run directory — the right parallel unit: CIGALE builds
    its model grid once per run and shares it across all objects in the data
    file (row-splitting a catalog would recompute the same grid per chunk),
    while within a task pcigale itself fans out over ``cores`` processes.

    Each task ``cd``s into its run dir, rewrites ``cores`` in ``pcigale.ini``
    to the actual ``$SLURM_CPUS_PER_TASK``, sets ``OMP_NUM_THREADS=1``
    (pcigale multiprocessing + threaded BLAS would oversubscribe), runs the
    fit, and optionally ``pcigale-plots sed`` (plot failures do not fail the
    task). By default a run with existing results is **re-fit**: pcigale
    itself renames the pre-existing ``out/`` to a timestamped
    ``<YYYYMMDDHHMM>_out/`` backup before writing the new one (native CIGALE
    behaviour). With ``skip_if_done=True`` such tasks exit immediately
    instead — resubmitting the same job file after a partial failure then
    only runs what is missing.

    Parameters
    ----------
    run_dirs : sequence of str
        Prepared run directories (each must contain a ``pcigale.ini`` from
        :func:`prepare_run`).
    job_file : str
        Path of the sbatch script to write; the run-dir list goes next to it
        (``<stem>.dirs``) and logs into ``<dirname>/slurm_logs/``.
    pcigale_cmd : str
        Absolute path to the ``pcigale`` executable of the CIGALE conda env
        (calling the binary directly needs no ``conda activate``);
        ``<pcigale_cmd>-plots`` is used for the figures.
    partition : str, optional
        SLURM partition(s); omitted from the script if None (pass at submit
        time via ``sbatch -p ...``).
    cores : int
        ``--cpus-per-task`` (pcigale worker processes per task).
    mem_per_cpu_mb : int
        ``--mem-per-cpu`` in MB.
    time : str
        SLURM time limit per task.
    array_throttle : int, optional
        Max tasks running simultaneously (``--array=...%N``).
    plots : bool
        Append the ``pcigale-plots sed`` step to each task.
    skip_if_done : bool
        False (default): re-fit everything, pcigale timestamps-and-keeps any
        previous ``out/``. True: tasks whose ``out/results.fits`` exists exit
        immediately (cheap resubmits).

    Returns
    -------
    str : *job_file*
    """
    run_dirs = [os.path.abspath(d) for d in run_dirs]
    if not run_dirs:
        raise ValueError("run_dirs is empty")
    missing = [d for d in run_dirs
               if not os.path.exists(os.path.join(d, "pcigale.ini"))]
    if missing:
        raise FileNotFoundError(f"no pcigale.ini in {missing} — "
                                "call prepare_run on every run dir first")

    job_file = os.path.abspath(job_file)
    base = os.path.dirname(job_file)
    log_dir = os.path.join(base, "slurm_logs")
    os.makedirs(log_dir, exist_ok=True)
    dirs_file = os.path.splitext(job_file)[0] + ".dirs"
    with open(dirs_file, "w", encoding="utf-8") as f:
        f.write("\n".join(run_dirs) + "\n")

    array = f"0-{len(run_dirs) - 1}"
    if array_throttle:
        array += f"%{int(array_throttle)}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/{job_name}.%A_%a.o",
        f"#SBATCH --error={log_dir}/{job_name}.%A_%a.e",
    ]
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    lines += [
        f"#SBATCH -t {time}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(cores)}",
        f"#SBATCH --mem-per-cpu={int(mem_per_cpu_mb)}",
        f"#SBATCH --array={array}",
        "",
        "set -euo pipefail",
        f'PCIGALE="{pcigale_cmd}"',
        f'DIRS_FILE="{dirs_file}"',
        'RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$DIRS_FILE")',
        'echo "[task ${SLURM_ARRAY_TASK_ID}] ${RUN_DIR}"',
        'cd "$RUN_DIR"',
    ]
    if skip_if_done:
        lines += [
            'if [ -s out/results.fits ]; then',
            '    echo "out/results.fits exists — skipping (skip_if_done)"',
            '    exit 0',
            'fi',
        ]
    else:
        lines += [
            '# a pre-existing out/ is renamed to <YYYYMMDDHHMM>_out/ by '
            'pcigale itself (kept, not overwritten)',
        ]
    lines += [
        '# match the pcigale worker count to the actual allocation',
        'sed -i "s/^cores = .*/cores = ${SLURM_CPUS_PER_TASK}/" pcigale.ini',
        '# pcigale parallelizes over processes; keep BLAS single-threaded',
        'export OMP_NUM_THREADS=1',
        '"$PCIGALE" run',
    ]
    if plots:
        lines += [
            f'"${{PCIGALE}}-plots" sed --format {plots_format} --nologo '
            '|| echo "[warn] pcigale-plots failed (fit results unaffected)"',
        ]
    lines.append('echo "[task ${SLURM_ARRAY_TASK_ID}] done"')

    with open(job_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(job_file, 0o755)

    if verbose:
        n_done = sum(os.path.exists(os.path.join(d, "out", "results.fits"))
                     for d in run_dirs)
        _mode = ("skipped in-task" if skip_if_done
                 else "re-fit; old out/ kept as <timestamp>_out/")
        print(f"[cigale] job array: {len(run_dirs)} tasks "
              f"({n_done} already have results.fits -> {_mode})")
        print(f"[cigale]   run-dir list -> {dirs_file}")
        print(f"[cigale]   submit with:  sbatch"
              + ("" if partition else " -p <partition>") + f" {job_file}")
    return job_file


def plot_seds(run_dir, format="pdf", type="mJy", xrange=None, yrange=None,
              series=None, nologo=True, outdir=None,
              pcigale_plots_cmd="pcigale-plots"):
    """`pcigale-plots sed` — one best-fit SED figure per object.

    Needs a completed :func:`run` with ``save_best_sed=True`` (the default
    here): the plots are drawn from the ``<id>_best_model.fits`` files in
    ``<run_dir>/out/``, where the figures also land (one
    ``<id>_best_model.<format>`` each) unless *outdir* is given.

    Parameters
    ----------
    format : str
        'pdf' or 'png' (any matplotlib-supported format).
    type : str
        'mJy' (observed-frame flux) or 'lum' (rest-frame luminosity).
    xrange, yrange : str, optional
        Axis ranges as ``'<min>:<max>'`` (either side may be empty),
        x in μm.
    series : sequence of str, optional
        Components to draw: stellar_attenuated, stellar_unattenuated,
        nebular, dust, agn, radio, model. Default: all.
    """
    if not os.path.exists(os.path.join(run_dir, "out", "results.fits")):
        raise FileNotFoundError(f"no out/results.fits in {run_dir} — "
                                "run() must complete first")
    cmd = [pcigale_plots_cmd, "sed", "--type", type, "--format", format]
    if nologo:
        cmd.append("--nologo")
    if xrange:
        cmd += ["--xrange", xrange]
    if yrange:
        cmd += ["--yrange", yrange]
    if outdir:
        cmd += ["--outdir", outdir]
    if series:
        cmd += ["--series", *series]
    try:
        proc = subprocess.run(cmd, cwd=run_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{pcigale_plots_cmd}' not found on PATH — pass "
            "pcigale_plots_cmd='/path/to/pcigale-plots'")
    if proc.returncode != 0:
        raise RuntimeError(f"pcigale-plots sed failed (exit {proc.returncode})"
                           f" in {run_dir}")
    return outdir or os.path.join(run_dir, "out")


# ══════════════════════════════════════════════════════════════════════════
# truth vs fit comparison
# ══════════════════════════════════════════════════════════════════════════

# fixed-order colorblind-safe hues (Okabe & Ito) for the redshift groups
_CB_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
              "#D55E00", "#F0E442", "#000000")

# axis labels for the properties this package saves by default
# (Myr-valued CIGALE outputs are DISPLAYED in Gyr — see _GYR_RE below)
_PROP_UNITS = {
    "stellar.m_star": r"$M_\star$ [$M_\odot$]",
    "sfh.sfr": r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]",
    "sfh.sfr10Myrs": r"$\langle$SFR$\rangle_{10\,\mathrm{Myr}}$ "
                     r"[$M_\odot\,\mathrm{yr}^{-1}$]",
    "sfh.sfr100Myrs": r"$\langle$SFR$\rangle_{100\,\mathrm{Myr}}$ "
                      r"[$M_\odot\,\mathrm{yr}^{-1}$]",
    "sfh.age_bq": "burst/quench lookback [Gyr]",
    "sfh.age_main": "SF onset age [Gyr]",
    "sfh.tau_main": r"$\tau_\mathrm{main}$ [Gyr]",
    "stellar.age_m_star": "mass-weighted stellar age [Gyr]",
    "sfh.r_sfr": r"$r_\mathrm{SFR}$ (after/before age$_\mathrm{bq}$)",
    "attenuation.Av_ISM": r"$A_{V,\,\mathrm{ISM}}$ [mag]",
    "dust.luminosity": r"$L_\mathrm{dust}$ [W]",
    "dust.mass": r"$M_\mathrm{dust}$ [$M_\odot$]",
    "stellar.metallicity": "Z",
}

# CIGALE reports these in Myr; every display (panels, stats, printed table)
# converts to Gyr. The saved comparison FITS keeps the original Myr values.
_GYR_RE = re.compile(r"age|tau")


def nmad(x):
    """1.4826 x median absolute deviation from the median (robust sigma)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


_nmad = nmad


def compare_results(run_dir, truth, log_props=None, outdir=None,
                    basename="simba_vs_cigale", show=True, verbose=True,
                    highlight_col="dusty"):
    """Truth (simulation) vs CIGALE estimates: print, plot, save into out/.

    Parameters
    ----------
    run_dir : str
        A completed run directory (reads ``<run_dir>/out/results.fits``).
    truth : str or :class:`~astropy.table.Table`
        Per-galaxy truth with an ``id`` column matching the CIGALE ids
        (``snapNNN_galID``) and one column **per CIGALE output property to
        compare, named exactly like it** — e.g. ``stellar.m_star`` [Msun],
        ``sfh.sfr`` [Msun/yr], ``sfh.age_bq`` [Myr]. Every other column
        (``z_snap``, ``agn_class``, ...) is carried into the saved table;
        ``z_snap``/``redshift``/``z_target`` (first found) colors the points.
    log_props : sequence of str, optional
        Properties compared in log space. Default: any name containing
        ``m_star``/``mass``/``sfr``/``luminosity``. Non-positive values are
        drawn as open symbols pinned at a floor, not dropped.
    outdir : str, optional
        Destination of the saved table + figure (default ``<run_dir>/out``).
    basename : str
        Stem of the saved files: ``<basename>.fits`` + ``<basename>.png``.
    show : bool
        Show the figure (else it is only saved and closed).
    highlight_col : str, optional
        Name of a 1/0(/−1 = unknown) truth column marking a subsample to
        highlight (default ``'dusty'``, silently skipped when absent):
        value-1 objects get a red ring in every panel on top of the redshift
        colors, and the offset/NMAD stats are additionally printed for the
        1 and 0 subsamples separately. Pass None to disable.

    Returns
    -------
    :class:`~astropy.table.Table`
        One row per matched galaxy: the carried truth columns plus, per
        property, ``<prop>_true``, ``<prop>_cigale`` (bayes),
        ``<prop>_cigale_err`` and ``<prop>_best`` — the same table saved to
        ``<basename>.fits``.
    """
    import matplotlib.pyplot as plt

    res = read_results(run_dir)
    if isinstance(truth, str):
        truth = Table.read(truth)
    if "id" not in truth.colnames:
        raise ValueError("truth table needs an 'id' column (snapNNN_galID)")

    rid = {str(r): i for i, r in enumerate(res["id"])}
    tid = [str(t) for t in truth["id"]]
    keep = [i for i, t in enumerate(tid) if t in rid]
    if not keep:
        raise ValueError("no truth id matches the ids in out/results.fits")
    missing = len(tid) - len(keep)
    if missing and verbose:
        print(f"[cigale] {missing} truth object(s) not in results.fits "
              "(dropped)")
    ridx = np.array([rid[tid[i]] for i in keep])

    props = [c for c in truth.colnames
             if c != "id" and f"bayes.{c}" in res.colnames]
    if not props:
        raise ValueError(
            "no truth column matches a bayes.* property in results.fits — "
            "name the truth columns after the CIGALE variables "
            f"(saved here: {[c[6:] for c in res.colnames if c.startswith('bayes.') and not c.endswith('_err')]})")
    if log_props is None:
        log_props = [p for p in props if any(
            k in p for k in ("m_star", "mass", "sfr", "luminosity"))]
        # ages/taus are never logged ('age_m_star' contains 'm_star'):
        # they are linear in Gyr (see _GYR_RE)
        log_props = [p for p in log_props
                     if p != "sfh.r_sfr" and not _GYR_RE.search(p)]

    comp = Table()
    comp["id"] = [tid[i] for i in keep]
    extras = [c for c in truth.colnames if c != "id" and c not in props]
    for c in extras:
        comp[c] = truth[c][keep]

    # highlighted subsample (e.g. the notebook's Part 7a dusty flag): 1 / 0 / -1-unknown
    hmask = nmask = None
    if highlight_col and highlight_col in extras:
        _hv = np.asarray(comp[highlight_col], float)
        hmask = _hv == 1
        nmask = _hv == 0
    for p in props:
        comp[f"{p}_true"] = np.asarray(truth[p], float)[keep]
        comp[f"{p}_cigale"] = np.asarray(res[f"bayes.{p}"], float)[ridx]
        ecol = f"bayes.{p}_err"
        comp[f"{p}_cigale_err"] = (np.asarray(res[ecol], float)[ridx]
                                   if ecol in res.colnames else np.nan)
        bcol = f"best.{p}"
        if bcol in res.colnames:
            comp[f"{p}_best"] = np.asarray(res[bcol], float)[ridx]

    # Myr-valued (linear) properties are displayed in Gyr everywhere
    gyr_props = {p for p in props
                 if p not in log_props and _GYR_RE.search(p)}

    # ── per-galaxy print + offset/scatter stats ──
    if verbose:
        disp = Table()
        disp["id"] = comp["id"]
        for c in extras:
            disp[c] = comp[c]
        for p in props:
            if p in log_props:
                with np.errstate(all="ignore"):
                    disp[f"log {p} true"] = np.round(
                        np.log10(comp[f"{p}_true"]), 2)
                    disp[f"log {p} cigale"] = np.round(
                        np.log10(comp[f"{p}_cigale"]), 2)
            elif p in gyr_props:
                disp[f"{p} true [Gyr]"] = np.round(
                    np.asarray(comp[f"{p}_true"], float) / 1e3, 2)
                disp[f"{p} cigale [Gyr]"] = np.round(
                    np.asarray(comp[f"{p}_cigale"], float) / 1e3, 2)
            else:
                disp[f"{p} true"] = np.round(comp[f"{p}_true"], 1)
                disp[f"{p} cigale"] = np.round(comp[f"{p}_cigale"], 1)
        disp.pprint(max_lines=-1, max_width=-1)
        print()
        for p in props:
            t, e = comp[f"{p}_true"], comp[f"{p}_cigale"]
            if p in log_props:
                with np.errstate(all="ignore"):
                    d = np.log10(e) - np.log10(t)
                unit = " dex"
            elif p in gyr_props:
                d = np.asarray(e - t, float) / 1e3
                unit = " Gyr"
            else:
                d = np.asarray(e - t, float)
                unit = ""
            ok = np.isfinite(d)
            if not ok.any():
                print(f"  {p}: no valid (truth, estimate) pairs — check "
                      "out/results.fits")
                continue
            print(f"  {p}: median offset {np.median(d[ok]):+.2f}{unit}, "
                  f"NMAD {_nmad(d):.2f}{unit}  ({ok.sum()}/{len(d)} valid)")
            if hmask is not None:
                for _lab, _mm in ((highlight_col, hmask),
                                  (f"non-{highlight_col}", nmask)):
                    _okm = ok & _mm
                    if _okm.any():
                        print(f"      {_lab:>12s}: median "
                              f"{np.median(d[_okm]):+.2f}{unit}, "
                              f"NMAD {_nmad(d[_okm]):.2f}{unit}  "
                              f"({_okm.sum()})")

    # ── one-to-one panels ──
    zcol = next((c for c in ("z_snap", "redshift", "z_target")
                 if c in extras), None)
    if zcol is not None:
        zvals = np.round(np.asarray(comp[zcol], float), 3)
        groups = [(z, zvals == z) for z in np.unique(zvals)]
    else:
        groups = [(None, np.ones(len(comp), bool))]

    from matplotlib.ticker import MaxNLocator

    n = len(props)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.9 * ncols, 4.5 * nrows),
                             squeeze=False)
    for k, p in enumerate(props):
        ax = axes[k // ncols][k % ncols]
        scale = 1e-3 if p in gyr_props else 1.0     # Myr -> Gyr display
        t = np.asarray(comp[f"{p}_true"], float) * scale
        e = np.asarray(comp[f"{p}_cigale"], float) * scale
        err = np.asarray(comp[f"{p}_cigale_err"], float) * scale
        logp = p in log_props
        tt, ee = t.copy(), e.copy()
        pinned = np.zeros(len(t), bool)
        if logp:
            pos = np.concatenate([t[np.isfinite(t) & (t > 0)],
                                  e[np.isfinite(e) & (e > 0)]])
            if len(pos):
                # cap the plotted range at 8 decades: zeros AND posterior
                # underflow (1e-90 Msun/yr "quenched" SFRs) pin at the floor
                floor = max(pos.min() / 5, pos.max() * 1e-8)
            else:
                floor = 1.0

            def _clip(x):
                pin = np.isfinite(x) & (x <= floor)
                return np.where(pin, floor, x), pin

            tt, pin_t = _clip(t)
            ee, pin_e = _clip(e)
            pinned = pin_t | pin_e
        both = np.isfinite(tt) & np.isfinite(ee)
        if not both.any():
            ax.text(0.5, 0.5, "no finite\n(truth, estimate) pairs",
                    transform=ax.transAxes, ha="center", va="center",
                    color="0.4")
            ax.set_title(p, fontsize=11)
            continue
        lo = np.nanmin(np.concatenate([tt[both], ee[both]]))
        hi = np.nanmax(np.concatenate([tt[both], ee[both]]))
        pad = (hi / lo) ** 0.08 if logp else 0.06 * (hi - lo)
        lim = ((lo / pad, hi * pad) if logp
               else (lo - pad, hi + pad))
        ax.plot(lim, lim, ls="--", lw=1, color="0.6", zorder=1)
        for (z, m), color in zip(groups, _CB_COLORS):
            m = m & both
            if not m.any():
                continue
            lab = f"z={z:g}" if z is not None else None
            ax.errorbar(tt[m], ee[m], yerr=np.where(np.isfinite(err[m]),
                                                    err[m], 0),
                        fmt="none", ecolor="0.75", elinewidth=1, zorder=2)
            solid = m & ~pinned
            ax.scatter(tt[solid], ee[solid], s=40, color=color, label=lab,
                       zorder=3)
            if (m & pinned).any():
                ax.scatter(tt[m & pinned], ee[m & pinned], s=40,
                           facecolors="none", edgecolors=color, zorder=3)
        if hmask is not None and (hmask & both).any():
            _hm = hmask & both               # ring on top of the z colors
            ax.scatter(tt[_hm], ee[_hm], s=110, facecolors="none",
                       edgecolors="#c0392b", lw=1.3, zorder=4,
                       label=highlight_col if k == 0 else None)
        if logp:
            ax.set_xscale("log"), ax.set_yscale("log")
            if pinned.any():
                ax.text(0.97, 0.03, f"open: {int(pinned.sum())} at floor",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="bottom", color="0.35")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=9)
        ax.set_xlim(lim), ax.set_ylim(lim)
        unit = _PROP_UNITS.get(p, p)
        ax.set_xlabel(f"SIMBA {unit}", fontsize=10)
        ax.set_ylabel(f"CIGALE {unit}", fontsize=10)
        ax.set_title(p, fontsize=11)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    # one figure-level legend (z colors + highlight ring) instead of
    # cramming it into the first panel
    _hnd, _lbl = axes[0][0].get_legend_handles_labels()
    fig.suptitle(os.path.basename(os.path.normpath(run_dir)), y=0.995,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if _hnd:
        fig.legend(_hnd, _lbl, loc="upper center", ncol=min(len(_hnd), 7),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.975))

    # ── save into this run's out/ ──
    outdir = outdir or os.path.join(run_dir, "out")
    os.makedirs(outdir, exist_ok=True)
    fits_path = os.path.join(outdir, f"{basename}.fits")
    png_path = os.path.join(outdir, f"{basename}.png")
    comp.write(fits_path, overwrite=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"\n[cigale] comparison table -> {fits_path}")
        print(f"[cigale] comparison figure -> {png_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return comp


# ══════════════════════════════════════════════════════════════════════════
# parameter-grid diagnostics (how well the grid samples the posteriors)
# ══════════════════════════════════════════════════════════════════════════

# short axis labels for the parameters that are usually put on a grid, keyed
# by the module-input (leaf) name — the way they appear in DEFAULT_MODULE_PARAMS
_GRID_LABELS = {
    "tau_main": r"$\tau_\mathrm{main}$ [Myr]",
    "age_main": r"age$_\mathrm{main}$ [Myr]",
    "age_bq": r"age$_\mathrm{bq}$ [Myr]",
    "r_sfr": r"$r_\mathrm{SFR}$ (after/before age$_\mathrm{bq}$)",
    "metallicity": "Z (stellar)",
    "Av_ISM": r"$A_{V,\,\mathrm{ISM}}$ [mag]",
    "mu": r"$\mu$ (ISM/total attenuation)",
    "qpah": r"$q_\mathrm{PAH}$ [%]",
    "umin": r"$U_\mathrm{min}$",
    "gamma": r"$\gamma$ (PDR fraction)",
    "alpha": r"$\alpha$ (dust)",
    "logU": "log $U$",
    "zgas": "Z$_\\mathrm{gas}$",
}


def _parse_grid(display):
    """Grid values (sorted floats) from a pcigale.ini value, or None.

    Accepts what :func:`_read_ini` returns — e.g. ``"500, 1000, 2000"`` or a
    one-element ``"0.02,"``. Returns None when any token is non-numeric (a
    string parameter like ``filters`` or an empty auto-filled grid).
    """
    toks = [t.strip() for t in str(display).split(",")]
    toks = [t for t in toks if t]
    if not toks:
        return None
    try:
        vals = sorted({float(t) for t in toks})
    except ValueError:
        return None
    return vals


def _result_cols(colnames, param):
    """(bayes_col, best_col) whose dotted leaf equals *param*, else (None,..).

    CIGALE writes the fitted parameters as ``<estimator>.<prefix>.<leaf>``
    (``best.sfh.tau_main``, ``bayes.stellar.metallicity``, ...); matching on
    the leaf avoids hard-coding each module's output prefix. ``_err`` columns
    have leaf ``<leaf>_err`` and are skipped by the exact-match.
    """
    def pick(kind):
        hits = [c for c in colnames
                if c.startswith(kind + ".") and c.split(".")[-1] == param]
        return hits[0] if hits else None
    return pick("bayes"), pick("best")


def plot_parameter_priors(run_dir, module_params=None, sed_modules=None,
                          analysis_params=None, min_grid=2,
                          edge_frac=0.15, gap_frac=0.25, unused_frac=0.02,
                          outdir=None, basename="param_priors",
                          show=True, verbose=True):
    """Fitted distribution of each gridded parameter vs. the prior grid nodes.

    For every parameter sampled on a grid (≥ *min_grid* values in the run's
    ``pcigale.ini``) this draws, over the whole fitted sample:

    * the **prior grid nodes** you chose (triangles on the axis);
    * the **best-fit usage** of each node (stems = fraction of galaxies whose
      best model sits on that node) — reveals never-picked nodes and pile-up;
    * the **Bayesian posterior** spread (filled histogram of the PDF-weighted
      ``bayes.*`` estimate, binned one cell per node with an over/under-flow
      bin) — reveals mass falling *between* nodes or pushing *past* an edge.

    and prints a one-line verdict per parameter so you can decide how to
    change the grid:

    * **extend** — too much mass sits on / past an end node (``edge_frac``);
    * **refine** — a between-node cell holds a large posterior share while the
      fits split across it (``gap_frac``): the grid is too coarse there;
    * **trim** — a node is essentially never selected (``unused_frac``).

    Parameters
    ----------
    run_dir : str
        A completed run directory (reads ``<run_dir>/out/results.fits`` and,
        for the grid, ``<run_dir>/pcigale.ini``).
    module_params, sed_modules, analysis_params
        Optional. If given, the grid is taken from these merged with the
        package defaults (as in :func:`prepare_run`) instead of from the ini —
        use to preview a grid you have not written yet against existing
        results. ``analysis_params`` is accepted for signature parity and is
        unused here.
    min_grid : int
        Only parameters with at least this many grid values are shown
        (1 to include fixed, single-value parameters too).
    edge_frac, gap_frac, unused_frac : float
        Thresholds (fraction of the sample) for the extend / refine / trim
        verdicts described above.
    outdir : str, optional
        Where to write ``<basename>.fits`` + ``<basename>.png`` (default
        ``<run_dir>/out``).
    show : bool
        Show the figure (else only saved and closed).

    Returns
    -------
    :class:`~astropy.table.Table`
        One row per gridded parameter: ``param``, ``property`` (the matched
        results column stem), ``grid`` (nodes as a string), ``n_grid``,
        ``n_obj``, ``frac_low_node``/``frac_high_node`` (best-fit usage of the
        end nodes), ``frac_below``/``frac_above`` (posterior mass past the end
        nodes), ``n_unused`` and ``verdict`` — also saved to ``<basename>.fits``.
    """
    import matplotlib.pyplot as plt

    res = read_results(run_dir)

    # ── grid (priors): from the ini that ran, or from merged overrides ──
    if module_params is not None or sed_modules is not None:
        modules, mp, _ = _merged_params(sed_modules, module_params,
                                        analysis_params)
        grids = {(mod, par): _parse_grid(_fmt(mp[mod][par]))
                 for mod in modules for par in mp[mod]}
    else:
        _, mods, _ = _read_ini(os.path.join(run_dir, "pcigale.ini"))
        grids = {(mod, par): _parse_grid(val)
                 for mod, pars in mods.items() for par, val in pars.items()}

    # keep numeric grids with >= min_grid nodes that have a results column
    items = []
    skipped = []
    for (mod, par), g in grids.items():
        if g is None or len(g) < min_grid:
            continue
        bcol, best = _result_cols(res.colnames, par)
        if bcol is None and best is None:
            skipped.append(f"{mod}.{par} (no bayes.*/best.*.{par} column)")
            continue
        prop = (bcol or best).split(".", 1)[1]        # strip estimator prefix
        items.append((mod, par, np.array(g, float), prop, bcol, best))
    if not items:
        raise ValueError(
            "no gridded parameter matched a results column — need >= "
            f"{min_grid} grid values and a bayes.*/best.* output. Skipped: "
            + ("; ".join(skipped) if skipped else "none"))
    if skipped and verbose:
        print("[cigale] not plotted (no matching output column): "
              + "; ".join(skipped))

    # ── figure ──
    n = len(items)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 3.9 * nrows),
                             squeeze=False)

    rows = []
    for k, (mod, par, g, prop, bcol, best) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        nnode = len(g)
        # bin edges: one cell per node (midpoints), plus an over/under-flow bin
        mids = 0.5 * (g[:-1] + g[1:])
        step = np.median(np.diff(g)) if nnode > 1 else (abs(g[0]) or 1.0)
        inner = np.concatenate([[g[0] - 0.5 * step], mids,
                                [g[-1] + 0.5 * step]])
        edges = np.concatenate([[inner[0] - step], inner, [inner[-1] + step]])

        # best-fit usage per node (snap to nearest node to absorb float noise)
        best_frac = np.zeros(nnode)
        nb = 0
        if best is not None:
            bv = np.asarray(res[best], float)
            bv = bv[np.isfinite(bv)]
            nb = len(bv)
            if nb:
                idx = np.abs(bv[:, None] - g[None, :]).argmin(axis=1)
                best_frac = np.bincount(idx, minlength=nnode) / nb

        # bayesian spread + edge pressure
        frac_below = frac_above = np.nan
        nby = 0
        if bcol is not None:
            yv = np.asarray(res[bcol], float)
            yv = yv[np.isfinite(yv)]
            nby = len(yv)
            if nby:
                frac_below = np.mean(yv < g[0])
                frac_above = np.mean(yv > g[-1])
                clip = np.clip(yv, edges[0] + 1e-9 * step, edges[-1] - 1e-9 * step)
                hist, _ = np.histogram(clip, bins=edges)
                ax.stairs(hist / nby, edges, fill=True, alpha=0.35,
                          color="#0072B2",
                          label=f"posterior (bayes, N={nby})")

        # prior grid nodes + best-fit usage stems
        ax.plot(g, np.zeros(nnode), marker="^", ls="none", ms=9,
                color="0.35", clip_on=False, label="prior grid node")
        if nb:
            ax.vlines(g, 0, best_frac, color="#D55E00", lw=2, zorder=3)
            ax.plot(g, best_frac, "o", color="#D55E00", ms=5, zorder=4,
                    label=f"best-fit usage (N={nb})")

        # verdicts
        verdicts = []
        if nb:
            if best_frac[0] >= edge_frac:
                verdicts.append("extend-low")
            if best_frac[-1] >= edge_frac:
                verdicts.append("extend-high")
        if np.isfinite(frac_below) and frac_below >= edge_frac:
            verdicts.append("extend-low")
        if np.isfinite(frac_above) and frac_above >= edge_frac:
            verdicts.append("extend-high")
        n_unused = int(np.sum(best_frac < unused_frac)) if nb else 0
        if nb and n_unused:
            verdicts.append(f"trim({n_unused})")
        # refine: a lot of posterior mass sits in the *middle* of an inter-node
        # gap (central half of a cell), i.e. between nodes rather than hugging
        # one — the grid is too coarse to resolve it there. Mass peaking ON a
        # node (the common, healthy case) is excluded by the central-half band.
        if nby and nnode > 1:
            mid_share = [np.mean((yv > lo + 0.25 * (hi - lo)) &
                                 (yv < hi - 0.25 * (hi - lo)))
                         for lo, hi in zip(g[:-1], g[1:])]
            if max(mid_share, default=0.0) >= gap_frac:
                verdicts.append("refine")
        verdict = ", ".join(dict.fromkeys(verdicts)) or "ok"

        # cosmetics
        if g[0] > 0 and g[-1] / g[0] >= 50:
            ax.set_xscale("log")
        ax.set_xlabel(_GRID_LABELS.get(par, f"{mod}.{par}"))
        ax.set_ylabel("fraction of galaxies")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{prop}   [{verdict}]", fontsize=10)
        ax.margins(x=0.08)
        if k == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")

        rows.append({
            "param": f"{mod}.{par}", "property": prop,
            "grid": ", ".join(f"{v:g}" for v in g), "n_grid": nnode,
            "n_obj": max(nb, nby),
            "frac_low_node": round(float(best_frac[0]), 3) if nb else np.nan,
            "frac_high_node": round(float(best_frac[-1]), 3) if nb else np.nan,
            "frac_below": (round(float(frac_below), 3)
                           if np.isfinite(frac_below) else np.nan),
            "frac_above": (round(float(frac_above), 3)
                           if np.isfinite(frac_above) else np.nan),
            "n_unused": n_unused, "verdict": verdict,
        })

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    fig.suptitle(f"{os.path.basename(os.path.normpath(run_dir))} — "
                 "parameter grid vs. fitted distribution", y=1.0)
    fig.tight_layout()

    summary = Table(rows=rows, names=list(rows[0].keys()))
    if verbose:
        print("[cigale] grid diagnostics "
              "(verdict: extend/refine/trim = act; ok = leave):")
        summary.pprint(max_lines=-1, max_width=-1)
        acts = [r for r in rows if r["verdict"] != "ok"]
        if acts:
            print("\n  suggested grid changes:")
            for r in acts:
                print(f"    {r['param']:<32} [{r['grid']}]  -> {r['verdict']}")
        else:
            print("\n  grid looks well-matched to the posteriors "
                  "(no edge pile-up, gaps or dead nodes).")

    outdir = outdir or os.path.join(run_dir, "out")
    os.makedirs(outdir, exist_ok=True)
    fits_path = os.path.join(outdir, f"{basename}.fits")
    png_path = os.path.join(outdir, f"{basename}.png")
    summary.write(fits_path, overwrite=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"\n[cigale] grid-diagnostics table  -> {fits_path}")
        print(f"[cigale] grid-diagnostics figure -> {png_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return summary
