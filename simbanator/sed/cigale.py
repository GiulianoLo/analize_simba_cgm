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

Only needs numpy + astropy; ``pcigale`` itself is required just where
:func:`run`/:func:`check` execute. The config layout and the filter names are
pinned to **CIGALE 2025.0** (verified against its source); other versions may
need :data:`MODULE_REGISTRY` / the band sets updated.
"""

import os
import subprocess
import warnings

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck13

__all__ = [
    "cigale_band", "write_cigale_input", "prepare_run", "check", "run",
    "read_results", "DEFAULT_SED_MODULES", "DEFAULT_MODULE_PARAMS",
    "DEFAULT_ANALYSIS_PARAMS", "CIGALE_KNOWN_BANDS",
]


# ══════════════════════════════════════════════════════════════════════════
# band-name mapping: '<Facility>.<Instrument>.<filter>' -> CIGALE 2025.0 names
# (sets verified against pcigale/data/filters of the 2025.0 release)
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
    }
    | {f"hst.wfc3.ir.{f}" for f in WFC3_IR_FILTERS}
    | {f"hst.wfc3.uvis.{f}" for f in WFC3_UVIS_FILTERS}
    | {f"jwst.nircam.{f}" for f in NIRCAM_FILTERS}
)

# local filter files used by the notebooks -> CIGALE names
CIGALE_OVERRIDES = {
    "2MASS.J.J": "2mass.J",
    "Johnson.V.V": "generic.johnson.V",
    "Johnson2.U.U": "generic.johnson.U",
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
        sub = "ir" if filt in WFC3_IR_FILTERS else "uvis"
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

    dropped = []
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
        if dropped:
            print(f"[cigale]   dropped (not in the CIGALE 2025.0 DB): "
                  f"{sorted(set(dropped))}")
    return outpath


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

# defaults tuned for the quenched-m25 sample: delayed SFH with a burst/quench
# episode (sfhdelayedbq), Chabrier IMF (SIMBA), two-phase CF00 attenuation,
# DL2014 dust emission. ~126k models per redshift with these grids.
DEFAULT_SED_MODULES = ("sfhdelayedbq", "bc03", "nebular",
                       "dustatt_modified_CF00", "dl2014", "redshifting")

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
}

DEFAULT_ANALYSIS_PARAMS = {
    "variables": ["stellar.m_star", "stellar.metallicity",
                  "sfh.sfr", "sfh.sfr10Myrs", "sfh.sfr100Myrs",
                  "sfh.tau_main", "sfh.age_bq", "sfh.r_sfr",
                  "dust.luminosity"],
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
    return str(value)


def prepare_run(run_dir, data_file, sed_modules=None, module_params=None,
                analysis_params=None, cores=None, additional_error=0.1,
                properties=(), verbose=True):
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

    Returns
    -------
    str : path to the written ``pcigale.ini``.
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

    ap = dict(DEFAULT_ANALYSIS_PARAMS)
    ap.update(analysis_params or {})
    bad = set(ap) - set(ANALYSIS_REGISTRY["pdf_analysis"])
    if bad:
        raise KeyError(f"unknown analysis_params {sorted(bad)} "
                       f"(known: {sorted(ANALYSIS_REGISTRY['pdf_analysis'])})")
    ap["bands"] = bands   # fluxes to predict; independent of the fit itself

    ini, spec = [], []

    def emit(key, value, typ, indent=0):
        pad = "  " * indent
        ini.append(f"{pad}{key} = {_fmt(value)}")
        spec.append(f"{pad}{key} = {typ}")

    emit("data_file", data_file, "string()")
    emit("parameters_file", "", "string()")
    emit("sed_modules", list(sed_modules), "cigale_string_list()")
    emit("analysis_method", "pdf_analysis", "string()")
    emit("cores", int(cores), "integer(min=1)")
    emit("bands", band_entries, "cigale_string_list()")
    emit("properties", list(properties), "cigale_string_list()")
    emit("additionalerror", float(additional_error), "float(min=0.0)")

    ini.append("[sed_modules_params]")
    spec.append("[sed_modules_params]")
    for mod in sed_modules:
        ini.append(f"  [[{mod}]]")
        spec.append(f"  [[{mod}]]")
        for par, (typ, default) in MODULE_REGISTRY[mod].items():
            emit(par, mp[mod].get(par, default), typ, indent=2)

    ini.append("[analysis_params]")
    spec.append("[analysis_params]")
    for par, (typ, default) in ANALYSIS_REGISTRY["pdf_analysis"].items():
        emit(par, ap.get(par, default), typ, indent=1)

    os.makedirs(run_dir, exist_ok=True)
    ini_path = os.path.join(run_dir, "pcigale.ini")
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ini) + "\n")
    with open(os.path.join(run_dir, "pcigale.ini.spec"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(spec) + "\n")

    if verbose:
        nz = len(np.unique(np.round(np.asarray(obs["redshift"], float),
                                    int(ap.get("redshift_decimals", 2)))))
        nmod = 1
        for mod in sed_modules:
            for par in MODULE_REGISTRY[mod]:
                v = mp[mod].get(par, MODULE_REGISTRY[mod][par][1])
                if isinstance(v, (list, tuple, np.ndarray)):
                    nmod *= max(len(v), 1)
        print(f"[cigale] {ini_path}: {len(obs)} objects, {len(bands)} bands, "
              f"{nz} redshift(s) x {nmod} models = {nz * nmod} total")
    return ini_path


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
