"""Standalone Draine & Li (2014) fit of the MIR/FIR SED — no energy balance.

CIGALE ties the dust-emission normalization to the luminosity absorbed by the
attenuation module (energy balance), so it cannot fit the IR side of an SED
independently of the stars. This module does exactly that: it fits the DL2014
templates (pulled from the CIGALE 2025 database, combined exactly as
``pcigale.sed_modules.dl2014`` does) to the IR photometry alone, with the
normalization — the total dust luminosity — as a free analytic parameter.

Designed to run on the *optical-only* CIGALE runs produced by
:func:`simbanator.sed.cigale.optical_only_run`: for each run directory it

1. reads the run's own ``data_file`` (observed-frame fluxes, mJy) and its
   ``out/results.fits``;
2. selects the IR bands (MIRI, MIPS, SCUBA-2, ALMA, ...) from the data file —
   the bands the optical-only run deliberately did not fit;
3. subtracts the attenuated **stellar continuum** predicted by the
   optical-only fit (``bayes.<band>`` — with no dust-emission module in that
   run this is starlight only). Bands with no prediction (BC03 stops at
   rest-frame 160 um, so SCUBA-2/ALMA are NaN there) are used unsubtracted:
   the Rayleigh-Jeans tail is negligible at those wavelengths;
4. fits ``F_dust = L_dust * template(qpah, umin, gamma; alpha)`` over a grid
   of shape parameters with the amplitude ``L_dust`` solved analytically per
   grid point (chi^2-linear least squares, clamped at 0);
5. writes ``out/dl2014_results.fits`` with ``best.*``/``bayes.*`` columns
   mirroring the CIGALE naming (``dust.luminosity`` in W, ``dust.mass`` in kg
   — the CIGALE conventions, directly comparable to the coupled run).

Error budget per band: Monte-Carlo error (+) ``additionalerror * F_obs``
(read from the run's pcigale.ini, the same 10% CIGALE adds at fit time) (+)
the uncertainty of the stellar prediction, all in quadrature.

MUST run under the CIGALE conda env's python (needs ``pcigale.data``); the
module is still importable elsewhere (pcigale imports are deferred), so a
notebook can locate it via ``dl2014_fit.__file__`` and subprocess it::

    ~/miniforge3/envs/cigale/bin/python -m simbanator.sed.dl2014_fit \
        --run-base output/cis25/cigale_runs --pattern 'optonly_dust_on_*'

(or call the file path directly — it has a __main__ guard).
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from astropy.table import Table

_trapz = getattr(np, "trapezoid", None) or np.trapz

MPC_M = 3.0856775814913673e22          # 1 Mpc in m (astropy value)

# dust-emission-dominated bands fitted here (must complement the fitted-band
# drop of cigale.optical_only_run)
IR_BAND_RE = r"^(jwst\.miri\.|spitzer\.mips\.|herschel\.|jcmt\.|alma\.)"

# shape-parameter grid: qpah/umin restricted to DL2014 database nodes.
# qpah + umin deliberately include the coupled-run grid (cigale notebook Part
# 7c': qpah 0.47/1.77/3.90, umin 1-35) so best-fit points are comparable;
# gamma floats here (the coupled grid pins it at 0.0085).
DEFAULT_QPAH = (0.47, 1.77, 3.90)
DEFAULT_UMIN = (0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 7.0, 12.0, 20.0, 35.0, 50.0)
DEFAULT_GAMMA = (0.0, 0.0085, 0.05, 0.2, 0.5)
DEFAULT_ALPHA = 2.0


def load_templates(qpah=DEFAULT_QPAH, umin=DEFAULT_UMIN, gamma=DEFAULT_GAMMA,
                   alpha=DEFAULT_ALPHA):
    """DL2014 templates normalized to 1 W of total dust luminosity.

    Combines the database entries exactly as ``pcigale.sed_modules.dl2014``
    does: ``(1-gamma) * delta(Umin)`` (the ``umax=umin, alpha=1`` entry) plus
    ``gamma * powerlaw(Umin -> 1e7, alpha)``.

    Returns
    -------
    (wl, models) : wl the rest-frame wavelength grid in nm; models a list of
        ``(qpah, umin, gamma, spec_per_W, emissivity)`` tuples with *spec*
        in W/nm per W of L_dust and *emissivity* in W per kg of dust
        (so ``M_dust = L_dust / emissivity``).
    """
    from pcigale.data import SimpleDatabase

    models = []
    with SimpleDatabase("dl2014") as db:
        for name, vals in (("qpah", qpah), ("umin", umin), ("alpha", [alpha])):
            bad = sorted(set(vals) - set(db.parameters[name]))
            if bad:
                raise ValueError(
                    f"{name}={bad} not in the DL2014 database grid "
                    f"(allowed: {sorted(db.parameters[name])})")
        for q in qpah:
            for u in umin:
                minmin = db.get(qpah=q, umin=u, umax=u, alpha=1.0)
                minmax = db.get(qpah=q, umin=u, umax=1e7, alpha=alpha)
                wl = minmin.wl                       # nm, rest frame
                for g in gamma:
                    spec = (1. - g) * minmin.spec + g * minmax.spec
                    emissivity = _trapz(spec, x=wl)  # W / kg of dust
                    models.append((q, u, g, spec / emissivity, emissivity))
    return wl, models


def emissivity_table(qpah=DEFAULT_QPAH, umin=None,
                     gamma=(0.0, 0.01, 0.02, 0.05, 0.1), alpha=DEFAULT_ALPHA):
    """Table of DL2014 emissivities: ``(qpah, umin, gamma, emissivity)``.

    *emissivity* is the dust luminosity per kg of dust [W/kg] of the combined
    template — the constant CIGALE's dl2014 module divides by to report
    ``dust.mass`` (``M_dust = L_dust / emissivity``). ``umin=None`` uses every
    umin node of the CIGALE DL2014 database, giving the finest grid for
    interpolation (``simbanator.sed.cigale.pin_umin`` consumes this table).

    MUST run under the CIGALE conda env (needs ``pcigale.data``); notebooks in
    other envs subprocess ``python -m simbanator.sed.dl2014_fit
    --emissivity-out table.fits`` and read the FITS.
    """
    if umin is None:
        from pcigale.data import SimpleDatabase
        with SimpleDatabase("dl2014") as db:
            umin = tuple(sorted(db.parameters["umin"]))
    _, models = load_templates(qpah=qpah, umin=umin, gamma=gamma, alpha=alpha)
    return Table(rows=[(q, u, g, e) for q, u, g, _s, e in models],
                 names=("qpah", "umin", "gamma", "emissivity"))


def load_filters(names):
    """``{band: (wl_nm, transmission)}`` from the CIGALE filter database.

    The stored transmission is pre-normalized by CIGALE so that
    ``trapz(T * L_lambda) / (4 pi D^2)`` is directly F_nu in mJy (the exact
    convention of ``pcigale.sed.SED.compute_fnu``).
    """
    from pcigale.data import SimpleDatabase

    out = {}
    with SimpleDatabase("filters") as db:
        for name in names:
            f = db.get(name=name)
            out[name] = (np.asarray(f.wl, float), np.asarray(f.tr, float))
    return out


def model_band_fluxes(wl_rest, models, filters, redshift, dist_mpc):
    """Band fluxes of every template at 1 W of L_dust.

    Redshifting follows ``pcigale.sed_modules.redshifting`` (wl*(1+z),
    L_lambda/(1+z); IGM absorption is irrelevant in the IR) and the filter
    convolution follows ``SED.compute_fnu``. Bands not fully covered by the
    template grid are NaN.

    Returns
    -------
    ndarray (n_models, n_bands) : mJy per W, band order = ``list(filters)``.
    """
    zp1 = 1. + redshift
    wl_obs = wl_rest * zp1
    inv_4pid2 = 1. / (4. * np.pi * (dist_mpc * MPC_M) ** 2)
    out = np.full((len(models), len(filters)), np.nan)
    for j, (fwl, ftr) in enumerate(filters.values()):
        if wl_obs[0] > fwl[0] or wl_obs[-1] < fwl[-1]:
            continue                                  # not covered -> NaN
        w = (wl_obs >= fwl[0]) & (wl_obs <= fwl[-1])
        grid = np.union1d(wl_obs[w], fwl)
        tr = np.interp(grid, fwl, ftr)
        for i, (_q, _u, _g, spec, _e) in enumerate(models):
            llam = np.interp(grid, wl_obs, spec) / zp1
            out[i, j] = _trapz(tr * llam, x=grid) * inv_4pid2
    return out


def _parse_ini(run_dir):
    """(data_file, additionalerror) from a prepare_run-written pcigale.ini."""
    data_file, addl = None, 0.1
    with open(os.path.join(run_dir, "pcigale.ini"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("data_file = "):
                data_file = line[len("data_file = "):].strip().strip('"')
            elif line.startswith("additionalerror = "):
                addl = float(line.split("=", 1)[1])
    if data_file is None:
        raise ValueError(f"no data_file entry in {run_dir}/pcigale.ini")
    return data_file, addl


def fit_run_dir(run_dir, templates=None, ir_bands=IR_BAND_RE, min_bands=2,
                additionalerror=None, outname="dl2014_results.fits",
                verbose=True):
    """Fit the DL2014 IR residual for every object of one run directory.

    Parameters
    ----------
    run_dir : str
        Optical-only CIGALE run dir (``pcigale.ini`` + ``out/results.fits``).
    templates : (wl, models), optional
        :func:`load_templates` product; loaded on demand (pass it when
        looping over many run dirs).
    ir_bands : str
        Regex selecting the IR bands of the data file to fit.
    min_bands : int
        Objects with fewer finite IR bands get NaN results (``nbands`` still
        records what was available).
    additionalerror : float, optional
        Fractional error added in quadrature to F_obs; default: the value
        from the run's pcigale.ini (CIGALE's own fit-time budget).
    outname : str
        Written into ``<run_dir>/out/``.

    Returns
    -------
    str : path of the written FITS table.
    """
    results_path = os.path.join(run_dir, "out", "results.fits")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} missing — run the "
                                "optical-only CIGALE fit first")
    data_file, ini_addl = _parse_ini(run_dir)
    if additionalerror is None:
        additionalerror = ini_addl

    obs = Table.read(data_file)
    res = Table.read(results_path)
    ir_re = re.compile(ir_bands)
    bands = [c for c in obs.colnames
             if c not in ("id", "redshift", "distance")
             and not c.endswith("_err") and ir_re.search(c)]
    if not bands:
        raise ValueError(f"no IR bands matching {ir_bands!r} in {data_file}")

    if templates is None:
        templates = load_templates()
    wl_rest, models = templates
    filters = load_filters(bands)
    theta = np.array([(q, u, g) for q, u, g, _s, _e in models])
    emissivity = np.array([e for _q, _u, _g, _s, e in models])

    obs_idx = {str(i): k for k, i in enumerate(obs["id"])}
    model_cache = {}
    rows = []
    n_unpredicted = 0
    for i, oid in enumerate(np.asarray(res["id"], str)):
        k = obs_idx.get(oid)
        if k is None:
            continue
        z = float(obs["redshift"][k])
        dmpc = float(obs["distance"][k])
        zkey = (round(z, 5), round(dmpc, 3))
        if zkey not in model_cache:
            model_cache[zkey] = model_band_fluxes(wl_rest, models, filters,
                                                  z, dmpc)
        mflux = model_cache[zkey]                     # (n_models, n_bands)

        f_obs = np.array([float(obs[b][k]) for b in bands])
        e_obs = np.array([float(obs[f"{b}_err"][k])
                          if f"{b}_err" in obs.colnames else np.nan
                          for b in bands])
        f_star = np.array([float(res[f"bayes.{b}"][i])
                           if f"bayes.{b}" in res.colnames else np.nan
                           for b in bands])
        e_star = np.array([float(res[f"bayes.{b}_err"][i])
                           if f"bayes.{b}_err" in res.colnames else np.nan
                           for b in bands])
        # no stellar prediction (spectrum does not reach the band, e.g.
        # SCUBA-2/ALMA vs BC03's 160 um limit) -> subtract nothing: the
        # stellar contribution there is a negligible Rayleigh-Jeans tail
        no_pred = ~np.isfinite(f_star)
        n_unpredicted += int(np.count_nonzero(no_pred & np.isfinite(f_obs)))
        f_star = np.where(no_pred, 0.0, f_star)
        e_star = np.where(np.isfinite(e_star), e_star, 0.0)

        f_dust = f_obs - f_star
        var = (np.where(np.isfinite(e_obs), e_obs, 0.0) ** 2
               + (additionalerror * f_obs) ** 2 + e_star ** 2)
        # template coverage is a per-band property (all models share the
        # wavelength grid): an uncovered band is dropped from the fit
        covered = np.all(np.isfinite(mflux), axis=0)
        ok = np.isfinite(f_dust) & (var > 0) & covered

        row = {"id": oid, "redshift": z, "nbands": int(ok.sum()),
               "n_star_sub": int(np.count_nonzero(~no_pred & ok))}
        for jb, b in enumerate(bands):
            row[f"fobs.{b}"] = f_obs[jb]
            row[f"fstar.{b}"] = f_star[jb] if not no_pred[jb] else np.nan
            row[f"fdust.{b}"] = f_dust[jb]
            row[f"fmodel.{b}"] = np.nan

        if row["nbands"] < min_bands:
            row.update({"chi2": np.nan, "chi2_red": np.nan,
                        "flag_upperlimit": -1,
                        "best.dust.luminosity": np.nan,
                        "best.dust.mass": np.nan,
                        "best.dust.qpah": np.nan, "best.dust.umin": np.nan,
                        "best.dust.gamma": np.nan,
                        "bayes.dust.luminosity": np.nan,
                        "bayes.dust.luminosity_err": np.nan,
                        "bayes.dust.mass": np.nan,
                        "bayes.dust.mass_err": np.nan,
                        "bayes.dust.qpah": np.nan, "bayes.dust.umin": np.nan,
                        "bayes.dust.gamma": np.nan})
            rows.append(row)
            continue

        fd, v = f_dust[ok], var[ok]
        m = mflux[:, ok]                               # (n_models, n_ok)
        denom = np.sum(m * m / v, axis=1)
        a_hat = np.sum(fd * m / v, axis=1) / denom     # L_dust per model, W
        sig_a = 1. / np.sqrt(denom)
        a = np.clip(a_hat, 0., None)
        chi2 = np.sum((fd[None, :] - a[:, None] * m) ** 2 / v, axis=1)

        ib = int(np.argmin(chi2))
        w = np.exp(-0.5 * (chi2 - chi2[ib]))
        w /= w.sum()
        emis = emissivity
        th = theta

        def _bayes(x, sig=None):
            mean = np.sum(w * x)
            var_x = np.sum(w * (x ** 2 + (sig ** 2 if sig is not None
                                          else 0.))) - mean ** 2
            return mean, np.sqrt(max(var_x, 0.))

        ld_b, ld_e = _bayes(a, sig_a)
        md_b, md_e = _bayes(a / emis, sig_a / emis)
        q_b, q_e = _bayes(th[:, 0])
        u_b, u_e = _bayes(th[:, 1])
        g_b, g_e = _bayes(th[:, 2])

        row.update({
            "chi2": float(chi2[ib]),
            "chi2_red": float(chi2[ib] / max(int(ok.sum()) - 1, 1)),
            "flag_upperlimit": int(a_hat[ib] <= 0),
            "best.dust.luminosity": float(a[ib]),          # W
            "best.dust.mass": float(a[ib] / emis[ib]),     # kg (CIGALE unit)
            "best.dust.qpah": float(th[ib, 0]),
            "best.dust.umin": float(th[ib, 1]),
            "best.dust.gamma": float(th[ib, 2]),
            "bayes.dust.luminosity": float(ld_b),
            "bayes.dust.luminosity_err": float(ld_e),
            "bayes.dust.mass": float(md_b),
            "bayes.dust.mass_err": float(md_e),
            "bayes.dust.qpah": float(q_b),
            "bayes.dust.umin": float(u_b),
            "bayes.dust.gamma": float(g_b),
        })
        _mb = a[ib] * m[ib]
        for jb_ok, jb in enumerate(np.flatnonzero(ok)):
            row[f"fmodel.{bands[jb]}"] = float(_mb[jb_ok])
        rows.append(row)

    if not rows:
        raise ValueError(f"{run_dir}: no common ids between {data_file} "
                         f"and {results_path}")
    out = Table(rows=rows)
    outpath = os.path.join(run_dir, "out", outname)
    out.write(outpath, overwrite=True)
    if verbose:
        nfit = int(np.sum(out["flag_upperlimit"] >= 0))
        nul = int(np.sum(out["flag_upperlimit"] == 1))
        print(f"[dl2014] {os.path.basename(run_dir)}: {len(out)} objects "
              f"({nfit} fitted, {nul} upper limits, "
              f"{len(out) - nfit} too few bands; "
              f"{n_unpredicted} band values used unsubtracted) -> {outpath}")
    return outpath


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Standalone DL2014 fit of the IR residual of "
                    "optical-only CIGALE runs (run under the cigale env "
                    "python).")
    p.add_argument("run_dirs", nargs="*",
                   help="explicit run directories (else --run-base/--pattern)")
    p.add_argument("--run-base", help="directory containing the run dirs")
    p.add_argument("--pattern", default="optonly_dust_on_*",
                   help="glob under --run-base [%(default)s]")
    p.add_argument("--ir-bands", default=IR_BAND_RE,
                   help="regex of the bands to fit [%(default)s]")
    p.add_argument("--qpah", default=",".join(map(str, DEFAULT_QPAH)))
    p.add_argument("--umin", default=None,
                   help="comma list of umin nodes; default: DL2014 fit grid, "
                        "or EVERY database node with --emissivity-out")
    p.add_argument("--gamma", default=None,
                   help="comma list of gamma values; default: DL2014 fit "
                        "grid, or 0,0.01,0.02,0.05,0.1 with --emissivity-out")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--min-bands", type=int, default=2)
    p.add_argument("--additionalerror", type=float, default=None,
                   help="override the per-run pcigale.ini value")
    p.add_argument("--outname", default="dl2014_results.fits")
    p.add_argument("--emissivity-out", metavar="PATH.fits",
                   help="write the (qpah, umin, gamma, emissivity) table "
                        "instead of fitting run dirs, then exit")
    args = p.parse_args(argv)

    if args.emissivity_out:
        tab = emissivity_table(
            qpah=tuple(float(x) for x in args.qpah.split(",")),
            umin=(tuple(float(x) for x in args.umin.split(","))
                  if args.umin else None),
            gamma=(tuple(float(x) for x in args.gamma.split(","))
                   if args.gamma else (0.0, 0.01, 0.02, 0.05, 0.1)),
            alpha=args.alpha)
        tab.write(args.emissivity_out, overwrite=True)
        print(f"[dl2014] {len(tab)} emissivities "
              f"({len(set(tab['qpah']))} qpah x {len(set(tab['umin']))} umin "
              f"x {len(set(tab['gamma']))} gamma) -> {args.emissivity_out}")
        return 0

    args.umin = args.umin or ",".join(map(str, DEFAULT_UMIN))
    args.gamma = args.gamma or ",".join(map(str, DEFAULT_GAMMA))
    run_dirs = list(args.run_dirs)
    if args.run_base:
        run_dirs += sorted(glob.glob(os.path.join(args.run_base,
                                                  args.pattern)))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        p.error("no run directories (pass them explicitly or via "
                "--run-base/--pattern)")

    grid = [tuple(float(x) for x in getattr(args, k).split(","))
            for k in ("qpah", "umin", "gamma")]
    templates = load_templates(*grid, alpha=args.alpha)
    print(f"[dl2014] {len(templates[1])} templates "
          f"(qpah x umin x gamma = {'x'.join(str(len(g)) for g in grid)}), "
          f"alpha={args.alpha}")

    done, skipped = 0, []
    for d in run_dirs:
        if not os.path.exists(os.path.join(d, "out", "results.fits")):
            skipped.append(os.path.basename(d))
            continue
        fit_run_dir(d, templates=templates, ir_bands=args.ir_bands,
                    min_bands=args.min_bands,
                    additionalerror=args.additionalerror,
                    outname=args.outname)
        done += 1
    if skipped:
        print(f"[dl2014] skipped {len(skipped)} run dir(s) without "
              f"out/results.fits: {skipped}")
    print(f"[dl2014] {done}/{len(run_dirs)} run dirs fitted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
