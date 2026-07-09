import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from hyperion.model import ModelOutput
from astropy import units as u
from astropy import constants
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d
import os
from tqdm import trange

from pathlib import Path
from astroquery.svo_fps import SvoFps

def resolve_filter_ids(facility, instrument, filters=None):
    """
    Return dict: {short_name: full_filterID}
    """

    filt_table = SvoFps.get_filter_list(
        facility=facility,
        instrument=instrument,
        cache=False
    )

    mapping = {}
    for fid in filt_table['filterID']:
        short = fid.split('.')[-1]
        mapping[short] = fid

    # --- If no filters requested → return all ---
    if filters is None:
        return mapping

    # --- Single filter ---
    if isinstance(filters, str):
        if filters not in mapping:
            raise ValueError(f"{filters} not found in {facility}/{instrument}")
        return {filters: mapping[filters]}

    # --- List of filters ---
    out = {}
    for f in filters:
        if f in mapping:
            out[f] = mapping[f]
        else:
            print(f"Skipping {f}: not found in {facility}/{instrument}")

    return out

def get_svo_filters(facility, instrument, filters=None, wave_unit='micron'):
    """
    Retrieve SVO filter transmission curves as a dictionary.

    Parameters
    ----------
    facility : str
        e.g. 'JWST'
    instrument : str
        e.g. 'NIRCam'
    filters : list or None
        List of filter names (e.g. ['F200W','F356W']).
        If None → fetch all available filters.
    wave_unit : str
        Output wavelength unit ('micron', 'angstrom', etc.)

    Returns
    -------
    dict
        {
            'F200W': {
                'wavelength': array,
                'transmission': array
            },
            ...
        }
    """
    # --- Normalize inputs to lists ---
    if isinstance(facility, str):
        facility = [facility]
    if isinstance(instrument, str):
        instrument = [instrument]

    if len(instrument) != len(facility):
        raise ValueError("facility and instrument must have same length")

    def fetch_single_filter(filter_id, wave_unit):
        data = SvoFps.get_transmission_data(filter_id, cache=False)
    
        wl = (data['Wavelength']).to(wave_unit).value
        trans = data['Transmission']
    
        return {
            'Wavelength': wl,
            'Transmission': trans
        }

    out = {}

    # --- Loop over facility/instrument pairs ---
    for fac, inst in zip(facility, instrument):

        out.setdefault(fac, {})
        out[fac].setdefault(inst, {})
        # --- Resolve correct SVO filter IDs ---
        mapping = resolve_filter_ids(fac, inst, filters)
        
        # --- Fetch filters using FULL filter_id ---
        for short_name, filter_id in mapping.items():
            try:
                out[fac][inst][short_name] = fetch_single_filter(filter_id, wave_unit)
            except Exception as e:
                print(f"Skipping {filter_id}: {e}")

    return out


def load_local_filters(local_filters_spec, wave_unit='micron'):
    """Load filter transmission curves from local ASCII files.

    Parameters
    ----------
    local_filters_spec : dict
        ``{facility: {instrument: {filter_name: filepath}}}``.
        Each file must be two-column ASCII: wavelength (Angstrom), transmission.
    wave_unit : str
        Output wavelength unit. Default ``'micron'``.

    Returns
    -------
    dict
        Same nested format as :func:`get_svo_filters`.
    """
    out = {}
    for fac, inst_dict in local_filters_spec.items():
        out.setdefault(fac, {})
        for inst, filt_dict in inst_dict.items():
            out[fac].setdefault(inst, {})
            for fname, fpath in filt_dict.items():
                try:
                    data = np.loadtxt(fpath, comments=['#', '!'])
                    wl = (data[:, 0] * u.AA).to(wave_unit).value
                    trans = data[:, 1]
                    out[fac][inst][fname] = {'Wavelength': wl, 'Transmission': trans}
                except Exception as e:
                    print(f"Skipping local filter {fname} ({fpath}): {e}")
    return out


def magTo_mJy(mag):
    mjy = 10 ** (-mag / 2.5) * 3631 * 1e3
    return mjy
def mJyToMag(mJy):
    jy = mJy.to(u.Jy).value
    mag = -2.5*np.log10(jy/3631)
    return mag

def _trapz_weights(x):
    """Weights w such that np.trapz(y, x) == np.sum(w * y) (trapezoid quadrature)."""
    w = np.zeros(len(x), dtype=float)
    dx = np.diff(x)
    w[:-1] += dx / 2.0
    w[1:] += dx / 2.0
    return w


def convolveFilterWithSED(sedX, sedY, transX, transY, sedYerr=None):
    """Filter-convolve an SED; if `sedYerr` (same shape/units as sedY) is given,
    propagate it through the same quadrature assuming independent wavelength bins:
    sigma_conv = sqrt(sum (w_i T_i sigma_i)^2) / int T dlambda.

    Returns (xmean, realY, realYerr); realYerr is NaN*unit when sedYerr is None."""
    ind = np.where((sedX.value > np.min(transX)) & (sedX.value < np.max(transX)))[0]
    xnew = sedX[ind].value
    fluxNew = sedY[ind]

    fInterp = interp1d(transX, transY)
    ynew = fInterp(xnew)

    F = fluxNew * ynew

    yFlux = np.trapz(F, xnew)
    norm = np.trapz(ynew, xnew)
    xmean = transX[transY == np.max(transY)][0]
    realY = yFlux / norm
    if sedYerr is None:
        realYerr = np.nan * realY
    else:
        w = _trapz_weights(xnew)
        # abs(norm): with descending-wavelength SEDs (hyperion) both integrals
        # are negative — the flux ratio self-corrects but the sqrt does not
        realYerr = np.sqrt(np.sum((w * ynew * sedYerr[ind]) ** 2)) / np.abs(norm)
    return xmean, realY, realYerr

def annular_flux_table(inner, outer, verbose=True):
    """Differential (annular) fluxes between two cumulative-aperture flux tables.

    Hyperion SED apertures are cumulative — the flux within a projected
    (image-plane) radius — so the flux in the annulus r_in < R <= r_out is
    ``F(<r_out) - F(<r_in)``, band by band. Filter convolution is linear in
    flux, so differencing the convolved photometry equals convolving the
    differenced SED.

    Parameters
    ----------
    inner, outer : str or :class:`~astropy.table.Table`
        ``MakeSED.extract_flux_batch`` tables (or paths to the FITS) of the
        SAME sources/run/inclination at two consecutive aperture indices
        (*inner* = smaller radius). Sources are matched on
        ``(snap, gal_id_at_snap)``; only sources present in both survive.

    Returns
    -------
    :class:`~astropy.table.Table`
        Same schema as the inputs (``gal_id_at_snap, snap, redshift,
        '<band>', '<band>_err'``), directly consumable by
        :func:`simbanator.sed.cigale.write_cigale_input`.

    Notes
    -----
    * Non-positive or non-finite annular flux (Monte-Carlo noise / empty
      annulus) -> **NaN flux AND error** (the missing-band convention
      downstream, e.g. CIGALE ignores NaN bands).
    * Errors: both photometries come from the same photon run and the photons
      inside r_in are counted in both, so under photon independence
      ``Var(F_out) = Var(F_in) + Var(F_ann)`` and
      ``err_ann = sqrt(err_out^2 - err_in^2)``; where MC noise makes that
      non-positive, the conservative quadrature sum
      ``sqrt(err_out^2 + err_in^2)`` is used instead.
    """
    t_in = Table.read(inner) if isinstance(inner, str) else inner
    t_out = Table.read(outer) if isinstance(outer, str) else outer

    key_cols = ("gal_id_at_snap", "snap", "redshift")
    idx_in = {(int(s), int(g)): i for i, (s, g)
              in enumerate(zip(t_in["snap"], t_in["gal_id_at_snap"]))}
    rows_out, rows_in = [], []
    dropped = []
    for j, (s, g) in enumerate(zip(t_out["snap"], t_out["gal_id_at_snap"])):
        i = idx_in.get((int(s), int(g)))
        if i is None:
            dropped.append((int(s), int(g)))
        else:
            rows_out.append(j)
            rows_in.append(i)
    if not rows_out:
        raise ValueError("no (snap, gal_id_at_snap) pair present in both tables")

    bands = [c for c in t_out.colnames
             if c not in key_cols and not c.endswith("_err")]
    missing = [c for c in bands if c not in t_in.colnames]
    bands = [c for c in bands if c in t_in.colnames]

    ann = Table()
    for c in key_cols:
        ann[c] = np.asarray(t_out[c])[rows_out]

    n_nonpos, n_fallback = 0, 0
    for c in bands:
        f_out = np.asarray(t_out[c], float)[rows_out]
        f_in = np.asarray(t_in[c], float)[rows_in]
        f_ann = f_out - f_in

        ec = f"{c}_err"
        e_out = (np.abs(np.asarray(t_out[ec], float))[rows_out]
                 if ec in t_out.colnames else np.full(len(rows_out), np.nan))
        e_in = (np.abs(np.asarray(t_in[ec], float))[rows_in]
                if ec in t_in.colnames else np.full(len(rows_in), np.nan))
        with np.errstate(invalid='ignore'):
            var_ann = e_out ** 2 - e_in ** 2
            fallback = np.isfinite(var_ann) & (var_ann <= 0)
            e_ann = np.where(fallback, np.hypot(e_out, e_in), np.sqrt(var_ann))
        n_fallback += int(fallback.sum())

        bad = ~np.isfinite(f_ann) | (f_ann <= 0)
        n_nonpos += int((np.isfinite(f_ann) & (f_ann <= 0)).sum())
        f_ann[bad] = np.nan
        e_ann[bad] = np.nan
        ann[c] = f_ann
        ann[ec] = e_ann

    if verbose:
        n_meas = len(ann) * len(bands)
        print(f"[annular_flux_table] {len(ann)} sources x {len(bands)} bands; "
              f"{n_nonpos}/{n_meas} non-positive annular fluxes -> NaN; "
              f"{n_fallback} error(s) via quadrature-sum fallback")
        if dropped:
            print(f"[annular_flux_table]   dropped (missing in inner table): {dropped}")
        if missing:
            print(f"[annular_flux_table]   bands absent from inner table, skipped: {missing}")
    return ann


def flux_extraction(facility, instrument, wav, flux, filters=None, wave_unit='micron', filter_list=None,
                    flux_unc=None):
    """Per-filter photometry from an SED.

    If `flux_unc` (same shape/units as `flux`, e.g. the Hyperion Monte-Carlo
    uncertainty) is given, each filter entry also carries 'mJy_err' / 'mag_err'
    propagated through the same filter convolution; otherwise they are NaN.
    """

    from astropy import units as u
    import numpy as np

    # --- Ensure wavelength has units ---
    if not hasattr(wav, 'unit'):
        raise ValueError("wav must be an astropy Quantity with units")

    # Convert SED wavelength to target unit
    wav = wav.to(wave_unit)
    if filter_list is not None:
        profiles = filter_list
    else:
        profiles = get_svo_filters(
            facility,
            instrument,
            filters=filters,
            wave_unit=wave_unit
        )

    results = {}

    for fac in profiles:
        results.setdefault(fac, {})

        for inst in profiles[fac]:
            results[fac].setdefault(inst, {})

            for f in profiles[fac][inst]:

                filtw = profiles[fac][inst][f]['Wavelength']
                filtf = profiles[fac][inst][f]['Transmission']

                # --- Ensure numpy arrays ---
                filtw = np.asarray(filtw)
                filtf = np.asarray(filtf)

                # --- Check overlap ---
                mask = (wav.value > filtw.min()) & (wav.value < filtw.max())

                if np.sum(mask) < 5:
                    # Not enough overlap → skip
                    continue

                try:
                    xmean, flux_conv, flux_conv_err = convolveFilterWithSED(
                        wav, flux, filtw, filtf, sedYerr=flux_unc
                    )

                    # --- Convert flux ---
                    mJy = flux_conv
                    mag = mJyToMag(mJy)
                    mJy_err = flux_conv_err
                    with np.errstate(all='ignore'):
                        # dm = 2.5/ln(10) * (sigma_f / f)
                        mag_err = float(2.5 / np.log(10) * (mJy_err / mJy).decompose().value) \
                            if np.isfinite(mJy_err.value) else np.nan

                    results[fac][inst][f] = {
                        'xmean': xmean,
                        'mJy': mJy,
                        'mag': mag,
                        'mJy_err': mJy_err,
                        'mag_err': mag_err,
                    }

                except Exception as e:
                    print(f"Skipping {fac}/{inst}/{f}: {e}")

    return results


    