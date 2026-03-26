import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from hyperion.model import ModelOutput
from astropy import units as u
from astropy import constants
from astropy.io import fits, ascii
from scipy.interpolate import interp1d
from astropy.cosmology import LambdaCDM, FlatwCDM
import os
from tqdm import trange

from pathlib import Path
from astroquery.svo_fps import SvoFps

cosmo = LambdaCDM(H0 = 70, Om0= 0.3, Ode0= 0.7)

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

def magTo_mJy(mag):
    mjy = 10 ** (-mag / 2.5) * 3631 * 1e3
    return mjy
def mJyToMag(mJy):
    jy = mJy.to(u.Jy).value
    mag = -2.5*np.log10(jy/3631)
    return mag

def convolveFilterWithSED(sedX, sedY, transX, transY):
    ind = np.where((sedX.value > np.min(transX)) & (sedX.value < np.max(transX)))[0]
    xnew = sedX[ind].value
    fluxNew = sedY[ind]

    fInterp = interp1d(transX, transY)
    ynew = fInterp(xnew)

    F = fluxNew * ynew

    yFlux = np.trapz(F, xnew)
    norm = np.trapz(ynew, xnew)
    xmean = transX[transY == np.max(transY)]
    realY = yFlux / norm
    return xmean, realY

def flux_extraction(facility, instrument, wav, flux, filters=None, wave_unit='micron'):

    from astropy import units as u
    import numpy as np

    # --- Ensure wavelength has units ---
    if not hasattr(wav, 'unit'):
        raise ValueError("wav must be an astropy Quantity with units")

    # Convert SED wavelength to target unit
    wav = wav.to(wave_unit)

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
                    xmean, flux_conv = convolveFilterWithSED(
                        wav, flux, filtw, filtf
                    )

                    # --- Convert flux ---
                    mJy = flux_conv
                    mag = mJyToMag(mJy)

                    results[fac][inst][f] = {
                        'xmean': xmean,
                        'mJy': mJy,
                        'mag': mag
                    }

                except Exception as e:
                    print(f"Skipping {fac}/{inst}/{f}: {e}")

    return results


    