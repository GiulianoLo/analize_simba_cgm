import os
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator

try:
    from astropy.io import fits
except ImportError:
    fits = None


def _save_quenching_fits(
    output_path,
    galaxy_id,
    quench_times,
    sft_times,
    persistence_end_times,
    rejuvenation_times,
):
    if fits is None:
        raise ImportError("astropy is required to save FITS output. Install with `pip install astropy`.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    galaxy_id_str = '' if galaxy_id is None else str(galaxy_id)
    n_events = len(quench_times)
    if n_events == 0:
        return

    # New rows (strict dtypes for interoperability, e.g. TOPCAT)
    new_galaxy_id = np.array([galaxy_id_str] * n_events, dtype='S64')
    new_event_id = np.arange(1, n_events + 1, dtype=np.int32)
    new_sft = np.asarray(sft_times, dtype=np.float64)
    new_qt = np.asarray(quench_times, dtype=np.float64)
    new_persist = np.asarray(persistence_end_times, dtype=np.float64)
    new_rejuv = np.asarray(rejuvenation_times, dtype=np.float64)

    if os.path.exists(output_path):
        with fits.open(output_path) as hdul:
            if len(hdul) >= 2 and hdul[1].name == 'EVENTS' and hdul[1].data is not None:
                existing = hdul[1].data
                prev_galaxy = np.asarray(existing['GALAXY_ID']).astype('S64')
                prev_event = np.asarray(existing['EVENT_ID'], dtype=np.int32)
                prev_sft = np.asarray(existing['SFT_YR'], dtype=np.float64)
                prev_qt = np.asarray(existing['QT_YR'], dtype=np.float64)
                prev_persist = np.asarray(existing['PERSIST_END_YR'], dtype=np.float64)
                prev_rejuv = np.asarray(existing['REJUV_YR'], dtype=np.float64)

                new_galaxy_id = np.concatenate([prev_galaxy, new_galaxy_id])
                new_event_id = np.concatenate([prev_event, new_event_id])
                new_sft = np.concatenate([prev_sft, new_sft])
                new_qt = np.concatenate([prev_qt, new_qt])
                new_persist = np.concatenate([prev_persist, new_persist])
                new_rejuv = np.concatenate([prev_rejuv, new_rejuv])

    events_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='GALAXY_ID', array=new_galaxy_id, format='64A'),
        fits.Column(name='EVENT_ID', array=new_event_id, format='J'),
        fits.Column(name='SFT_YR', array=new_sft, format='D'),
        fits.Column(name='QT_YR', array=new_qt, format='D'),
        fits.Column(name='PERSIST_END_YR', array=new_persist, format='D'),
        fits.Column(name='REJUV_YR', array=new_rejuv, format='D'),
    ], name='EVENTS')
    fits.HDUList([fits.PrimaryHDU(), events_hdu]).writeto(output_path, overwrite=True)


def load_quenching_events(
    fits_path='output/quenching_times/quenching_times.fits',
    galaxy_id=None,
):
    """Load saved quenching events and optionally filter by galaxy id.

    Parameters
    ----------
    fits_path : str
        Path to the quenching events FITS file.
    galaxy_id : str or int or None
        If provided, only rows matching this galaxy id are returned.

    Returns
    -------
    np.recarray
        EVENTS table rows (possibly filtered).
    """
    if fits is None:
        raise ImportError("astropy is required to read FITS output. Install with `pip install astropy`.")

    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        if len(hdul) < 2 or hdul[1].name != 'EVENTS':
            raise ValueError(f"No EVENTS table found in: {fits_path}")
        data = hdul[1].data

    if data is None:
        empty_dtype = [
            ('GALAXY_ID', 'U64'),
            ('EVENT_ID', np.int32),
            ('SFT_YR', np.float64),
            ('QT_YR', np.float64),
            ('PERSIST_END_YR', np.float64),
            ('REJUV_YR', np.float64),
        ]
        return np.rec.array([], dtype=empty_dtype)

    if galaxy_id is None:
        return data

    galaxy_id_str = str(galaxy_id)
    raw_ids = np.asarray(data['GALAXY_ID'])
    if raw_ids.dtype.kind == 'S':
        raw_ids = np.char.decode(raw_ids, 'ascii', errors='ignore')
    else:
        raw_ids = raw_ids.astype(str)
    mask = raw_ids == galaxy_id_str
    return data[mask]

def find_quenching_times(
    tarr,
    ssfr,
    galaxy_id=None,
    min_duration=None,
    max_events=3,
    smooth_window=3,
    eps=1e-14,
    interp_factor=20,        # density of interpolation grid
    tolerance=0.1,           # dex tolerance for 1/t persistence
    post_quench_frac=0.2,
    save_fits_path='output/quenching_times/quenching_times.fits',
    plot=True,
    return_debug=False
):
    """
    Quenching finder with strict 1/t persistence.

    Event logic:
    - Find a crossing below 1/t (SFT)
    - Find the subsequent crossing below 0.2/t (QT)
    - Require sSFR to stay below 1/t from SFT through QT + persistence time

    Persistence time is `post_quench_frac * QT` by default. If
    `min_duration` is provided, it overrides this with a fixed duration.

    Parameters
    ----------
    save_fits_path : str or None
        FITS output path relative to the current working directory by default
        (`output/quenching_times/quenching_times.fits`). Set to None to disable saving.
        Events from each call are appended into one table.
    """

    tarr = np.asarray(tarr)
    ssfr = np.asarray(ssfr)
    ssfr = np.clip(ssfr, eps, None)
    log_ssfr = np.log10(ssfr)

    # Optional smoothing
    if smooth_window > 1:
        log_ssfr = uniform_filter1d(log_ssfr, size=smooth_window)

    # Interpolation (PCHIP)
    interp = PchipInterpolator(tarr, log_ssfr)
    t_dense = np.linspace(tarr.min(), tarr.max(), len(tarr) * interp_factor)
    log_ssfr_dense = interp(t_dense)

    # Thresholds
    log_tau1_dense = np.log10(1 / t_dense)
    log_tau02_dense = np.log10(0.2 / t_dense)

    events = []
    i = 0

    while i < len(t_dense) - 1 and len(events) < max_events:

        # Find SFT (cross below 1/t)
        found_sft = False
        while i < len(t_dense) - 1:
            if log_ssfr_dense[i] > log_tau1_dense[i] and log_ssfr_dense[i+1] <= log_tau1_dense[i+1]:
                sft = t_dense[i+1]
                found_sft = True
                break
            i += 1
        if not found_sft:
            break

        # Find quenching (cross below 0.2/t) while keeping sSFR below 1/t
        found_qt = False
        restarted = False
        while i < len(t_dense) - 1:
            crosses_above_tau1 = (
                log_ssfr_dense[i] <= (log_tau1_dense[i] + tolerance)
                and log_ssfr_dense[i + 1] > (log_tau1_dense[i + 1] + tolerance)
            )
            if crosses_above_tau1:
                i += 1
                restarted = True
                break

            if (
                log_ssfr_dense[i] > log_tau02_dense[i]
                and log_ssfr_dense[i + 1] <= log_tau02_dense[i + 1]
            ):
                qt = t_dense[i + 1]
                found_qt = True
                break
            i += 1

        if restarted:
            continue

        if not found_qt:
            break

        # Strict persistence check against 1/t:
        # from SFT until QT + persistence duration.
        persistence_duration = min_duration if min_duration is not None else post_quench_frac * qt
        end_t = qt + persistence_duration
        interval_mask = (t_dense >= sft) & (t_dense <= end_t)

        if np.any(interval_mask):
            stays_below_tau1 = np.all(
                log_ssfr_dense[interval_mask] <= (log_tau1_dense[interval_mask] + tolerance)
            )
        else:
            stays_below_tau1 = False

        # Move cursor to the end of the checked window
        j = np.searchsorted(t_dense, end_t, side='right')

        if stays_below_tau1:
            events.append((qt, sft, end_t))
            i = j
        else:
            i += 1

    rejuvenation_times = []
    for _, _, end_t in events:
        k = np.searchsorted(t_dense, end_t, side='left')
        rejuvenation_time = np.nan
        while k < len(t_dense) - 1:
            if (
                log_ssfr_dense[k] <= (log_tau1_dense[k] + tolerance)
                and log_ssfr_dense[k + 1] > (log_tau1_dense[k + 1] + tolerance)
            ):
                rejuvenation_time = t_dense[k + 1]
                break
            k += 1
        rejuvenation_times.append(rejuvenation_time)

    # Outputs
    quench_times = [qt for qt, _, _ in events]
    sft_times = [sft for _, sft, _ in events]
    persistence_end_times = [end_t for _, _, end_t in events]
    time_since_quench = [tarr[-1] - qt for qt in quench_times]

    if save_fits_path:
        _save_quenching_fits(
            output_path=save_fits_path,
            galaxy_id=galaxy_id,
            quench_times=quench_times,
            sft_times=sft_times,
            persistence_end_times=persistence_end_times,
            rejuvenation_times=rejuvenation_times,
        )

    # Plotting
    if plot:
        import matplotlib.pyplot as plt
        t_gyr = tarr / 1e9
        t_dense_gyr = t_dense / 1e9

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(t_gyr, ssfr, 'o', label='Raw sSFR')
        ax.plot(t_dense_gyr, 10**log_ssfr_dense, '-', label='Interpolated')
        ax.plot(t_dense_gyr, 10**log_tau1_dense, '--', label='1/t')
        ax.plot(t_dense_gyr, 10**log_tau02_dense, ':', label='0.2/t')

        for sft, qt, end_t in zip(sft_times, quench_times, persistence_end_times):
            ax.axvspan(sft/1e9, qt/1e9, alpha=0.3, label='Quenching phase')
            ax.axvspan(qt/1e9, end_t/1e9, alpha=0.15, label='Persistence')

        ax.set_xlabel('Cosmic Time [Gyr]')
        ax.set_ylabel('sSFR [1/yr]')
        ax.set_yscale('log')
        ax.set_title('Quenching Detection')

        handles, labels = ax.get_legend_handles_labels()
        from collections import OrderedDict
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    if return_debug:
        return quench_times, sft_times, time_since_quench, {
            "t_dense": t_dense,
            "log_ssfr_dense": log_ssfr_dense,
            "log_tau1_dense": log_tau1_dense,
            "log_tau02_dense": log_tau02_dense,
            "persistence_end_times": persistence_end_times,
            "rejuvenation_times": rejuvenation_times
        }

    return quench_times, sft_times, time_since_quench