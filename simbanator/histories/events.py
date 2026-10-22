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
    fits_path=None,
    sim_name='default',
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

    if fits_path is None:
        fits_path = os.path.join(
            os.getcwd(), 'output', sim_name, 'quenching_times', 'quenching_times.fits'
        )

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
    save_fits_path='default',
    sim_name='default',
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
    save_fits_path : {'default', str, None}
        FITS output path relative to the current working directory by default
        (``output/<sim_name>/quenching_times/quenching_times.fits``).
        Set to ``None`` to disable saving.
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

    if save_fits_path == 'default':
        save_fits_path = os.path.join(
            os.getcwd(), 'output', sim_name, 'quenching_times', 'quenching_times.fits'
        )

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


# ═════════════════════════════════════════════════════════════════════════════
# Quench events per galaxy, critical points, values at stages, and clocks
#
# Self-contained (numpy only, plus the finder above): tests load this file by path.
# ═════════════════════════════════════════════════════════════════════════════

def ssfr_track(t_yr, mstar, sfr, keep_zero=True):
    """One galaxy's sSFR(t) ready for :func:`find_quenching_times`: valid snapshots, sorted, duplicate times collapsed.

    ``keep_zero=True`` keeps the SFR = 0 snapshots (the finder clips them to ``eps``): they carry
    the quench, and dropping them flattens SIMBA's fast mode. ``keep_zero=False`` reproduces the
    products built with ``sSFR > 0`` (ks_tracks, powderday_flux Part 2, quench_mode).
    Returns ``(t_unique, ssfr_unique)``.
    """
    t_yr = np.asarray(t_yr, float); mstar = np.asarray(mstar, float); sfr = np.asarray(sfr, float)
    with np.errstate(all="ignore"):
        ssfr = np.where(mstar > 0, sfr / mstar, np.nan)
    valid = np.isfinite(ssfr) & ((ssfr >= 0) if keep_zero else (ssfr > 0)) & np.isfinite(t_yr)
    t, s = t_yr[valid], ssfr[valid]
    o = np.argsort(t); t, s = t[o], s[o]
    tu, ui = np.unique(t, return_index=True)
    return tu, s[ui]


def quench_event(t_yr, mstar, sfr, keep_zero=True, min_valid=5, galaxy_id=None, **finder_kwargs):
    """The quench event of one galaxy history, with :func:`find_quenching_times`.

    The LAST event is used (the one the galaxy is still in at the anchor). Returns a dict with
    ``t_sft``, ``t_qt``, ``tau_q`` [yr], ``t_post_quench`` (end of the persistence window),
    ``t_sft_first`` / ``t_qt_first`` (the first event), and ``n_events``; NaN when the history has
    fewer than ``min_valid`` valid snapshots or no event. ``finder_kwargs`` go to the finder
    (e.g. ``smooth_window``).
    """
    out = dict(t_sft=np.nan, t_qt=np.nan, tau_q=np.nan, t_post_quench=np.nan,
               t_sft_first=np.nan, t_qt_first=np.nan, n_events=0)
    tu, su = ssfr_track(t_yr, mstar, sfr, keep_zero=keep_zero)
    if len(tu) < min_valid:
        return out
    qts, sfts, _, dbg = find_quenching_times(tu, su, galaxy_id=galaxy_id, plot=False, save_fits_path=None,
                                             return_debug=True, **finder_kwargs)
    qts, sfts = np.asarray(qts, float), np.asarray(sfts, float)
    out["n_events"] = int(len(qts))
    if not len(qts):
        return out
    k, kf = int(np.argmax(qts)), int(np.argmin(qts))
    out.update(t_qt=float(qts[k]), t_sft=float(sfts[k]), t_qt_first=float(qts[kf]), t_sft_first=float(sfts[kf]))
    out["tau_q"] = out["t_qt"] - out["t_sft"]
    pe = dbg.get("persistence_end_times") if isinstance(dbg, dict) else None
    if pe is not None and len(pe) > k and np.isfinite(pe[k]):
        out["t_post_quench"] = float(pe[k])
    return out


def quench_records(P, t_yr, redshift, galaxy_ids, cols, age_of_z_gyr=None, keep_zero=True, **finder_kwargs):
    """:func:`quench_event` for every selected column of a history block ``P`` (``{property: (n_snap, n_gal)}``).

    Every column gets a record (``gid``, ``col``, the event times, ``tau_q``, ``z_qt``, and
    ``tau_q_over_tH`` when ``age_of_z_gyr(z) -> t_H [Gyr]`` is given); galaxies without an event
    keep NaN times.
    """
    t_yr = np.asarray(t_yr, float); redshift = np.asarray(redshift, float)
    out = []
    for col in np.asarray(cols, int):
        gid = int(galaxy_ids[col])
        ev = quench_event(t_yr, P["masses.stellar"][:, col], P["sfr"][:, col], keep_zero=keep_zero,
                          galaxy_id=gid, **finder_kwargs)
        rec = dict(gid=gid, col=int(col), **ev, z_qt=np.nan, tau_q_over_tH=np.nan)
        if np.isfinite(ev["t_qt"]):
            rec["z_qt"] = float(np.interp(ev["t_qt"], t_yr[::-1], redshift[::-1]))
            if age_of_z_gyr is not None:
                rec["tau_q_over_tH"] = ev["tau_q"] / (float(age_of_z_gyr(rec["z_qt"])) * 1e9)
        out.append(rec)
    return out


def track_start(t_yr, mstar, sfr):
    """Earliest time with a valid star-forming track (finite, positive sSFR); NaN if never."""
    t_yr = np.asarray(t_yr, float)
    with np.errstate(all="ignore"):
        ssfr = np.where(np.asarray(mstar, float) > 0, np.asarray(sfr, float) / np.asarray(mstar, float), np.nan)
    valid = np.isfinite(ssfr) & (ssfr > 0) & np.isfinite(t_yr)
    return float(np.nanmin(t_yr[valid])) if valid.any() else np.nan


def first_downward_crossing(t, y, threshold):
    """Linearly interpolated time where ``y`` first crosses ``threshold`` going down; NaN if it never does."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    d = y - threshold
    for i in range(len(d) - 1):
        if d[i] > 0 and d[i + 1] <= 0:
            return t[i] + d[i] / (d[i] - d[i + 1]) * (t[i + 1] - t[i])
    return np.nan


def values_at_rows(arr, rows, cols):
    """``arr[row, col]`` for each (row, col) pair (row < 0 -> NaN): a property at each galaxy's stage row."""
    arr = np.asarray(arr)
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    out = np.full(len(rows), np.nan)
    ok = rows >= 0
    out[ok] = arr[rows[ok], cols[ok]]
    return out


def ratio_at_rows(num, den, rows, cols):
    """``num / den`` at each (row, col) pair; NaN where the denominator is not positive."""
    a, b = values_at_rows(num, rows, cols), values_at_rows(den, rows, cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def regrid(t, y, grid, nmin=3):
    """One galaxy's track y(t) on ``grid`` (no extrapolation: NaN outside its sampled range; all NaN below ``nmin`` points).

    ``t`` is usually the time relative to an event (``t - t_SFT``).
    """
    t, y, grid = np.asarray(t, float), np.asarray(y, float), np.asarray(grid, float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < nmin:
        return np.full(len(grid), np.nan)
    return np.interp(grid, t[ok], y[ok], left=np.nan, right=np.nan)


def log_track(t, q, nmin=3):
    """``(t sorted, log10 q)`` over the snapshots where q is finite and positive; ``(None, None)`` below ``nmin``."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    good = np.isfinite(q) & (q > 0) & np.isfinite(t)
    if good.sum() < nmin:
        return None, None
    ts, ys = t[good], np.log10(q[good])
    o = np.argsort(ts)
    return ts[o], ys[o]


def aligned_stack(t_yr, Q, t0_yr, grid_gyr, log=True, nmin=3):
    """Each galaxy's whole history re-zeroed on its own event time and interpolated onto ``grid_gyr``.

    ``Q`` is ``(n_snap, n_gal)``, ``t0_yr`` one event time per galaxy (NaN rows stay NaN). With
    ``log`` the stack holds log10 of the positive values. Returns ``(n_gal, n_grid)``.
    """
    Q = np.asarray(Q, float); t_yr = np.asarray(t_yr, float); t0 = np.asarray(t0_yr, float)
    out = np.full((Q.shape[1], len(grid_gyr)), np.nan)
    for j in range(Q.shape[1]):
        if not np.isfinite(t0[j]):
            continue
        if log:
            ts, ys = log_track(t_yr, Q[:, j], nmin=nmin)
        else:
            ok = np.isfinite(Q[:, j]) & np.isfinite(t_yr)
            ts, ys = (np.sort(t_yr[ok]), Q[:, j][ok][np.argsort(t_yr[ok])]) if ok.sum() >= nmin else (None, None)
        if ts is None:
            continue
        out[j] = np.interp(grid_gyr, (ts - t0[j]) / 1e9, ys, left=np.nan, right=np.nan)
    return out


def binned_event_profile(t_yr, values, t0_yr, xwin_gyr=2.0, nbins=12, log=True, ref0=False):
    """Per galaxy, the median of its values in bins of ``t - t0`` [Gyr] over ``[-xwin, +xwin]``.

    ``values`` is ``(n_snap, n_gal)``; with ``log`` non-positive values are dropped and log10 is
    taken. ``ref0=True`` subtracts each galaxy's own value in the bin containing 0 (or the
    nearest filled bin), so the profile is the change about the event and immune to how the
    contributing cohort changes. Returns ``(bin centres, (n_gal, nbins) profile)``.
    """
    t_yr = np.asarray(t_yr, float); V = np.asarray(values, float); t0 = np.asarray(t0_yr, float)
    xe = np.linspace(-xwin_gyr, xwin_gyr, nbins + 1); xc = 0.5 * (xe[:-1] + xe[1:])
    prof = np.full((V.shape[1], nbins), np.nan)
    for j in range(V.shape[1]):
        if not np.isfinite(t0[j]):
            continue
        tt = (t_yr - t0[j]) / 1e9
        with np.errstate(divide="ignore", invalid="ignore"):
            vv = np.where(V[:, j] > 0, np.log10(V[:, j]), np.nan) if log else V[:, j]
        for b in range(nbins):
            sel = (tt >= xe[b]) & (tt < xe[b + 1]) & np.isfinite(vv)
            if sel.any():
                prof[j, b] = np.nanmedian(vv[sel])
    if ref0:
        b0 = int(np.clip(np.searchsorted(xe, 0.0) - 1, 0, nbins - 1))
        for j in range(prof.shape[0]):
            fin = np.where(np.isfinite(prof[j]))[0]
            if len(fin):
                bref = b0 if np.isfinite(prof[j, b0]) else fin[np.argmin(np.abs(fin - b0))]
                prof[j] = prof[j] - prof[j, bref]
    return xc, prof


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════

# ── stages / apertures ────────────────────────────────────────────────────────────────────────────
STAGES_BH = ["agn_ign", "jet_on"]   # AGN ignition / jet-mode onset (m25 Part 2d windows table; attach_bh_stages), 2026-08-28

STAGES_KS = ["sf_peak", "ssfr_min", "sft", "qt", "post_quench", "gas_min", "anchor", "end"] + STAGES_BH

STAGES_PLOT = ["sf_peak", "sft", "qt", "post_quench", "gas_min", "anchor"]   # time order (summary / t_dep clock)

FH2_MIN_END = 1e-4        # `end` stage: the anchor when M_H2/M_star (history) > this, else the last snapshot above it

# ── critical epochs (port of quench_mode_vs_sigma_gas build_records, trimmed to the KS stages) ────
def _smooth(y, w=3):
    n = len(y)
    h = w // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = y[max(0, i - h):min(n, i + h + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out

def _ordered_positive(q, valid, t):
    d = np.where(valid, q, np.nan).astype(float)
    pos = np.isfinite(d) & (d > 0)
    if pos.sum() < 2:
        return None, None
    order = np.where(pos)[0]
    order = order[np.argsort(t[order])]
    return order, _smooth(d[order])

def _peak_time(q, valid, t):
    order, ds = _ordered_positive(q, valid, t)
    if order is None or not np.isfinite(ds).any():
        return np.nan
    return t[order[np.nanargmax(ds)]]

def _trough_before_floor(q, valid, t, floor_frac=1e-6):
    order, ds = _ordered_positive(q, valid, t)
    if order is None or not np.isfinite(ds).any():
        return np.nan
    floor = floor_frac * np.nanmax(ds)
    below = np.isfinite(ds) & (ds <= floor)
    cut = int(np.argmax(below)) if below.any() else len(ds)
    cut = max(cut, 2)
    seg = ds[:cut]
    if not np.isfinite(seg).any():
        return np.nan
    return t[order[np.nanargmin(seg)]]

def build_stage_records(P, t_cosmic_yr, redshift, galaxy_ids, cols, find_quenching_times,
                        age_of_z_gyr=None, stages=STAGES_KS, min_valid=5, gas_key="masses.H2",
                        fh2_min=FH2_MIN_END):
    """Critical epochs of every selected history column (quench_mode_vs_sigma_gas §3, KS stages).

    P : {property: (n_snap, n_gal) array} with row 0 = anchor; t_cosmic_yr / redshift per row;
    cols : column indices of the sample galaxies; find_quenching_times : the simbanator finder
    (injected); age_of_z_gyr(z) -> t_H [Gyr] (for tau_q / t_H; optional).
    Every galaxy gets a record (unlike the original, which skipped event-less galaxies): stages
    without a definition carry t_<stage>=NaN and row_<stage>=-1. The LAST quench event is used
    (k = argmax(qts)), the same rule as the m25 selection table.

    Departures from the residual_dust/quench_mode definitions, specific to a KS track:
    * ``gas_min`` is the trough (before any floor) of ``gas_key`` — the H2 mass when the history has
      it, else ``masses.gas`` — searched only AFTER t_QT (after the sSFR peak for event-less
      galaxies), i.e. the post-quench molecular depletion, not the formation-epoch minimum.
    * a stage whose time lies beyond the anchor (``post_quench`` = QT + persistence often does)
      keeps its t_<stage> but gets row_<stage> = -1: it is not observable inside the history and
      must not alias onto the anchor row.
    * ``end`` (the drawn track endpoint) is the anchor row when the anchor's catalogue molecular
      fraction ``masses.H2 / masses.stellar`` exceeds ``fh2_min``; otherwise the LATEST history row
      whose fraction does (``end_is_anchor`` False, ``t_end`` / ``row_end`` of that snapshot; -1 when no
      row qualifies). Without an H2 column in ``P`` the end is always the anchor.
    """
    t_cosmic_yr = np.asarray(t_cosmic_yr, float)
    redshift = np.asarray(redshift, float)
    galaxy_ids = np.asarray(galaxy_ids)
    cols = np.asarray(cols, int)

    def _nearest_row(t_target):
        return nearest_row(t_cosmic_yr, t_target)

    records = []
    for col in cols:
        gid = galaxy_ids[col]
        mstar = np.asarray(P["masses.stellar"][:, col], float)
        sfr = np.asarray(P["sfr"][:, col], float)
        gas = np.asarray(P["masses.gas"][:, col], float)
        with np.errstate(all="ignore"):
            ssfr = np.where(mstar > 0, sfr / mstar, np.nan)
        valid = np.isfinite(ssfr) & (ssfr > 0) & np.isfinite(t_cosmic_yr)
        rec = dict(gid=int(gid), col=int(col), n_valid=int(valid.sum()), n_events=0,
                   t_sft=np.nan, t_qt=np.nan, t_post_quench=np.nan, tau_q=np.nan,
                   tau_q_over_tH=np.nan, z_qt=np.nan, t_qt_first=np.nan, t_sft_first=np.nan)
        if valid.sum() >= min_valid:
            t = t_cosmic_yr[valid]
            sv = ssfr[valid]
            o = np.argsort(t)
            t, sv = t[o], sv[o]
            tu, ui = np.unique(t, return_index=True)
            su = sv[ui]
            if len(tu) >= min_valid:
                qts, sfts, _, dbg = find_quenching_times(
                    tu, su, galaxy_id=int(gid), plot=False, save_fits_path=None, return_debug=True)
                qts = np.asarray(qts, float)
                sfts = np.asarray(sfts, float)
                rec["n_events"] = int(len(qts))
                if len(qts):
                    k = int(np.argmax(qts))
                    rec["t_qt"], rec["t_sft"] = float(qts[k]), float(sfts[k])
                    pe = dbg.get("persistence_end_times") if isinstance(dbg, dict) else None
                    if pe is not None and len(pe) > k and np.isfinite(pe[k]):
                        rec["t_post_quench"] = float(pe[k])
                    kf = int(np.argmin(qts))
                    rec["t_qt_first"], rec["t_sft_first"] = float(qts[kf]), float(sfts[kf])
                    rec["tau_q"] = rec["t_qt"] - rec["t_sft"]
                    rec["z_qt"] = float(np.interp(rec["t_qt"], t_cosmic_yr[::-1], redshift[::-1]))
                    if age_of_z_gyr is not None:
                        rec["tau_q_over_tH"] = rec["tau_q"] / (float(age_of_z_gyr(rec["z_qt"])) * 1e9)
        t_sfpeak = _peak_time(ssfr, valid, t_cosmic_yr)
        gq = np.asarray(P[gas_key][:, col], float) if gas_key in P else gas
        t_start = rec["t_qt"] if np.isfinite(rec["t_qt"]) else t_sfpeak
        after = valid & (t_cosmic_yr >= t_start) if np.isfinite(t_start) else valid
        t_gasmin = _trough_before_floor(gq, after, t_cosmic_yr)
        if not np.isfinite(t_gasmin):                      # e.g. gas already at the floor: fall back
            t_gasmin = _trough_before_floor(gas, after, t_cosmic_yr)
        t_times = {
            "sf_peak": t_sfpeak,
            "ssfr_min": _trough_before_floor(ssfr, valid, t_cosmic_yr),
            "sft": rec["t_sft"], "qt": rec["t_qt"], "post_quench": rec["t_post_quench"],
            "gas_min": t_gasmin,
            "anchor": float(t_cosmic_yr[0]),
        }
        t_anchor = float(t_cosmic_yr[0])
        # end = anchor, or the last snapshot with a measurable H2 fraction (history level)
        with np.errstate(all="ignore"):
            fh2 = np.where(mstar > 0, np.asarray(P[gas_key][:, col], float) / mstar, np.nan) \
                if gas_key in P else np.full(len(mstar), np.nan)
        rec["fh2_anchor"] = float(fh2[0]) if np.isfinite(fh2[0]) else np.nan
        if gas_key not in P or (np.isfinite(fh2[0]) and fh2[0] > fh2_min):
            row_end, end_is_anchor = 0, True
        else:
            ok_h2 = np.isfinite(fh2) & (fh2 > fh2_min) & np.isfinite(t_cosmic_yr)
            row_end = int(np.where(ok_h2)[0][np.argmax(t_cosmic_yr[ok_h2])]) if ok_h2.any() else -1
            end_is_anchor = False
        rec["end_is_anchor"] = bool(end_is_anchor)
        t_times["end"] = float(t_cosmic_yr[row_end]) if row_end >= 0 else np.nan
        for st in stages:
            tt = t_times.get(st, np.nan)
            rec["t_%s" % st] = float(tt) if np.isfinite(tt) else np.nan
            if st == "anchor":
                rec["row_%s" % st] = 0
            elif st == "end":
                rec["row_%s" % st] = int(row_end)
            elif np.isfinite(tt) and tt > t_anchor + 1.0:      # beyond the history (1 yr tolerance)
                rec["row_%s" % st] = -1
            else:
                rec["row_%s" % st] = _nearest_row(tt)
        records.append(rec)
    return records

def stage_time_order(rec, stages=STAGES_PLOT):
    """Stages of one record that are defined, sorted by their cosmic time (for drawing a track)."""
    have = [(rec["t_%s" % st], st) for st in stages
            if np.isfinite(rec.get("t_%s" % st, np.nan)) and rec.get("row_%s" % st, -1) >= 0]
    return [st for _, st in sorted(have, key=lambda x: x[0])]

def nearest_row(t_cosmic_yr, t_target, t_anchor=None, tol_yr=1.0):
    """Index of the history row nearest to `t_target` [yr]; -1 when the time is not finite or (with `t_anchor`)
    lies more than `tol_yr` beyond the anchor, i.e. outside the history."""
    if not np.isfinite(t_target):
        return -1
    if t_anchor is not None and t_target > float(t_anchor) + tol_yr:
        return -1
    return int(np.argmin(np.abs(np.asarray(t_cosmic_yr, float) - float(t_target))))

# ── histories on a common clock (notebook Part 5: evolutionary properties of the KS regions) ─────
def interp_track(t, y, grid):
    """One galaxy's y(t) linearly interpolated onto `grid`; NaN outside the coverage of its finite samples
    (no extrapolation), non-finite samples dropped, duplicate times collapsed; all-NaN with < 2 samples."""
    t, y, grid = np.asarray(t, float), np.asarray(y, float), np.asarray(grid, float)
    out = np.full(len(grid), np.nan)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return out
    tt, ui = np.unique(t[ok], return_index=True)
    yy = y[ok][ui]
    if len(tt) < 2:
        return out
    inside = (grid >= tt[0]) & (grid <= tt[-1])
    out[inside] = np.interp(grid[inside], tt, yy)
    return out
