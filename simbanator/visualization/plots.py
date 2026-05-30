"""Matplotlib-based plotting utilities for property histories."""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class HistoryPlots:
    """Multi-panel figure for plotting galaxy property histories.

    Parameters
    ----------
    x : array-like
        X-axis data (e.g. cosmic age).
    y : array-like
        Y-axis data (e.g. stellar mass).
    rows, cols : int
        Subplot grid dimensions.
    *args, **kwargs
        Forwarded to :func:`matplotlib.pyplot.subplots`.
    """

    def __init__(self, x, y, rows=1, cols=1, *args, **kwargs):
        self.fig, self.axs = plt.subplots(rows, cols, *args, **kwargs)
        if isinstance(self.axs, np.ndarray):
            self.axs = self.axs.flatten()
        else:
            self.axs = [self.axs]
        self.x = x
        self.y = y
        self.custom_setup()

    def custom_setup(self):
        """Override for custom figure/axes configuration."""
        pass

    def plot(self, **kwargs):
        """Plot data on all axes."""
        for ax in self.axs:
            ax.plot(self.x, self.y, **kwargs)

    def z_on_top(self, zlist, cosmo):
        """Add a secondary x-axis showing redshift labels.

        Parameters
        ----------
        zlist : list of float
            Redshift tick values.
        cosmo : astropy cosmology
            Cosmology with an ``age`` method.
        """
        for i, ax in enumerate(self.axs):
            zticks = zlist
            ax2 = ax.twiny()
            ax2.set_xticks(zticks)
            ax2.set_xticklabels([f'{z:g}' for z in zlist])

            zmin, zmax = min(zlist), max(zlist)
            ax.set_xlim(cosmo.age(zmin).value, cosmo.age(zmax).value)
            ax2.set_xlim(zmin, zmax)
            ax.minorticks_on()

            if i == len(self.axs) - 1:
                ax.set_xlabel('Age (Gyr)')
            if i == 0:
                ax2.set_xlabel('Redshift')
            else:
                ax2.set_xticklabels([])

            ax.axvline(cosmo.age(2).value, ls='--', alpha=0.4)

    def interpolate_plot(self, num_points=100, kind='linear', **kwargs):
        """Plot interpolated data.

        Parameters
        ----------
        num_points : int
            Number of interpolation samples.
        kind : str
            Interpolation method for ``scipy.interpolate.interp1d``.
        """
        x_interp = np.linspace(min(self.x), max(self.x), num_points)
        interp_func = interp1d(self.x, self.y, kind=kind)
        y_interp = interp_func(x_interp)
        for ax in self.axs:
            ax.plot(x_interp, y_interp, **kwargs)

    def get_fig(self):
        """Return the figure object."""
        return self.fig

    def get_axs(self):
        """Return the axes objects."""
        return self.axs

    def show(self):
        """Display the figure."""
        plt.show()

    def save(self, outname, output_dir=None, sim_name='default'):
        """Save the figure to disk.

        Parameters
        ----------
        outname : str
            Filename (e.g. ``'history.png'``).
        output_dir : str, optional
            Directory to save in.  Defaults to ``./output/<sim_name>/plots/``.
        sim_name : str, optional
            Simulation name used when *output_dir* is not provided.
        """
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, outname)
        self.fig.savefig(output_file, bbox_inches='tight')


# ── Merger-rate-by-phase plot ────────────────────────────────────────────────


def _find_first_crossing_below(tarr, values, threshold):
    """Return the linearly-interpolated time when *values* first drops below *threshold*.

    Returns ``np.nan`` if the crossing never occurs.
    """
    for i in range(len(values) - 1):
        v1 = float(values[i])
        v2 = float(values[i + 1])
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        if v1 >= threshold > v2:
            frac = (v1 - threshold) / (v1 - v2)
            return float(tarr[i]) + frac * (float(tarr[i + 1]) - float(tarr[i]))
    return np.nan


def plot_merger_rate_by_phase(
    tarr,
    sfr,
    ssfr,
    h2_frac,
    major_mergers,
    minor_mergers,
    event_idx=0,
    h2_threshold=1e-4,
    quenching_kwargs=None,
    figsize=(12, 8),
    title=None,
    save_path=None,
):
    """Bar plot of major/minor merger rates during star-forming, quenching,
    and post-quenching phases for a single galaxy.

    Phases are defined by ``find_quenching_times``:

    * **Star forming** — before the start-of-quenching time (SFT, sSFR crosses
      below 1/t).
    * **Quenching** — from SFT to the quenching time (QT, sSFR crosses below
      0.2/t).
    * **Post-quenching** — after QT.

    A companion sSFR panel marks the phase boundaries and two additional
    interest points:

    * Peak of the star formation history.
    * First time H₂/M★ falls below *h2_threshold* (galaxy cannot rejuvenate).

    Parameters
    ----------
    tarr : array-like
        Cosmic time in **years**, shape ``(N_snaps,)``.  Need not be sorted.
    sfr : array-like
        Star formation rate in M☉/yr, shape ``(N_snaps,)``.
    ssfr : array-like
        Specific SFR in yr⁻¹, shape ``(N_snaps,)``.
    h2_frac : array-like
        H₂ / M★ mass fraction, shape ``(N_snaps,)``.
    major_mergers : array-like
        Major merger companion count at each snapshot, shape ``(N_snaps,)``.
    minor_mergers : array-like
        Minor merger companion count at each snapshot, shape ``(N_snaps,)``.
    event_idx : int
        Index of the quenching event to use when multiple are detected
        (0 = first in cosmic time).
    h2_threshold : float
        H₂ fraction threshold for the no-rejuvenation marker (default ``1e-4``).
    quenching_kwargs : dict, optional
        Extra keyword arguments forwarded to ``find_quenching_times``
        (e.g. ``smooth_window``, ``tolerance``).  ``plot`` and
        ``save_fits_path`` are forced to ``False`` / ``None``.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Figure title.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_bar : matplotlib.axes.Axes
        Top panel — grouped bar chart.
    ax_ssfr : matplotlib.axes.Axes
        Bottom panel — sSFR history with phase shading.
    """
    from ..analysis.quenching import find_quenching_times

    tarr          = np.asarray(tarr,          dtype=float)
    sfr           = np.asarray(sfr,           dtype=float)
    ssfr          = np.asarray(ssfr,          dtype=float)
    h2_frac       = np.asarray(h2_frac,       dtype=float)
    major_mergers = np.asarray(major_mergers, dtype=float)
    minor_mergers = np.asarray(minor_mergers, dtype=float)

    # Sort everything by ascending cosmic time
    sort_idx      = np.argsort(tarr)
    tarr          = tarr[sort_idx]
    sfr           = sfr[sort_idx]
    ssfr          = ssfr[sort_idx]
    h2_frac       = h2_frac[sort_idx]
    major_mergers = major_mergers[sort_idx]
    minor_mergers = minor_mergers[sort_idx]

    # ── Quenching times ──────────────────────────────────────────────────
    kw = dict(quenching_kwargs or {})
    kw['plot'] = False
    kw['save_fits_path'] = None

    qt_list, sft_list, _ = find_quenching_times(tarr, ssfr, **kw)

    if len(qt_list) == 0:
        warnings.warn("No quenching event detected; cannot define phases.")
        return None, None, None

    if event_idx >= len(qt_list):
        warnings.warn(
            f"event_idx={event_idx} out of range ({len(qt_list)} event(s)); "
            "using the last event."
        )
        event_idx = len(qt_list) - 1

    t_sft = sft_list[event_idx]   # yr
    t_qt  = qt_list[event_idx]    # yr

    # ── Peak SFR ─────────────────────────────────────────────────────────
    valid_sfr = np.isfinite(sfr)
    if valid_sfr.any():
        t_peak_sfr = tarr[int(np.nanargmax(np.where(valid_sfr, sfr, np.nan)))]
    else:
        t_peak_sfr = np.nan

    # ── H2 depletion time ────────────────────────────────────────────────
    t_h2_deplete = _find_first_crossing_below(tarr, h2_frac, h2_threshold)

    # ── Phase masks and durations ────────────────────────────────────────
    sf_mask  = tarr <= t_sft
    q_mask   = (tarr > t_sft) & (tarr <= t_qt)
    pq_mask  = tarr > t_qt

    sf_dur_gyr  = (t_sft    - tarr[0])   / 1e9
    q_dur_gyr   = (t_qt     - t_sft)     / 1e9
    pq_dur_gyr  = (tarr[-1] - t_qt)      / 1e9

    def _rate(arr, mask, dur_gyr):
        n = int(np.nansum(arr[mask]))
        r = n / dur_gyr if dur_gyr > 0 else 0.0
        return n, r

    sf_maj_n, sf_maj_r = _rate(major_mergers, sf_mask,  sf_dur_gyr)
    sf_min_n, sf_min_r = _rate(minor_mergers, sf_mask,  sf_dur_gyr)
    q_maj_n,  q_maj_r  = _rate(major_mergers, q_mask,   q_dur_gyr)
    q_min_n,  q_min_r  = _rate(minor_mergers, q_mask,   q_dur_gyr)
    pq_maj_n, pq_maj_r = _rate(major_mergers, pq_mask,  pq_dur_gyr)
    pq_min_n, pq_min_r = _rate(minor_mergers, pq_mask,  pq_dur_gyr)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, (ax_bar, ax_ssfr) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [2, 1.2]},
    )
    fig.subplots_adjust(hspace=0.5)

    # Grouped bar chart
    phases = [
        f'Star Forming\n({sf_dur_gyr:.2f} Gyr)',
        f'Quenching\n({q_dur_gyr:.2f} Gyr)',
        f'Post-Quenching\n({pq_dur_gyr:.2f} Gyr)',
    ]
    maj_rates = [sf_maj_r, q_maj_r, pq_maj_r]
    min_rates = [sf_min_r, q_min_r, pq_min_r]
    maj_ns    = [sf_maj_n, q_maj_n, pq_maj_n]
    min_ns    = [sf_min_n, q_min_n, pq_min_n]

    x = np.arange(len(phases))
    w = 0.35
    bars_maj = ax_bar.bar(x - w / 2, maj_rates, w, label='Major mergers',
                          color='steelblue', zorder=3)
    bars_min = ax_bar.bar(x + w / 2, min_rates, w, label='Minor mergers',
                          color='coral', zorder=3)
    ax_bar.set_axisbelow(True)
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Count labels on top of bars
    y_scale = max(max(maj_rates), max(min_rates), 1e-9)
    for bar, n in zip(bars_maj, maj_ns):
        if n > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_scale * 0.02,
                str(n), ha='center', va='bottom', fontsize=9, color='steelblue',
            )
    for bar, n in zip(bars_min, min_ns):
        if n > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_scale * 0.02,
                str(n), ha='center', va='bottom', fontsize=9, color='coral',
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(phases, fontsize=10)
    ax_bar.set_ylabel(r'Merger rate [Gyr$^{-1}$]', fontsize=11)
    ax_bar.set_title(title or 'Merger rate by galaxy phase', fontsize=12)
    ax_bar.legend(fontsize=10)
    ax_bar.set_xlim(-0.6, len(phases) - 0.4)

    # sSFR history panel
    t_gyr = tarr / 1e9

    valid = np.isfinite(ssfr) & (ssfr > 0)
    ax_ssfr.semilogy(t_gyr[valid], ssfr[valid], 'k-', lw=1.5,
                     label='sSFR', zorder=4)
    ax_ssfr.semilogy(t_gyr, 1.0 / tarr, 'b--', lw=1.0, alpha=0.7, label='1/t')
    ax_ssfr.semilogy(t_gyr, 0.2 / tarr, 'g:',  lw=1.0, alpha=0.7, label='0.2/t')

    # Phase shading
    ax_ssfr.axvspan(t_gyr[0],  t_sft / 1e9, alpha=0.10, color='green',
                    label='Star forming')
    ax_ssfr.axvspan(t_sft / 1e9, t_qt / 1e9, alpha=0.15, color='orange',
                    label='Quenching')
    ax_ssfr.axvspan(t_qt / 1e9,  t_gyr[-1],  alpha=0.10, color='red',
                    label='Post-quenching')

    # Interest points
    if np.isfinite(t_peak_sfr):
        ax_ssfr.axvline(
            t_peak_sfr / 1e9, ls='--', color='purple', lw=1.5,
            label=f'Peak SFR ({t_peak_sfr / 1e9:.2f} Gyr)',
        )
    if np.isfinite(t_h2_deplete):
        ax_ssfr.axvline(
            t_h2_deplete / 1e9, ls='-.', color='saddlebrown', lw=1.5,
            label=f'H₂ depleted ({t_h2_deplete / 1e9:.2f} Gyr)',
        )

    ax_ssfr.set_xlabel('Cosmic time [Gyr]', fontsize=11)
    ax_ssfr.set_ylabel(r'sSFR [yr$^{-1}$]', fontsize=11)
    ax_ssfr.legend(fontsize=8, loc='upper right', ncol=2)

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax_bar, ax_ssfr


# ── Galaxy track and neighborhood diagnostic plots ───────────────────────────


def _unwrap_pos_1d(raw_pos, box_size):
    """Unwrap a (n_snap, 3) position track across periodic boundaries.

    NaN entries are preserved; offsets do not propagate through gaps.
    """
    out = raw_pos.copy()
    for i in range(1, len(out)):
        if np.any(np.isnan(out[i])) or np.any(np.isnan(out[i - 1])):
            continue
        d = out[i] - out[i - 1]
        d[d >  0.5 * box_size] -= box_size
        d[d < -0.5 * box_size] += box_size
        out[i] = out[i - 1] + d
    return out


def plot_main_galaxy_track(
    galaxy,
    caesar_paths,
    redshifts,
    box_size,
    *,
    projection=('x', 'y'),
    figsize=(8, 8),
    cmap='plasma_r',
    title=None,
    save_path=None,
):
    """Plot the main galaxy position track as a line colored by redshift.

    Mirrors the sfh_caesar line-plot approach (HDF5BuildHistory) but in
    projected position space rather than property-vs-time space.
    Periodic wrapping is removed via minimum-image unwrapping so the
    trajectory is continuous.

    Parameters
    ----------
    galaxy : Galaxy
        Galaxy object with ``.track`` set — one catalog index per snapshot,
        aligned with *caesar_paths*.
    caesar_paths : list of str
        Caesar HDF5 catalog paths, one per snapshot.
    redshifts : array-like
        Redshift at each snapshot, same order as *caesar_paths*.
    box_size : float
        Periodic box side length in position units (Mpc/h).
    projection : tuple of str
        Two of ``'x'``, ``'y'``, ``'z'`` to project onto.  Default ``('x', 'y')``.
    figsize : tuple
        Figure size in inches.
    cmap : str
        Colormap name for redshift coloring.
    title : str, optional
        Figure title.
    save_path : str, optional
        If given, save the figure here.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    _POS = 'galaxy_data/pos'
    _AX  = {'x': 0, 'y': 1, 'z': 2}
    i0, i1 = _AX[projection[0]], _AX[projection[1]]

    redshifts = np.asarray(redshifts, dtype=float)
    track = np.asarray(galaxy.track, dtype=np.int64)
    n_snaps = len(caesar_paths)

    raw_pos = np.full((n_snaps, 3), np.nan)
    for si, cpath in enumerate(caesar_paths):
        idx = int(track[si])
        if idx < 0:
            continue
        with h5py.File(cpath, 'r') as f:
            if idx < f[_POS].shape[0]:
                raw_pos[si] = f[_POS][idx]

    pos = _unwrap_pos_1d(raw_pos, box_size)
    valid = ~np.isnan(pos[:, i0])
    z_valid = redshifts[valid]

    norm = mcolors.Normalize(vmin=z_valid.min(), vmax=z_valid.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pos[valid, i0], pos[valid, i1], '-', color='k', lw=0.8, alpha=0.35, zorder=2)
    sc = ax.scatter(
        pos[valid, i0], pos[valid, i1],
        c=z_valid, cmap=cmap, norm=norm,
        s=60, marker='*', edgecolors='k', lw=0.4, zorder=4,
        label='Main galaxy',
    )
    fig.colorbar(sc, ax=ax, label='Redshift')
    ax.set_xlabel(f'{projection[0]} [Mpc/h]', fontsize=12)
    ax.set_ylabel(f'{projection[1]} [Mpc/h]', fontsize=12)
    ax.set_title(title or f'Main galaxy track ({projection[0]}-{projection[1]})', fontsize=13)
    ax.legend(fontsize=10)

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax


def plot_neighborhood_track(
    galaxy,
    caesar_paths,
    redshifts,
    box_size,
    *,
    radius_mpc=0.5,
    projection=('x', 'y'),
    mass_threshold=1e9,
    mass_threshold_maj=0.25,
    mass_threshold_min=0.10,
    figsize=(10, 10),
    cmap='plasma_r',
    title=None,
    save_path=None,
):
    """Plot the main galaxy track, all neighbors within *radius_mpc*, and merger events.

    For every snapshot, every galaxy within *radius_mpc* of the tracked
    galaxy (minimum-image convention) is shown as a small dot colored by
    redshift.  Companions stored in ``galaxy._progenitors`` are additionally
    highlighted: red pentagons for major mergers (mass ratio ≥
    *mass_threshold_maj*), orange diamonds for minor.  All positions are
    rendered in the unwrapped trajectory frame so the main galaxy's path is
    continuous.

    Parameters
    ----------
    galaxy : Galaxy
        Output of ``process_galaxies_with_tracks``.  Must have ``.track``
        and ``._progenitors`` populated.
    caesar_paths : list of str
        Caesar HDF5 catalog paths, one per snapshot.
    redshifts : array-like
        Redshift at each snapshot, same order as *caesar_paths*.
    box_size : float
        Periodic box side length in position units (Mpc/h).
    radius_mpc : float
        Neighbor search radius in Mpc/h.  Default 0.5 ≈ 500 kpc/h.
    projection : tuple of str
        Two of ``'x'``, ``'y'``, ``'z'`` to project onto.  Default ``('x', 'y')``.
    mass_threshold : float
        Minimum stellar mass (M☉) for neighbor galaxies shown in the scatter.
        Default 1e9 to suppress noise from tiny objects.
    mass_threshold_maj : float
        Mass ratio threshold for major merger highlights (default 0.25).
    mass_threshold_min : float
        Minimum mass ratio for minor merger highlights (default 0.10).
    figsize : tuple
        Figure size in inches.
    cmap : str
        Colormap name for redshift coloring.
    title : str, optional
        Figure title.
    save_path : str, optional
        If given, save the figure here.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    _POS   = 'galaxy_data/pos'
    _SMASS = 'galaxy_data/dicts/masses.stellar'
    _AX    = {'x': 0, 'y': 1, 'z': 2}
    i0, i1 = _AX[projection[0]], _AX[projection[1]]

    redshifts = np.asarray(redshifts, dtype=float)
    track = np.asarray(galaxy.track, dtype=np.int64)
    n_snaps = len(caesar_paths)

    # ── 1. Read and unwrap main galaxy track ─────────────────────────────────
    raw_pos = np.full((n_snaps, 3), np.nan)
    for si, cpath in enumerate(caesar_paths):
        idx = int(track[si])
        if idx < 0:
            continue
        with h5py.File(cpath, 'r') as f:
            if idx < f[_POS].shape[0]:
                raw_pos[si] = f[_POS][idx]

    pos_uw = _unwrap_pos_1d(raw_pos, box_size)

    # ── 2. Collect neighbor positions in unwrapped frame ─────────────────────
    neigh_pos  = []
    neigh_zval = []

    for si, cpath in enumerate(caesar_paths):
        main_idx = int(track[si])
        if main_idx < 0 or np.isnan(raw_pos[si, 0]):
            continue
        with h5py.File(cpath, 'r') as f:
            all_pos   = f[_POS][:]
            all_smass = f[_SMASS][:]

        dx = all_pos[:, 0] - raw_pos[si, 0]
        dy = all_pos[:, 1] - raw_pos[si, 1]
        dz = all_pos[:, 2] - raw_pos[si, 2]
        dx -= box_size * np.round(dx / box_size)
        dy -= box_size * np.round(dy / box_size)
        dz -= box_size * np.round(dz / box_size)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        mask = (
            (np.arange(len(dx)) != main_idx)
            & (dist < radius_mpc)
            & (all_smass >= mass_threshold)
        )
        for i in np.where(mask)[0]:
            neigh_pos.append(pos_uw[si] + np.array([dx[i], dy[i], dz[i]]))
            neigh_zval.append(redshifts[si])

    neigh_pos  = np.array(neigh_pos)  if neigh_pos  else np.zeros((0, 3))
    neigh_zval = np.array(neigh_zval)

    # ── 3. Collect merger events from progenitor record ──────────────────────
    maj_pos, maj_zval = [], []
    min_pos, min_zval = [], []

    for prog in galaxy._progenitors.values():
        if prog.merger != 1 or prog.fragmentation != 0:
            continue
        si = int(prog.snapshot)
        if si >= n_snaps or np.isnan(raw_pos[si, 0]):
            continue
        dx_p = prog.x - raw_pos[si, 0]
        dy_p = prog.y - raw_pos[si, 1]
        dz_p = prog.z - raw_pos[si, 2]
        dx_p -= box_size * np.round(dx_p / box_size)
        dy_p -= box_size * np.round(dy_p / box_size)
        dz_p -= box_size * np.round(dz_p / box_size)
        pt = pos_uw[si] + np.array([dx_p, dy_p, dz_p])

        if prog.mass >= mass_threshold_maj:
            maj_pos.append(pt);  maj_zval.append(redshifts[si])
        elif prog.mass >= mass_threshold_min:
            min_pos.append(pt);  min_zval.append(redshifts[si])

    maj_pos  = np.array(maj_pos)  if maj_pos  else np.zeros((0, 3))
    min_pos  = np.array(min_pos)  if min_pos  else np.zeros((0, 3))
    maj_zval = np.array(maj_zval)
    min_zval = np.array(min_zval)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    valid   = ~np.isnan(pos_uw[:, i0])
    z_valid = redshifts[valid]
    norm    = mcolors.Normalize(vmin=redshifts.min(), vmax=redshifts.max())

    fig, ax = plt.subplots(figsize=figsize)

    if neigh_pos.shape[0]:
        ax.scatter(
            neigh_pos[:, i0], neigh_pos[:, i1],
            c=neigh_zval, cmap=cmap, norm=norm,
            s=15, alpha=0.5, marker='o', lw=0, zorder=3,
            label=f'Neighbors (r < {radius_mpc} Mpc/h)',
        )

    ax.plot(pos_uw[valid, i0], pos_uw[valid, i1], '-', color='k', lw=0.8, alpha=0.35, zorder=4)
    sc = ax.scatter(
        pos_uw[valid, i0], pos_uw[valid, i1],
        c=z_valid, cmap=cmap, norm=norm,
        s=80, marker='*', edgecolors='k', lw=0.5, zorder=5,
        label='Main galaxy',
    )

    if maj_pos.shape[0]:
        ax.scatter(
            maj_pos[:, i0], maj_pos[:, i1],
            c='red', s=120, marker='p', edgecolors='darkred', lw=0.8,
            zorder=6, label='Major merger companion',
        )
    if min_pos.shape[0]:
        ax.scatter(
            min_pos[:, i0], min_pos[:, i1],
            c='orange', s=80, marker='D', edgecolors='saddlebrown', lw=0.8,
            zorder=6, label='Minor merger companion',
        )

    fig.colorbar(sc, ax=ax, label='Redshift')
    ax.set_xlabel(f'{projection[0]} [Mpc/h]', fontsize=12)
    ax.set_ylabel(f'{projection[1]} [Mpc/h]', fontsize=12)
    ax.set_title(
        title or (
            f'Galaxy neighborhood (r < {radius_mpc} Mpc/h) + merger events '
            f'({projection[0]}-{projection[1]})'
        ),
        fontsize=12,
    )
    ax.legend(fontsize=9, loc='upper right')

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig, ax
