"""Matplotlib-based plotting utilities for property histories."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ..io.paths import SavePaths


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

    def save(self, outname, subdir='history_plots'):
        """Save the figure via :class:`~simbanator.io.paths.SavePaths`.

        Parameters
        ----------
        outname : str
            Filename (e.g. ``'history.png'``).
        subdir : str
            Sub-directory inside ``output/plot/``.
        """
        paths = SavePaths()
        output_dir = paths.create_subdir(paths.get_filetype_path('plot'), subdir)
        output_file = os.path.join(output_dir, outname)
        self.fig.savefig(output_file, bbox_inches='tight')
