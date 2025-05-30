import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
from .. import SavePaths

class HistoryPlots:
    def __init__(self, x, y, rows=1, cols=1, *args, **kwargs):
        """
        Initialize the HistoryPlots instance.

        :param args: Positional arguments passed to plt.subplots.
        :param kwargs: Keyword arguments passed to plt.subplots.
        """
        # Initialize the figure and axes using plt.subplots
        self.fig, self.axs = plt.subplots(rows, cols, *args, **kwargs)
        # Flatten axes in case of multiple subplots
        if isinstance(self.axs, np.ndarray):
            self.axs = self.axs.flatten()
        else:
            self.axs = [self.axs]
        # the axis with the data to plot
        self.x = x
        self.y = y
        # You can add custom initialization code here
        self.custom_setup()

    def custom_setup(self):
        """
        Custom setup code for the figure and axes.
        """
        pass  # You can customize your figure/axes here if needed

    def plot(self, **kwargs):
        """
        Plot data on all axes.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param kwargs: Additional keyword arguments for plotting.
        """
        for ax in self.axs:
            ax.plot(self.x, self.y, **kwargs)
            ax.set_yscale('log')  # Set y-axis to logarithmic scale

    def z_on_top(self, zlist, cosmo):
        """
        Add a secondary x-axis on top showing redshift values that correspond
        to cosmic age values on the bottom x-axis.

        Parameters
        ----------
        zlist : list or array
            List of redshift values to mark on the secondary axis.
        cosmo : astropy.cosmology object
            Cosmology with an `age(z)` method returning Quantity in Gyr.
        """
        # Convert redshifts to cosmic ages (in Gyr)
        age_ticks = [cosmo.age(z).value for z in zlist]  # in Gyr

        # Loop through all axes in the figure
        for i, ax in enumerate(self.axs):
            # Set main x-axis limits (age in Gyr) — make sure it's increasing
            xmin = min(age_ticks)
            xmax = max(age_ticks)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel('Cosmic Age (Gyr)' if i == len(self.axs) - 1 else '')

            # Create twin x-axis on top
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(age_ticks)
            if i == 0:
                ax2.set_xticklabels([f'{z:g}' for z in zlist])
                ax2.set_xlabel('Redshift')
            else:
                ax2.set_xticklabels([])
                ax2.set_xlabel('')

            ax.minorticks_on()



    def interpolate_plot(self, num_points=100, kind='linear', **kwargs):
        """
        Interpolate and plot the data.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param num_points: Number of points for interpolation.
        :param kind: Interpolation method (e.g., 'linear', 'cubic').
        :param kwargs: Additional keyword arguments for plotting.
        """
        x = self.x
        y = self.y
        x_interpolated = np.linspace(min(x), max(x), num_points)
        # Define the interpolation function
        interp_func = interp1d(x, y, kind=kind)
        # Interpolate the property value at redshift z
        y_interpolated = interp_func(x_interpolated)
        for ax in self.axs:
            ax.plot(x_interpolated, y_interpolated, **kwargs)

    def get_fig(self):
        """
        Return the figure object.
        """
        return self.fig

    def get_axs(self):
        """
        Return the axes objects.
        """
        return self.axs

    def show(self):
        """
        Display the figure.
        """
        plt.show()

    def save(self, outname, subdir='history_plots'):
        """Use SavingPaths to save the plot (a subdirectory can be specified and created)
        """
        # Instantiate SavePaths
        paths = SavePaths()
        # Determine the output directory for HDF5 files
        output_dir = paths.get_filetype_path('plot')
        output_dir = paths.create_subdir(output_dir, subdir)
        output_file = os.path.join(output_dir, outname)
        self.fig.savefig(output_file, bbox_inches='tight')

