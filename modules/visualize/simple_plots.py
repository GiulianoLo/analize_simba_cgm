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

    def z_on_top(self, zlist, cosmo):
        """
        Add a secondary x-axis with redshift values and corresponding ages.
        Secondary x-axis labels are displayed only on the top subplot,
        and primary x-axis labels are displayed only on the bottom subplot.
        Ticks are shown on all subplots.
    
        :param zlist: List of redshift values.
        :param cosmo: Cosmology object with an `age` method.
        """
        # Iterate through all axes to configure the secondary x-axis
        for i, ax in enumerate(self.axs):
            # Calculate the age ticks based on redshift values
            zticks = zlist
            # Create a secondary x-axis
            ax2 = ax.twiny()
            ax2.set_xticks(zticks)
            ax2.set_xticklabels(['{:g}'.format(age) for age in zlist])
            
            # Set x-limits for both primary and secondary axes
            zmin, zmax = min(zlist), max(zlist)
            ax.set_xlim(cosmo.age(zmin).value, cosmo.age(zmax).value)
            ax2.set_xlim(zmin, zmax)
            
            # Ensure minor ticks are on
            ax.minorticks_on()
            
            # Configure which axes will display labels
            if i == len(self.axs) - 1:  # Only on the bottom subplot
                ax.set_xlabel('Age (Gyr)')
            else:
                ax.set_xlabel('')  # Hide xlabel on non-bottom subplots
            
            if i == len(self.axs) - 1:  # Only on the bottom subplot
                ax2.set_xlabel('Redshift')  # Label for secondary x-axis
            
            if i == 0:  # Only on the top subplot
                ax2.set_xlabel('Redshift')  # Label for secondary x-axis
            
            # Optionally: Clear secondary x-axis labels from non-top subplots
            if i != 0:
                ax2.set_xticklabels([])

            ax.axvline(cosmo.age(2).value)
                
            #Title adjustment (optional)
            ax.set_title('Redshift')

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

