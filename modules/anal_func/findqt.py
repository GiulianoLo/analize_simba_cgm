import numpy as np
import os
from .. import SavePaths as save
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u



def findQT(tarr, ssfr, plot = False):
    tau = np.log10(0.2/(tarr))
    diff = ssfr - tau

    intersectionD = np.array([])
    intersectionU = np.array([])
    for i in range(len(ssfr)-1):
        if diff[i]*diff[i+1] < 0:
            if diff[i] > diff[i+1]:
                intersectionD = np.append(intersectionD, [tarr[i]])
            else:
                intersectionU = np.append(intersectionU, [tarr[i]])

    intersectionU = np.append(intersectionU, [tarr[-1]])
    for i in range(len(intersectionD)):
        if intersectionU[i] - intersectionD[i] > 0.2*intersectionD[i] or intersectionU[i] == tarr[-1]:
            QT = intersectionD[i]
            break


    tau = np.log10(1/(tarr))
    diff = ssfr - tau
    for i in range(len(ssfr) - 1):
        if tarr[i]<QT and diff[i] * diff[i + 1] < 0:
            SFRT = tarr[i]

    return QT, SFRT


def quenching_info(sfh, z, znow=None, Npoints=1000, ID=None, output_file=None, plot=False, verbose=0):
    sv = save.SavePaths()
    def process_single_galaxy(sfh, z, galaxy_id):
        sfh_gal = np.array(sfh[:, galaxy_id])
        if len(z) < 3:
            if verbose:
                print(f"Galaxy {galaxy_id}: Not enough snapshots.")
            return np.nan, np.nan, np.nan

        z_dense = np.linspace(z.min(), z.max(), Npoints)
        t_dense = cosmo.age(z_dense).to(u.yr).value


        t_sfh = cosmo.age(z).to(u.yr).value
        print(t_sfh[-10:-1])
        print(sfh_gal[-10:-1])
        sfh_interp = InterpolatedUnivariateSpline(t_sfh, sfh_gal, k=3)
        sfh_dense = sfh_interp(t_dense)[::-1]
        t_dense = t_dense[::-1]

        qm, sfe = findQT(t_dense, np.log10(sfh_dense), plot = False)
        tsq = cosmo.age(znow).to(u.yr).value - qm



        if plot:
            print(f"Plotting for galaxy ID: {galaxy_id}")
            # Save each plot with a unique filename based on galaxy_id
            svdir = sv.get_filetype_path('plot')
            svsub = sv.create_subdir(svdir, 'quenching_info')
            svfile = os.path.join(svsub, f'SFH_{galaxy_id}.png')  # Unique filename
            plt.figure(figsize=(18,15))
            plt.plot(t_dense/1e9, sfh_dense, label='SFH Interpolated')
            plt.plot(t_dense/1e9, 1/t_dense, 'r--', label='1/t')
            plt.plot(t_dense/1e9, 0.2/t_dense, 'k--', label='0.2/t')
            plt.axvline(sfe/1e9, color='gray', linestyle=':', label='t_s (start)')
            plt.axvline(qm/1e9, color='black', linestyle=':', label='t_q (quench)')
            if znow!=None:
                plt.axvline(cosmo.age(znow).to(u.yr).value/1e9, color='g', linestyle=':', label=f'z={znow}')
            plt.xlabel('Cosmic Age [Gyr]')
            plt.ylabel('SFR (or sSFR)')
            #plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.title(f"Galaxy ID: {galaxy_id}")
            plt.tight_layout()
            plt.savefig(svfile)
            plt.close()  # Ensure the plot is closed after saving
            print(f'Plot saved to {svfile}')
                
        return qm, sfe, tsq


    # If ID is iterable (list/array), run loop and save to file
    if hasattr(ID, "__iter__") and not isinstance(ID, str):
        quench_moments = []
        quench_times = []
        time_since_quenches = []

        if output_file is None:
            raise ValueError("output_file must be specified when ID is iterable")

        svdir  = sv.get_filetype_path('txt')
        svsub = sv.create_subdir(svdir, 'quenching_info')
        svfile = os.path.join(svsub, output_file)
        with open(svfile, "w") as f:
            f.write("# ID quench_moment(yr) quenchtime(yr) time_since_quench(yr)\n")
            print(ID)
            for galaxy_id in ID:
                try:
                    qm, sfe, tsq = process_single_galaxy(sfh, z, galaxy_id)
                except Exception as e:
                    if verbose:
                        print(f"Error for galaxy {galaxy_id}: {e}")
                    qm, sfe, tsq = np.nan, np.nan, np.nan
                qt = qm-sfe
                quench_moments.append(qm)
                quench_times.append(qt)
                time_since_quenches.append(tsq)

                f.write(f"{galaxy_id} {qm:.6e} {qt:.6e} {tsq:.6e}\n")

        if verbose:
            print(f"Results saved to {svfile}")

        return (np.array(quench_moments), np.array(quench_times), np.array(time_since_quenches))



    # If ID is single int, just process and return
    elif isinstance(ID, int):
        return process_single_galaxy(sfh, z, ID)

    else:
        raise ValueError("ID must be either an int or an iterable of ints")