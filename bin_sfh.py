import sys
import os
from tqdm.auto import tqdm
import fsps
import yt
import caesar
import numpy as np
import pickle
from multiprocessing import Pool
import argparse  # Import argparse for command-line arguments

import modules as anal
import modules.anal_func as anal_func



import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib

matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": 'STIXGeneral',
    "mathtext.fontset" : "cm"
})

binwidth = 3 #Myr

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SFH from snapshots and Caesar files.")
    parser.add_argument("--file", type=str, required=True, help="file with the produced sfh")
    return parser.parse_args()


#this function is given to scipy.stats.binned_statistics and acts on the particle masses per bin to give total(Msun) / timebin
def get_massform(massform):
        return np.sum(massform) / (binwidth * 1e6)

def get_galaxy_SFH(file_):
    dat = pd.read_pickle(file_)
    ids = dat['id']
    for galaxy_id in ids:
        # Extract data for this galaxy
        idx = np.where(np.asarray(dat['id']) == galaxy_id)[0][0]
        massform = np.array(dat['massform'][idx])
        tform = np.array(dat['tform'][idx]) * 1000  # Convert from Gyr to Myr
        t_H = np.max(tform)

        # Define bins
        bins = np.arange(100, t_H, binwidth)

        # Compute the binned statistic (SFR)
        sfrs, bins, binnumber = scipy.stats.binned_statistic(
            tform, massform, statistic='sum', bins=bins
        )
        sfrs[np.isnan(sfrs)] = 0
        bincenters = 0.5 * (bins[:-1] + bins[1:])

        # Save bincenters and SFH to a file
        output_file = os.path.join('/mnt/home/glorenzon/simbanator/', f"galaxy_{galaxy_id}_sfh.txt")
        np.savetxt(output_file, np.column_stack([bincenters, sfrs]), 
                   header="BinCenters(Myr) SFR(Msun/Myr)", fmt="%.6e")

        print(f"Saved SFH for galaxy {galaxy_id} to {output_file}")

if __name__ == "__main__":
    
    args = parse_arguments()
    
    time, sfh = get_galaxy_SFH(args.file)
    
    
    #plt.plot(time, sfh)
    #plt.ylabel('SFR [M$_{\odot}/$yr]')
    #plt.xlabel('t$_H$ [Myr]')