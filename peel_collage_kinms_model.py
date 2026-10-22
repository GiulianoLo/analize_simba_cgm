#!/usr/bin/env python
"""Regenerate the KinMS best-fit model cube of an ALMA-C11 source and write its moment maps next to the data's.

Runs in the *kinms-env* conda environment (KinMS + kinms_fitter live there, not in the base env):

    ~/anaconda3/envs/kinms-env/bin/python peel_collage_kinms_model.py [--source hc3]

The fit itself is NOT redone: the best-fit parameters are read from the table `kinms_fitter` wrote
(`Dynamics/kinms_out/<source>_out.fits`, `kinms.ipynb`), the fitter is set up exactly as in that notebook
(the trim_32 cubelet, `spectral_trim=[25, 89]`, the skySampler clouds of the masked cubelet, `nrings` = the number
of V_i rows) and `model_simple(bestvals)` is evaluated once.  Output (`output/peel_collage/<source>_kinms_model.npz`):

    data_mom0, data_mom1, model_mom0, model_mom1   [Jy/beam km/s, km/s]  on the fitter's grid, index [x, y]
    x_as, y_as   the grid offsets [arcsec] from the fitter's reference pixel: x positive to the EAST, y to the North
    ra0, dec0    the sky position of that reference pixel [deg]; xc_as, yc_as the fitted centre offsets [arcsec]
    beam         (bmaj, bmin, bpa) [arcsec, arcsec, deg];  pix_as the cell size;  params / labels the best fit
    vsys, inc, pa the fitted systemic velocity [km/s], inclination and PA [deg, receding side, E of N]
    ring_r_as, ring_v  the tilted-ring radii and velocities (the outer rings that sit at the prior ceiling are flagged)

Moments: the model's from the model cube alone; the data's from the trimmed cube inside a 3σ mask of the
beam-smoothed cube (a plain CASA-like clip), so the two velocity fields can be drawn with the same recipe.
"""
import os, sys, argparse
import numpy as np

HOME = os.path.expanduser("~")
DYN = f"{HOME}/Desktop/Projects/ALMA-C11/Dynamics"
CUBELETS = f"{HOME}/Desktop/Projects/ALMA-C11/Data/cubelets"
CUBE = {"hc3": f"{CUBELETS}/hc3_cube_X12055db_X1140a_X12055db_X11bc3_172434_trim_32_.image.pbcor.fits"}
CLOUDS = {"hc3": f"{CUBELETS}/hc3_cube_X12055db_X1140a_X12055db_X11bc3_172434_trim_32_.image.pbcor.masked.fits_clouds.npz"}
SPECTRAL_TRIM = {"hc3": [25, 89]}
LINEFREE = {"hc3": [90, 120]}

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="hc3")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "peel_collage"))
ap.add_argument("--vceil", type=float, default=0.99, help="rings whose V sits above this fraction of the prior ceiling are flagged unconstrained")
args = ap.parse_args()
src = args.source
os.makedirs(args.out, exist_ok=True)

from astropy.table import Table
from kinms_fitter import kinms_fitter

tab = Table.read(f"{DYN}/kinms_out/{src}_out.fits")
labels = [str(x).strip() for x in tab["Quantity"]]
best = np.array(tab["Bestfit"], float)
nrings = sum(l.startswith("V") and l[1:].isdigit() for l in labels)
print(f"{src}: best fit {dict(zip(labels, np.round(best, 3)))}\n  nrings = {nrings}")

fit = kinms_fitter(CUBE[src], spectral_trim=SPECTRAL_TRIM[src], linefree_chans=LINEFREE[src])
clouds = np.load(CLOUDS[src], allow_pickle=True)["arr_0"]
fit.skySampClouds = clouds
fit.pa_guess = 90 + 180
fit.nrings = nrings
fit.initial_guesses = best.copy()
fit.interactive = False; fit.show_plots = False; fit.pdf = False; fit.silent = True; fit.show_corner = False
fit.text_output = False
import matplotlib; matplotlib.use("Agg")
fit.run(method="simple", justplot=True, savepath=args.out + "/")       # sets the profiles, the KinMS instance, the offsets
par = np.array(fit.initial_guesses, float)                              # Xc / Yc now in arcsec from the reference pixel
model = np.asarray(fit.model_simple(par), float)
data = np.asarray(fit.cube, float)
print(f"  data cube {data.shape}, model cube {model.shape}, cell {fit.cellsize:.3f}\", dv {fit.dv:.2f} km/s, beam {fit.bmaj:.2f} x {fit.bmin:.2f} @ {fit.bpa:.1f}")
print(f"  rms {np.nanmean(fit.rms):.2e}; ring radii {np.round(fit.bincentroids, 2)} arcsec")

v = np.asarray(fit.v1, float)
vprior_max = float(fit.vel_range[1])
iv = [labels.index(f"V{i}") for i in range(nrings)]
ring_v = best[iv]
cloud_rmax = float(np.nanmax(np.hypot(clouds[:, 0], clouds[:, 1])))       # the sky-sampled clouds end here: outer rings carry no data
ring_ok = (ring_v < args.vceil * vprior_max) & (np.asarray(fit.bincentroids, float) <= cloud_rmax + 0.5 * fit.bmaj)
print(f"  ring velocities {np.round(ring_v, 1)} km/s; constrained {ring_ok} (prior ceiling {vprior_max:.0f}, clouds out to {cloud_rmax:.2f}\")")

def moments(cube, mask=None):
    c = np.where(np.isfinite(cube), cube, 0.0)
    if mask is not None:
        c = np.where(mask, c, 0.0)
    m0 = c.sum(axis=2) * fit.dv
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (c * v[None, None, :]).sum(axis=2) / c.sum(axis=2)
    return m0, m1

# data mask: 3 sigma in a cube smoothed by the beam in space + 3 channels in velocity (Hanning-like)
from scipy.ndimage import gaussian_filter
sig_pix = fit.bmaj / 2.3548 / fit.cellsize
sm = gaussian_filter(np.where(np.isfinite(data), data, 0.0), [sig_pix, sig_pix, 1.0])
rms_sm = np.nanstd(sm[:, :, :5]) if sm.shape[2] > 8 else np.nanstd(sm)
mask = sm > 3.0 * rms_sm
d0, d1 = moments(data, mask)
m0, m1 = moments(model, model > 0.02 * np.nanmax(model))
d1[d0 <= 0] = np.nan; m1[m0 <= 0] = np.nan

ra0, dec0 = float(fit._xc_img), float(fit._yc_img)
np.savez(os.path.join(args.out, f"{src}_kinms_model.npz"),
         data_mom0=d0, data_mom1=d1, model_mom0=m0, model_mom1=m1, data_mask_frac=mask.mean(axis=2),
         x_as=np.asarray(fit.x1, float), y_as=np.asarray(fit.y1, float), pix_as=float(fit.cellsize), dv=float(fit.dv),
         ra0=ra0, dec0=dec0, xc_as=float(par[0]), yc_as=float(par[1]),
         beam=np.array([fit.bmaj, fit.bmin, fit.bpa], float), params=best, labels=np.array(labels),
         vsys=float(best[labels.index("Vsys")]), inc=float(best[labels.index("inc")]), pa=float(best[labels.index("PA")]),
         veldisp=float(best[labels.index("veldisp")]), ring_r_as=np.asarray(fit.bincentroids, float), ring_v=ring_v, ring_ok=ring_ok,
         rms=float(np.nanmean(fit.rms)), cube=CUBE[src])
print(f"  written {os.path.join(args.out, f'{src}_kinms_model.npz')}")
print(f"  model mom0 peak {np.nanmax(m0):.3f}, data mom0 peak {np.nanmax(d0):.3f}; model mom1 range {np.nanmin(m1):.0f}..{np.nanmax(m1):.0f}, data {np.nanmin(d1):.0f}..{np.nanmax(d1):.0f}")
