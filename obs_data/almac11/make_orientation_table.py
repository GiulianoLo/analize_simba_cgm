#!/usr/bin/env python3
"""Build almac11_orientation.csv + almac11_stamps_f814w.npz — the sky orientation (position angle, axis ratio) of every
ALMA-C11 source and a small F814W postage stamp of each, for agn_states_dusty_vs_pure.ipynb Part 27 (the simulated
companions rendered at the observed PA and inclination).  Runs LOCALLY (the inputs live outside the repo); the two
products are copied to the cluster's obs_data/almac11/ by hand, like the other tables here.

Inputs (all local, nothing derived elsewhere):
  ~/Desktop/Proposals/JWST/Cycle 5/Data/cosmos_web_alma_c11_radii.fits   COSMOS-Web 2025 NIRCam Sersic fits (SE++):
        axratio_sersic, angle_sersic (nearest match per 'ALMA source name'); 13/17 sources (not hc5/hc9/hc10/hc12)
  ~/Desktop/Projects/ALMA-C11/ApJL GIANTS catalogues/GIANTS_ACS_CSOSMOS_sizes.csv   ACS F814W gim2d: Ellipticity (all 17)
  ~/Desktop/Projects/ALMA-C11/uvmcmc_shape.csv                            the uv-plane Gaussian fits: q, pa_deg per mode
  ~/Desktop/Projects/ALMA-C11/Dynamics/BB_hc5/rings_final1.txt            3D-Barolo (hc5 only: INC fixed 75, PA fitted)
  ~/Desktop/Proposals/JWST/Cycle 5/Data/cutouts_centers/<id>_RA*_DEC*_f814w.fits   60" ACS F814W cutouts, 0.05"/pix, N up E left
  ~/Desktop/Projects/ALMA-C11/Data/<id>/cont/*.image.pbcor.fits                  the imaging pipeline's (Imaging_v1) continuum
  ~/Desktop/Projects/ALMA-C11/Data/<id>/line/*.line.moments.integrated.fits      ... and CO(3-2) moment-0 maps (0.14"/pix, ~1.3" beam)
  obs_data/almac11/almac11_gas_dust.csv                                          S_dust_ujy / logMdust_fid, S_co_jykms / MH2_fid: the
                                                                                  mass per unit flux that turns a map level into a Sigma

The npz also carries, per source, the ALMA continuum (`cont`, Jy/beam) and CO moment-0 (`mom0`, Jy/beam km/s) cut around the SAME
centre as the F814W stamp (ALMA_HALF_AS half-size, `alma_pix_arcsec`), the rms of each map (MAD of the pixels 3-8" from the centre)
and the beams; the csv gets `ra_deg` / `dec_deg` of the stamp centre, `cont_rms_jyb`, `mom0_rms_jybkms`, the beams, and
`msun_per_jykms` = M_H2(fid) / S_CO, `msun_per_ujy_dust` = M_dust(fid) / S_dust (NaN when the dust is undetected).

Conventions.  pa_*_deg is the sky position angle of the MAJOR axis, degrees East of North, in [0, 180).  The COSMOS-Web
`angle_sersic` is already that angle (checked here: it agrees with the CO-fit PA for hc3 96/94, hc7 39/46, hc4 66/50, hc11
132/116 and with the second moments of the F814W stamps, `pa_stamp_deg`, to a few degrees), so pa_opt_deg = angle_sersic
mod 180; where COSMOS-Web has no fit the stamp moments (r < 1.5") stand in (pa_opt_src = 'acs_stamp').  q_opt is the
deconvolved Sersic axis ratio (COSMOS-Web; ACS 1 - Ellipticity where missing) - the stamp q is PSF-blurred and only a
cross-check.  The uv PA follows the same E-of-N convention (uvmcmc's astrometry: 'PA measured E of N').  The Barolo PA is
the receding side (0-360); pa_kin_deg keeps it, the notebook takes it mod 180.
"""
import os, re, glob
import numpy as np, pandas as pd
from astropy.io import fits

HOME = os.path.expanduser("~")
C5 = f"{HOME}/Desktop/Proposals/JWST/Cycle 5/Data"
AL = f"{HOME}/Desktop/Projects/ALMA-C11"
OUT = os.path.dirname(os.path.abspath(__file__))
PIX = 0.05                      # arcsec per pixel of the ACS cutouts
STAMP_HALF_AS = 4.0             # half-size of the stored stamps [arcsec] -> 160 x 160 pixels (4" = 22 kpc at z = 0.4)
R_MOM_AS = 1.5                  # the aperture radius of the stamp second moments
ALMA_HALF_AS = 4.0              # half-size of the stored ALMA cutouts [arcsec]
ALMA_RMS_ANNULUS = (3.0, 8.0)   # [arcsec] the pixels the map rms is measured on (around the source, inside the primary beam)

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as AU
ids = pd.read_csv(f"{OUT}/age_sersic_sigma.csv")["id"].tolist()
gd = pd.read_csv(f"{OUT}/almac11_gas_dust.csv").set_index("id")
z_by = pd.read_csv(f"{OUT}/age_sersic_sigma.csv").set_index("id")["z"]

# ── COSMOS-Web nearest match per source ──
t = fits.open(f"{C5}/cosmos_web_alma_c11_radii.fits")[1].data
names = np.array([str(x).strip() for x in t["ALMA source name"]])
web = pd.DataFrame({k: pd.to_numeric(pd.Series(np.asarray(t[k]).astype(str)), errors="coerce")
                    for k in ("axratio_sersic", "angle_sersic", "angle_sersic_err", "sersic", "Separation")})
web["src"] = names
web = web[np.isfinite(web["Separation"])].sort_values("Separation").groupby("src").first()

acs = pd.read_csv(f"{AL}/ApJL GIANTS catalogues/GIANTS_ACS_CSOSMOS_sizes.csv").set_index("ID")
uv = pd.read_csv(f"{AL}/uvmcmc_shape.csv")


def stamp_moments(img, r_px):
    """(PA E of N [deg], q) of the light inside r_px of the centroid, N up / E left image (CD1_1 < 0)."""
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]; x = x - (nx - 1) / 2.0; y = y - (ny - 1) / 2.0
    bg = np.nanmedian(img[(np.abs(x) > 0.4 * nx) | (np.abs(y) > 0.4 * ny)]); im = img - bg
    xc = yc = 0.0
    for _ in range(3):
        m = ((x - xc) ** 2 + (y - yc) ** 2 <= r_px ** 2) & np.isfinite(im) & (im > 0); w = im[m]
        xc = np.sum(w * x[m]) / w.sum(); yc = np.sum(w * y[m]) / w.sum()
    m = ((x - xc) ** 2 + (y - yc) ** 2 <= r_px ** 2) & np.isfinite(im) & (im > 0); w = im[m]
    xx = np.sum(w * (x[m] - xc) ** 2) / w.sum(); yy = np.sum(w * (y[m] - yc) ** 2) / w.sum()
    xy = np.sum(w * (x[m] - xc) * (y[m] - yc)) / w.sum()
    ee, nn, en = xx, yy, -xy                                  # East = -x, North = +y
    theta = 0.5 * np.arctan2(2 * en, nn - ee)                 # from N toward E
    d = np.sqrt(0.25 * (ee - nn) ** 2 + en ** 2)
    lam1, lam2 = 0.5 * (ee + nn) + d, 0.5 * (ee + nn) - d
    return np.degrees(theta) % 180.0, float(np.sqrt(max(lam2, 0) / lam1)), xc, yc


def alma_cut(path, ra0, dec0, half_as):
    """(cutout, pix_arcsec, x0_as, y0_as, rms, (bmaj_as, bmin_as, bpa_deg)) of a pipeline map around (ra0, dec0): x0/y0 = where the
    cutout's central pixel sits relative to (ra0, dec0) in the N-up E-left frame (x to the West), arcsec."""
    h = fits.open(path)[0]; img = np.squeeze(h.data).astype(float); H = h.header
    w = WCS(H).celestial; px, py = w.world_to_pixel(SkyCoord(ra0 * AU.deg, dec0 * AU.deg))
    scale = abs(float(H["CDELT2"])) * 3600.0; n = int(round(half_as / scale))
    ix, iy = int(round(float(px))), int(round(float(py)))
    x0, y0 = (ix - float(px)) * scale, (iy - float(py)) * scale
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]; rr = np.hypot(xx - px, yy - py) * scale
    ann = img[(rr >= ALMA_RMS_ANNULUS[0]) & (rr <= ALMA_RMS_ANNULUS[1]) & np.isfinite(img)]
    rms = 1.4826 * np.nanmedian(np.abs(ann - np.nanmedian(ann))) if len(ann) > 50 else np.nan
    cut = np.full((2 * n + 1, 2 * n + 1), np.nan)
    y1, y2, x1, x2 = iy - n, iy + n + 1, ix - n, ix + n + 1
    sy, sx = slice(max(0, y1), min(img.shape[0], y2)), slice(max(0, x1), min(img.shape[1], x2))
    cut[sy.start - y1:sy.stop - y1, sx.start - x1:sx.stop - x1] = img[sy, sx]
    beam = (float(H.get("BMAJ", np.nan)) * 3600.0, float(H.get("BMIN", np.nan)) * 3600.0, float(H.get("BPA", np.nan)))
    return cut.astype(np.float32), scale, x0, y0, float(rms), beam


rows, stamps, stamp_ids = [], [], []
alma = dict(cont=[], mom0=[], cont_scale=[], mom0_scale=[], cont_x0=[], cont_y0=[], mom0_x0=[], mom0_y0=[], alma_ids=[])
for src in ids:
    r = dict(id=src, z=float(z_by[src]), ra_deg=np.nan, dec_deg=np.nan)
    # optical q and PA
    if src in web.index:
        r.update(q_opt=float(web.loc[src, "axratio_sersic"]), q_opt_src="nircam", n_sersic=float(web.loc[src, "sersic"]),
                 pa_opt_deg=float(web.loc[src, "angle_sersic"]) % 180.0, pa_opt_err_deg=float(web.loc[src, "angle_sersic_err"]),
                 pa_opt_src="nircam")
    elif src in acs.index:
        r.update(q_opt=1.0 - float(acs.loc[src, "Ellipticity"]), q_opt_src="acs", n_sersic=float(acs.loc[src, "sersic_n_gim2d"]),
                 pa_opt_deg=np.nan, pa_opt_err_deg=np.nan, pa_opt_src="")
    else:
        r.update(q_opt=np.nan, q_opt_src="", n_sersic=np.nan, pa_opt_deg=np.nan, pa_opt_err_deg=np.nan, pa_opt_src="")
    # the F814W stamp: moments + the cut-out
    fs = glob.glob(f"{C5}/cutouts_centers/{src}_RA*_f814w.fits")
    r.update(pa_stamp_deg=np.nan, q_stamp=np.nan, has_stamp=False)
    if fs:
        img = fits.open(fs[0])[0].data.astype(float)
        if img.ndim == 2 and min(img.shape) > 300:
            pa_s, q_s, xc, yc = stamp_moments(img, R_MOM_AS / PIX)
            r.update(pa_stamp_deg=pa_s, q_stamp=q_s, has_stamp=True)
            h = int(round(STAMP_HALF_AS / PIX)); cy, cx = (img.shape[0] - 1) / 2.0 + yc, (img.shape[1] - 1) / 2.0 + xc
            iy, ix = int(round(cy)), int(round(cx))
            st = img[iy - h:iy + h, ix - h:ix + h]
            stamps.append(st.astype(np.float32)); stamp_ids.append(src)
            sky = WCS(fits.open(fs[0])[0].header).celestial.pixel_to_world(ix, iy)      # the stamp centre on the sky
            r.update(ra_deg=float(sky.ra.deg), dec_deg=float(sky.dec.deg))
            if not np.isfinite(r["pa_opt_deg"]):
                r.update(pa_opt_deg=pa_s, pa_opt_err_deg=np.nan, pa_opt_src="acs_stamp")
    # the CO / continuum uv shapes
    for mode, tag in (("line", "co"), ("cont", "dust")):
        u = uv[(uv["src"] == src) & (uv["mode"] == mode)]
        ok = len(u) and str(u["shape_verdict"].iloc[0]) in ("resolved_elliptical", "elongated_minor_limit")
        r.update({f"q_{tag}": float(u["q"].iloc[0]) if ok else np.nan, f"pa_{tag}_deg": float(u["pa_deg"].iloc[0]) % 180.0 if ok else np.nan,
                  f"pa_{tag}_err_deg": float(u["pa_err_deg"].iloc[0]) if ok else np.nan,
                  f"{tag}_shape": str(u["shape_verdict"].iloc[0]) if len(u) else ""})
    # 3D-Barolo (hc5 only has a converged run: INC fixed at 75, PA fitted per ring)
    r.update(inc_kin_deg=np.nan, pa_kin_deg=np.nan, kin_src="")
    rf = f"{AL}/Dynamics/BB_{src}/rings_final1.txt"
    if os.path.exists(rf):
        try:
            k = np.loadtxt(rf, comments="#")
            if k.ndim == 2 and len(k):
                r.update(inc_kin_deg=float(np.median(k[:, 4])), pa_kin_deg=float(np.median(k[:, 5])), kin_src="3D-Barolo rings_final1")
        except Exception:
            pass
    # the imaging pipeline's continuum and CO moment-0 maps around the stamp centre
    r.update(cont_rms_jyb=np.nan, mom0_rms_jybkms=np.nan, cont_bmaj_as=np.nan, cont_bmin_as=np.nan, cont_bpa_deg=np.nan,
             mom0_bmaj_as=np.nan, mom0_bmin_as=np.nan, mom0_bpa_deg=np.nan, has_alma=False,
             msun_per_jykms=np.nan, msun_per_ujy_dust=np.nan)
    fc = sorted(glob.glob(f"{AL}/Data/{src}/cont/*.image.pbcor.fits")); fm = sorted(glob.glob(f"{AL}/Data/{src}/line/*.line.moments.integrated.fits"))
    if fc and fm and np.isfinite(r["ra_deg"]):
        c_cut, c_sc, c_x0, c_y0, c_rms, c_b = alma_cut(fc[0], r["ra_deg"], r["dec_deg"], ALMA_HALF_AS)
        m_cut, m_sc, m_x0, m_y0, m_rms, m_b = alma_cut(fm[0], r["ra_deg"], r["dec_deg"], ALMA_HALF_AS)
        for k, v in zip(("cont", "mom0", "cont_scale", "mom0_scale", "cont_x0", "cont_y0", "mom0_x0", "mom0_y0", "alma_ids"),
                        (c_cut, m_cut, c_sc, m_sc, c_x0, c_y0, m_x0, m_y0, src)):
            alma[k].append(v)
        r.update(cont_rms_jyb=c_rms, mom0_rms_jybkms=m_rms, cont_bmaj_as=c_b[0], cont_bmin_as=c_b[1], cont_bpa_deg=c_b[2],
                 mom0_bmaj_as=m_b[0], mom0_bmin_as=m_b[1], mom0_bpa_deg=m_b[2], has_alma=True)
    if src in gd.index:
        g = gd.loc[src]
        if bool(g.get("det_co", False)) and float(g.get("S_co_jykms", 0)) > 0:
            r["msun_per_jykms"] = float(g["MH2_fid"]) / float(g["S_co_jykms"])
        if bool(g.get("det_dust", False)) and float(g.get("S_dust_ujy", 0)) > 0:
            r["msun_per_ujy_dust"] = 10.0 ** float(g["logMdust_fid"]) / float(g["S_dust_ujy"])
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/almac11_orientation.csv", index=False)
_n = min([a.shape[0] for a in alma["cont"] + alma["mom0"]]) if alma["cont"] else 0      # one array: crop every ALMA cutout to the smallest
_crop = lambda a: a[(a.shape[0] - _n) // 2:(a.shape[0] - _n) // 2 + _n, (a.shape[1] - _n) // 2:(a.shape[1] - _n) // 2 + _n]
np.savez_compressed(f"{OUT}/almac11_stamps_f814w.npz", ids=np.array(stamp_ids), stamps=np.array(stamps), pix_arcsec=PIX,
                    half_arcsec=STAMP_HALF_AS, band="ACS F814W", note="N up, E left; background-subtracted no; centred on the light centroid",
                    alma_ids=np.array(alma["alma_ids"]), cont=np.array([_crop(a) for a in alma["cont"]]), mom0=np.array([_crop(a) for a in alma["mom0"]]),
                    cont_pix_arcsec=np.array(alma["cont_scale"]), mom0_pix_arcsec=np.array(alma["mom0_scale"]),
                    cont_x0_as=np.array(alma["cont_x0"]), cont_y0_as=np.array(alma["cont_y0"]), mom0_x0_as=np.array(alma["mom0_x0"]), mom0_y0_as=np.array(alma["mom0_y0"]),
                    alma_half_arcsec=ALMA_HALF_AS, alma_note="Imaging_v1 pbcor continuum [Jy/beam] and CO(3-2) moment-0 [Jy/beam km/s], N up E left, "
                    "centred (to the pixel) on the stamp centre; x0/y0 = the offset of the central pixel from that centre [arcsec, x West y North]")
print(df[["id", "z", "q_opt", "q_opt_src", "pa_opt_deg", "pa_opt_src", "pa_stamp_deg", "q_stamp", "q_co", "pa_co_deg", "co_shape",
          "inc_kin_deg", "pa_kin_deg"]].round(2).to_string())
print(df[["id", "ra_deg", "dec_deg", "cont_rms_jyb", "mom0_rms_jybkms", "cont_bmaj_as", "mom0_bmaj_as", "msun_per_jykms", "msun_per_ujy_dust"]].round(5).to_string())
print(f"stamps: {len(stamp_ids)} x {np.array(stamps).shape[1:]} px; ALMA cutouts: {len(alma['alma_ids'])} x {_n} px")
