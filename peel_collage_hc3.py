#!/usr/bin/env python
"""One galaxy, three layers: the "peel" collage of an ALMA-C11 source.

A single square field, North up and East left, cut into three 120-degree slices that meet at the galaxy's centre.
Every slice shows the SAME physical field of view (``--ext`` kpc half-size, centred on the fitted kinematic centre)
through a different lens, so the picture reads as three properties peeled off one object:

  obs  the JWST image of the source itself (a Lupton RGB of the DJA NIRCam cutouts, ``--obs-bands``; hc3 has no
       MIRI coverage in COSMOS-Web -- its F770W entry is masked in the catalogue and every F770W "cutout" on disk is a
       "no cutout found" stub -- so the reddest NIRCam band plays the MIR part; pass ``--obs-bands f770w`` for a
       source that has it),
  kin  the KinMS best-fit model of the CO(3-2) kinematics (the velocity field of the model cube, iso-velocity
       contours, the model's moment-0 outline and the beam; rebuilt from the fitted parameters by
       ``peel_collage_kinms_model.py`` in the kinms-env),
  sim  the closest SIMBA galaxy of that source (``agn_states_companions.csv`` of agn_states_dusty_vs_pure.ipynb
       Part 27: the rank-1 companion, its reduced particle cutout) seen from the observed line of sight -- the
       source's inclination and position angle imposed on the companion's disc frame -- as its projected dust
       SURFACE DENSITY (log Sigma_dust, a sequential colour map, no stellar underlay, no beam smoothing) with the
       dust-weighted gas flow drawn as streamlines, coloured by the line-of-sight velocity on the SAME colour scale
       as the kinematic slice.

Viewing geometry of the simulated slice: the frame is `_view27` of the notebook (the projection frame [West, North,
toward-the-observer] that shows a galaxy with face-on frame E at inclination i, apparent major axis at PA East of
North); E is the disc frame of ``--frame`` (``dust_L``: the angular momentum of the dust inside ``--frame-rmax`` kpc,
so what is tilted is the dust disc; ``stellar`` = the stored stellar principal frame, Part 27's default); i and PA
come from the companions table (``--inc obs``: the q-based / Barolo inclination Part 27 used) or from the KinMS fit
(``--inc kin``).  The companion is then turned by 180 degrees about the line of sight if needed so that its receding
side lies on the side the KinMS position angle says (PA = the receding side, E of N).

Rendering: ``--renderer sphviewer`` (the default when py-sphviewer imports) draws the maps with simbanator's own
``ReducedRender`` (``simbanator/plotting/rendering.py``, loaded standalone so the package's yt import is not
needed) — the SAME ``sigma_map`` / ``velocity_field`` / ``weighted_map`` products the notebooks show, on the imposed
frame (``set_evecs``).  py-sphviewer is not installed in the base env: the script adds the cluster pd39 env's
``py_sphviewer`` + ``pykdtree`` eggs (python 3.9, x86_64) to ``sys.path``, from ``output/peel_collage/_sphviewer_egg/``
(copied there 2026-09-07) or straight from the cluster mount.  ``--renderer splat`` is the fallback: every particle a
Gaussian of sigma = ``--kernel`` x its ``--knn``-th neighbour distance (0.3 x h_32 ~ sphviewer's cubic spline),
splatted at the canvas resolution with signed weights kept.  Surface densities are normalised to the member mass
inside the frame.  Nothing else is smoothed: no beam, no band smoothing, no contour smoothing.


    python peel_collage_hc3.py [--source hc3] [--ext 16] [--rot 90] [--order obs,kin,sim] [--inc obs|kin] [--tag _v]
                               [--frame dust_L|H2_L|gas_L|stellar] [--obs-bands f115w-clear,f277w-clear,f444w-clear]

The page (``--page white``, default) is white: the sky of every slice fades into it outside an ELLIPSE that hugs the
galaxy (``--fade-a`` semi-major axis along the source's PA, ``--fade-q`` axis ratio; fully white from ``--fade-r1`` x
that ellipse, a smoothstep in between), the frame is cropped to that outer ellipse's bounding box (a landscape
picture for an E-W disc; ``--no-crop`` keeps the square), the names and the scale bar are black and the seams turn
from white on the sky to black on the page — a print with little ink.  ``--page black`` is the all-black version.

For the thesis cover (2026-09-07) the names are outreach-style and WRAPPED around the galaxy: "Optical+mid-IR
observations", "Dynamics", "Cosmological simulations" set letter by letter along a ring just outside the fade
ellipse (``--label-r``, ``--label-size``, ``--label-angles``), upright and reading left to right; the kpc bar sits on
the JWST slice's own sky.

Annotations are deliberately minimal: the three names (JWST / ALMA / SIMBA) in a corner of their slice, a kpc scale bar
in one free corner, plain white seams; no colour bars, no title, no beam, no contours on the JWST image (the ALMA
slice keeps thin iso-velocity lines, ``--vel-contours 0`` removes them).  The velocity colour scale (RdBu, +-vmax from
the model's 99.5th percentile) is shared by the model field and the simulated flow lines; the KinMS model is drawn
as it comes out of the model cube — nothing convolved or smoothed beyond KinMS's own beam.

Products (``output/peel_collage/``): ``peel_collage_<source>.png`` / ``.pdf`` and ``peel_collage_<source>_layers.png``
(the three layers side by side, uncut, as a check that they line up).
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18
from astropy.visualization import make_lupton_rgb
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Ellipse
from matplotlib.path import Path
from matplotlib.patheffects import withStroke

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
# the JWST cutouts (DJA thumbnail service, imaging_v1/cutouts.py; 0.05"/pix; N up E left): the 60" ones first, then the 20" ones
OBS_FILES = {"hc3": [f"{HOME}/Desktop/Proposals/JWST/Cycle 5/Data/cutouts/hc3_RA149.8458333333415_DEC1.9738888888890223_{{band}}.fits",
                     f"{HOME}/Nextcloud/alma_cycle11/dja/dja_hc3_RA149.845755_DEC1.974381_{{band}}.fits"]}

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="hc3")
ap.add_argument("--box", default="cis25")
ap.add_argument("--prefix", default="m25n512")
ap.add_argument("--ext", type=float, default=30.0, help="half-size of the field [kpc]")
ap.add_argument("--npix", type=int, default=1200, help="canvas pixels a side")
ap.add_argument("--rot", type=float, default=105.0, help="angle of the first seam [deg, from +x = West, counter-clockwise]; 105 = obs holds the whole East half, kin below (both velocity signs), sim upper-right")
ap.add_argument("--order", default="obs,kin,sim", help="the slices counter-clockwise from --rot")
ap.add_argument("--inc", default="obs", help="'obs' (the companions table: q-based / Barolo), 'kin' (the KinMS fit) or a number [deg]")
ap.add_argument("--pa", default="obs", help="'obs' (the companions table), 'kin' (the KinMS fit mod 180) or a number [deg E of N]")
ap.add_argument("--frame", default="dust_L", help="the companion's disc frame: dust_L | H2_L | gas_L | stellar")
ap.add_argument("--frame-rmax", type=float, default=10.0)
ap.add_argument("--obs-bands", default="f115w-clear,f277w-clear,f444w-clear", help="one band (grey/hot) or three (B,G,R)")
ap.add_argument("--obs-stretch", type=float, default=0.8, help="Lupton stretch, in units of the core flux (each band is first normalised to its flux inside --obs-core kpc, so the core comes out white)")
ap.add_argument("--obs-q", type=float, default=4.0)
ap.add_argument("--obs-core", type=float, default=4.0, help="[kpc] the normalisation aperture of the bands")
ap.add_argument("--obs-gamma", type=float, default=0.75, help="gamma applied to the JWST RGB after the Lupton stretch (< 1 lifts the outer disc)")
ap.add_argument("--obs-weights", default="1.0,1.0,1.0", help="R,G,B weights after the normalisation (the colour balance)")
ap.add_argument("--renderer", default="auto", help="auto | sphviewer (ReducedRender, the notebooks' maps) | splat (own Gaussian kernels)")
ap.add_argument("--knn", type=int, default=32, help="splat: the neighbour that sets each particle's kernel size")
ap.add_argument("--kernel", type=float, default=0.3, help="Gaussian sigma of a particle in units of its --knn-th neighbour distance")
ap.add_argument("--dust-dyn", type=float, default=2.8, help="decades of Sigma_dust below its --dust-pmax percentile")
ap.add_argument("--dust-pmax", type=float, default=99.9, help="the percentile of Sigma_dust that saturates the colour map")
ap.add_argument("--dust-cmap", default="dust", help="colour map of log Sigma_dust: 'dust' (black-red-orange-white) or any matplotlib name")
ap.add_argument("--stream-density", type=float, default=1.8)
ap.add_argument("--stream-min", type=float, default=0.012, help="no flow lines where the projected dust is below this fraction of its peak")
ap.add_argument("--vmax", type=float, default=0.0, help="velocity colour scale [km/s]; 0 = from the model")
ap.add_argument("--no-flip", action="store_true", help="do not turn the companion to put its receding side where the data's is")
ap.add_argument("--reduced", default=None, help="the reduced particle file (default: output/<box>/reduced_particles/snap_NNN/... in the repo, else on the cluster mount)")
ap.add_argument("--out", default=os.path.join(HERE, "output", "peel_collage"))
ap.add_argument("--dpi", type=int, default=220)
ap.add_argument("--page", default="white", help="white = the sky fades radially into a white page (black text; ink-saving), black = black page, white text")
ap.add_argument("--fade-a", type=float, default=16.5, help="semi-major axis [kpc] of the ellipse (along the galaxy's PA) where the fade to the page starts")
ap.add_argument("--fade-q", type=float, default=0.52, help="axis ratio of that ellipse")
ap.add_argument("--fade-r1", type=float, default=1.3, help="the page is fully white from this multiple of the ellipse outward")
ap.add_argument("--fade-pa", default="obs", help="PA of the ellipse [deg E of N]: 'obs' = the drawn PA of the companions table")
ap.add_argument("--no-crop", action="store_true", help="keep the square field instead of cropping to the fade ellipse")
ap.add_argument("--label-size", type=float, default=14.0, help="font size of the names wrapped around the galaxy [pt]")
ap.add_argument("--label-r", type=float, default=0.0, help="the ring the names sit on, as a multiple of the fade ellipse; 0 = --fade-r1 + 0.07")
ap.add_argument("--label-angles", default="", help="where each name is centred, e.g. obs:140,kin:280,sim:60 [deg, from +x = West, counter-clockwise]; default = the slice mid-angles pulled toward the top / bottom")
ap.add_argument("--vel-contours", type=float, default=50.0, help="iso-velocity contour step on the ALMA slice [km/s]; 0 = none")
ap.add_argument("--scale-kpc", type=float, default=5.0, help="the scale bar [kpc]; 0 = none")
ap.add_argument("--tag", default="", help="suffix of the product names (a layout variant)")
args = ap.parse_args()
src = args.source
os.makedirs(args.out, exist_ok=True)

# ── py-sphviewer + simbanator's ReducedRender (optional) ────────────────────────────────────────────────────────
import glob, importlib.util
CLUSTER = "/run/user/1000/gvfs/sftp:host=slurmusrint2.cis.gov.pl,user=glorenzon/mnt/home/glorenzon/analize_simba_cgm"
_PD39 = os.path.dirname(CLUSTER) + "/miniconda3/envs/pd39/lib/python3.9/site-packages"
ReducedRender = None
if args.renderer in ("auto", "sphviewer"):
    try:
        import sphviewer                                                 # noqa: F401
    except ImportError:
        for root in (os.path.join(HERE, "output", "peel_collage", "_sphviewer_egg"), _PD39):     # the local copy first: the mount may hang
            eggs = glob.glob(os.path.join(root, "py_sphviewer-*.egg")) + glob.glob(os.path.join(root, "pykdtree-*.egg"))
            for egg in eggs:
                sys.path.insert(0, egg)
            if len(eggs) >= 2:
                break
        try:
            import sphviewer                                             # noqa: F401
        except ImportError:
            sphviewer = None
    if sphviewer is not None:
        _spec = importlib.util.spec_from_file_location("simbanator_rendering", os.path.join(HERE, "simbanator", "plotting", "rendering.py"))
        _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        ReducedRender = _mod.ReducedRender
    elif args.renderer == "sphviewer":
        sys.exit("py-sphviewer is not importable (no local install, no egg in output/peel_collage/_sphviewer_egg, cluster not mounted)")
RENDERER = "sphviewer" if ReducedRender is not None else "splat"
print(f"renderer: {RENDERER}")

# ── the companion and the observed geometry ──────────────────────────────────────────────────────────────────────
comp = pd.read_csv(os.path.join(HERE, "output", "box_resolution", "agn_states", "agn_states_companions.csv"))
row = comp[(comp["id"] == src) & (comp["drawn"] == True)]
if len(row) == 0:
    row = comp[(comp["id"] == src) & (comp["rank"] == 1)]
row = row.iloc[0]
z_obs = float(row["z_obs"]); kpc_as = float(Planck18.kpc_proper_per_arcmin(z_obs).value / 60.0)
snap, gid = int(row["anchor_snap"]), int(row["gal_id"])
_rel = os.path.join("output", args.box, "reduced_particles", "snap_%03d" % snap, "%s_snap%03d_gal%06d.h5" % (args.prefix, snap, gid))
reduced = args.reduced or next((os.path.join(r, _rel) for r in (HERE, CLUSTER) if os.path.exists(os.path.join(r, _rel))), os.path.join(HERE, _rel))
print(f"{src}: z = {z_obs:.4f} ({kpc_as:.3f} kpc/arcsec); companion {args.box} snap {snap} gal {gid} "
      f"(z_sim {row['z_sim']:.2f}, age {row['age']:.1f} Gyr, log M* {row['lm']:.2f}, log Md/M* {row['lfd']:.2f}, class {row['cls']})")

kin = np.load(os.path.join(args.out, f"{src}_kinms_model.npz"))
vsys, inc_kin, pa_kin = float(kin["vsys"]), float(kin["inc"]), float(kin["pa"])
ra_c, dec_c = float(kin["params"][0]), float(kin["params"][1])          # the fitted centre [deg]
inc_use = {"obs": float(row["inc_used"]) if np.isfinite(row.get("inc_used", np.nan)) else float(row["inc_deg"]), "kin": inc_kin}.get(args.inc, None)
inc_use = float(args.inc) if inc_use is None else inc_use
pa_use = {"obs": float(row["pa_deg"]), "kin": pa_kin % 180.0}.get(args.pa, None)
pa_use = float(args.pa) if pa_use is None else pa_use
print(f"  KinMS: Vsys {vsys:.0f} km/s, inc {inc_kin:.1f}, PA {pa_kin:.1f} (receding side); drawn geometry: inc {inc_use:.1f}, PA {pa_use:.1f} ({args.inc}/{args.pa})")

# ── the canvas: x = West [kpc] (right), y = North [kpc] (up), centred on the fitted centre ────────────────────────
EXT, N = float(args.ext), int(args.npix)
xs = np.linspace(-EXT, EXT, N); X, Y = np.meshgrid(xs, xs)             # X[y, x]
pix_kpc = xs[1] - xs[0]
extent = [-EXT, EXT, -EXT, EXT]
centre = SkyCoord(ra_c * u.deg, dec_c * u.deg)
d_east = (-X / kpc_as) * u.arcsec                                       # x West -> East offset is -x
d_north = (Y / kpc_as) * u.arcsec
sky = centre.spherical_offsets_by(d_east.ravel(), d_north.ravel())     # the sky position of every canvas pixel

# ── layer 1: the JWST image ──────────────────────────────────────────────────────────────────────────────────────
bands = [b.strip() for b in args.obs_bands.split(",") if b.strip()]
def load_band(band):
    f = next((c.format(band=band) for c in OBS_FILES[src] if os.path.exists(c.format(band=band))), None)
    if f is None:
        sys.exit(f"no {band} cutout of {src} in {OBS_FILES[src]}")
    hd = fits.open(f)[0]; img = np.asarray(hd.data, float); w = WCS(hd.header)
    print(f"  obs: {band} <- {f.replace(HOME, '~')} ({img.shape[0]} px = {img.shape[0] * 0.05:.0f}\" a side)")
    px, py = w.world_to_pixel(sky)
    on = map_coordinates(img, [py.reshape(N, N), px.reshape(N, N)], order=1, mode="constant", cval=np.nan)
    # background + noise from the frame's outer 15 % (the source is at the centre)
    ring = (np.abs(X) > 0.85 * EXT) | (np.abs(Y) > 0.85 * EXT)
    bg = np.nanmedian(on[ring]); mad = 1.4826 * np.nanmedian(np.abs(on[ring] - bg))
    return on - bg, mad, hd.header.get("FILTER", band)
layers = [load_band(b) for b in bands]
core = np.hypot(X, Y) < args.obs_core
norm = [float(np.nanmean(l[0][core])) for l in layers]                # the core flux per band: the normalisation
if len(layers) == 3:
    wR, wG, wB = [float(x) for x in args.obs_weights.split(",")]
    r_, g_, b_ = [np.nan_to_num(layers[i][0] / norm[i]) * wgt for i, wgt in ((2, wR), (1, wG), (0, wB))]
    nz = [layers[i][1] / norm[i] * wgt for i, wgt in ((2, wR), (1, wG), (0, wB))]
    noise = float(np.mean(nz))
    rgb_obs = make_lupton_rgb(r_, g_, b_, minimum=nz, stretch=args.obs_stretch, Q=args.obs_q)   # 1 sigma floor per band
    rgb_obs = (255 * (rgb_obs / 255.0) ** args.obs_gamma).astype(np.uint8)
    obs_label = f"JWST/NIRCam  {layers[0][2].split('-')[0]} + {layers[1][2].split('-')[0]} + {layers[2][2].split('-')[0]}"
else:
    im, noise, nm = layers[0]; im = np.nan_to_num(im / norm[0]); noise = noise / norm[0]
    v = np.arcsinh(np.clip(im, 0, None) / args.obs_stretch) / np.arcsinh(1.0 / args.obs_stretch)
    rgb_obs = plt.get_cmap("magma")(np.clip(v, 0, 1))[..., :3]
    obs_label = f"JWST  {nm}"
print(f"  obs: {obs_label}; core-normalised noise {noise:.3g} (stretch {args.obs_stretch}, Q {args.obs_q})")

# ── layer 2: the KinMS model velocity field ─────────────────────────────────────────────────────────────────────
ra0, dec0 = float(kin["ra0"]), float(kin["dec0"])
ref = SkyCoord(ra0 * u.deg, dec0 * u.deg)
dE, dN = ref.spherical_offsets_to(sky)                                  # East / North offsets of the canvas pixels from the KinMS reference pixel
dE = dE.to(u.arcsec).value.reshape(N, N); dN = dN.to(u.arcsec).value.reshape(N, N)
x_as, y_as, pix_as = np.asarray(kin["x_as"]), np.asarray(kin["y_as"]), float(kin["pix_as"])
nx, ny = len(x_as), len(y_as)
ix = nx // 2 - dE / pix_as                                              # x_as = (i - n//2) * cdelt1 * 3600, cdelt1 < 0: East offsets
iy = ny // 2 + dN / pix_as
def on_canvas(arr, order=1):
    return map_coordinates(np.asarray(arr, float), [ix, iy], order=order, mode="constant", cval=np.nan)
from scipy.ndimage import distance_transform_edt
_m1 = np.asarray(kin["model_mom1"], float) - vsys
_bad = ~np.isfinite(_m1)
if _bad.any():                                                           # outside the model's CO the moment-1 is undefined: fill with the nearest
    _idx = distance_transform_edt(_bad, return_distances=False, return_indices=True)   # defined value, so the interpolation to the canvas never
    _m1 = _m1[tuple(_idx)]                                               # blends toward zero at the edge (the mask below cuts the map)
m0 = on_canvas(np.nan_to_num(kin["model_mom0"]), order=3); m1 = on_canvas(_m1)   # cubic for the mask edge only (no pixel stairs)
d0 = on_canvas(np.nan_to_num(kin["data_mom0"]))
m1_msk = np.where(m0 > 0.08 * np.nanmax(m0), m1, np.nan)
vmax = float(args.vmax) if args.vmax > 0 else float(np.ceil(np.nanpercentile(np.abs(m1_msk), 99.5) / 25.0) * 25.0)
bmaj, bmin, bpa = [float(v) for v in kin["beam"]]
print(f"  kin: model velocity range {np.nanmin(m1_msk):.0f}..{np.nanmax(m1_msk):.0f} km/s -> colour scale ±{vmax:.0f}; beam {bmaj:.2f}\" x {bmin:.2f}\" = {bmaj*kpc_as:.1f} x {bmin*kpc_as:.1f} kpc")

# ── layer 3: the SIMBA companion ────────────────────────────────────────────────────────────────────────────────
with h5py.File(reduced, "r") as f:
    attrs = {k: f.attrs[k] for k in f.attrs}
    G = {k: np.asarray(f["gas"][k]) for k in f["gas"]}
    S = {k: np.asarray(f["star"][k]) for k in f["star"]}
for P in (G, S):
    keep = np.asarray(P["member"], bool)
    for k in list(P):
        P[k] = P[k][keep]
E = np.asarray(attrs["evecs"], float)
if np.linalg.det(E) < 0:
    E = E.copy(); E[:, 2] *= -1.0

def spin_frame(pos, vel, w, rmax):
    """Columns (major, intermediate, spin) of the angular momentum of the weighted particles inside rmax [kpc]."""
    r = np.linalg.norm(pos, axis=1); k = (r <= rmax) & (w > 0)
    if k.sum() < 20:
        return None
    vb = np.average(vel[k], axis=0, weights=w[k])
    L = np.sum(w[k, None] * np.cross(pos[k], vel[k] - vb), axis=0); L /= np.linalg.norm(L)
    # the in-plane axes: the principal axes of the weighted positions projected on the plane
    p = pos[k] - np.outer(pos[k] @ L, L)
    I = (p * w[k, None]).T @ p / w[k].sum()
    ev, evec = np.linalg.eigh(I); e0 = evec[:, np.argmax(ev)]; e0 -= (e0 @ L) * L; e0 /= np.linalg.norm(e0)
    e1 = np.cross(L, e0)
    return np.column_stack([e0, e1, L])

FRAMES = {"dust_L": ("gas", "m_dust"), "H2_L": ("gas", "m_H2"), "gas_L": ("gas", "m_gas"), "star_L": ("star", "m_star")}
if args.frame in FRAMES:
    comp_, wkey = FRAMES[args.frame]; P = G if comp_ == "gas" else S
    Es = spin_frame(np.asarray(P["pos"], float), np.asarray(P["vel"], float), np.asarray(P[wkey], float), args.frame_rmax)
    if Es is not None:
        E = Es
        print(f"  sim: frame = {args.frame} (spin inside {args.frame_rmax:g} kpc)")
    else:
        print(f"  sim: {args.frame} has too few particles, the stellar frame is used")
else:
    print("  sim: frame = the stored stellar principal axes")

def view_frame(E, inc_deg, pa_deg, az_deg=0.0):
    """Part 27's _view27: the projection frame [West, North, toward-the-observer] (columns) that shows the galaxy whose
    face-on frame is E (columns major, intermediate, spin) at inclination inc_deg, apparent major axis at pa_deg E of N."""
    e0, e1, ez = E[:, 0], E[:, 1], E[:, 2]
    i, az, pa = np.radians(inc_deg), np.radians(az_deg), np.radians(pa_deg)
    uu = np.cos(az) * e1 + np.sin(az) * e0
    n = np.cos(i) * ez + np.sin(i) * uu; n /= np.linalg.norm(n)
    m = np.cross(ez, n)
    if np.linalg.norm(m) < 1e-6:
        m = e0 - np.dot(e0, n) * n
    m /= np.linalg.norm(m)
    p = np.cross(n, m)
    Nn = np.cos(pa) * m - np.sin(pa) * p
    Ea = np.sin(pa) * m + np.cos(pa) * p
    return np.column_stack([-Ea, Nn, n])

V = view_frame(E, inc_use, pa_use)
gpos = np.asarray(G["pos"], float) @ V; gvel = np.asarray(G["vel"], float) @ V
spos = np.asarray(S["pos"], float) @ V; svel = np.asarray(S["vel"], float) @ V
wd, wg, wh2, ws = [np.asarray(G[k], float) for k in ("m_dust", "m_gas", "m_H2")] + [np.asarray(S["m_star"], float)]
rin = np.hypot(gpos[:, 0], gpos[:, 1]) <= EXT
wb = np.where(rin, wd, 0.0)
vbulk = np.average(gvel, axis=0, weights=wb) if wb.sum() > 0 else np.average(gvel, axis=0, weights=wg)
gvel = gvel - vbulk; svel = svel - np.average(svel, axis=0, weights=ws)
# the receding side: v . n > 0 is toward the observer (approaching); the data's receding side is at PA_kin (E of N)
pa_r = np.radians(pa_kin)
rec_dir = np.array([-np.sin(pa_r), np.cos(pa_r)])                       # (x West, y North) unit vector toward the receding side
side = (gpos[:, :2] @ rec_dir)
k = (np.abs(side) > 1.0) & (wd > 0)
v_rec = np.average(gvel[k, 2], weights=wd[k] * np.abs(side[k])) if k.sum() > 10 else 0.0
flipped = False
if (v_rec > 0) and not args.no_flip:                                    # that side approaches: turn the galaxy by 180 deg about the LOS
    for A in (gpos, gvel, spos, svel):
        A[:, 0] *= -1.0; A[:, 1] *= -1.0
    flipped = True
print(f"  sim: receding-side test: <v . n> = {v_rec:+.0f} km/s on the data's receding side (n toward the observer) -> {'turned by 180 deg' if flipped else 'as is'}")

def knn_h(pos, k):
    t = cKDTree(pos); d, _ = t.query(pos, k=k + 1); return d[:, -1]

def splat(xy, w, h_kpc, ext, npix, nclass=16, hmin_pix=0.5):
    """Adaptive-kernel projection: sum of Gaussians of sigma = --kernel x h per particle, in nclass width classes."""
    img = np.zeros((npix, npix)); pix = 2 * ext / npix
    sig = np.clip(args.kernel * h_kpc / pix, hmin_pix, 0.35 * npix)
    edges = np.geomspace(sig.min(), sig.max() * 1.0001, nclass + 1)
    cls = np.clip(np.searchsorted(edges, sig, side="right") - 1, 0, nclass - 1)
    for c in range(nclass):
        m = (cls == c) & np.isfinite(w) & (w != 0)                       # signed weights (velocity moments) must keep their negatives
        if not m.any():
            continue
        H, _, _ = np.histogram2d(xy[m, 1], xy[m, 0], bins=npix, range=[[-ext, ext], [-ext, ext]], weights=w[m])
        s = float(np.sqrt(np.mean(sig[m] ** 2)))
        img += gaussian_filter(H, s, mode="constant")
    return img

NV = 96                                                                  # the flow-line grid
inside = (np.abs(gpos[:, 0]) <= EXT) & (np.abs(gpos[:, 1]) <= EXT)
if RENDERER == "sphviewer":
    V_eff = V.copy()
    if flipped:
        V_eff[:, 0] *= -1.0; V_eff[:, 1] *= -1.0                          # the 180-degree turn about the LOS, as a frame
    rr = ReducedRender(reduced, members_only=True)
    rr.set_evecs(V_eff, label="observed LOS", info=dict(inc=inc_use, pa=pa_use, flipped=flipped))
    sig_dust = rr.sigma_map("dust", ex=EXT, spos="faceon", xsize=N)[0]  # Msun / kpc^2, normalised to the mass in the frame
    XV, YV, VX, VY, den = rr.velocity_field(ex=EXT, spos="faceon", comp="gas", weight="m_dust", ngrid=NV, subtract_bulk=True)
    _vl = rr.project("gas", "faceon", what="vel")[:, 2]
    _wd = np.asarray(rr.parts["gas"]["m_dust"], float)
    _vl = _vl - (np.average(_vl, weights=_wd) if _wd.sum() > 0 else 0.0)
    VL = rr.weighted_map(_vl, comp="gas", weight="m_dust", ex=EXT, spos="faceon", xsize=NV, fill=np.nan)[0]
    del _vl, _wd
else:
    hg = knn_h(np.asarray(G["pos"], float), args.knn)
    sig_dust = splat(gpos[:, :2], wd, hg, EXT, N)                        # at the canvas resolution
    pixA = (2 * EXT / N) ** 2
    if sig_dust.sum() > 0:
        sig_dust *= wd[inside].sum() / (sig_dust.sum() * pixA)         # Msun / kpc^2
    den = splat(gpos[:, :2], wd, hg, EXT, NV)
    vx = splat(gpos[:, :2], wd * gvel[:, 0], hg, EXT, NV); vy = splat(gpos[:, :2], wd * gvel[:, 1], hg, EXT, NV); vl = splat(gpos[:, :2], wd * gvel[:, 2], hg, EXT, NV)
    with np.errstate(invalid="ignore", divide="ignore"):
        VX = np.where(den > 0, vx / den, 0.0); VY = np.where(den > 0, vy / den, 0.0); VL = np.where(den > 0, vl / den, np.nan)
    xv = np.linspace(-EXT, EXT, NV); XV, YV = np.meshgrid(xv, xv)
print(f"  sim [{RENDERER}]: {inside.sum()} gas members in the frame, M_dust {wd[inside].sum():.2e} Msun, peak Sigma_dust {sig_dust.max()/1e6:.3f} Msun/pc^2")
_k = (np.hypot(gpos[:, 0], gpos[:, 1]) < 10) & (wd > 0)
_xx, _yy, _xy = [np.average(a, weights=wd[_k]) for a in (gpos[_k, 0] ** 2, gpos[_k, 1] ** 2, gpos[_k, 0] * gpos[_k, 1])]
_d = np.sqrt(0.25 * (_xx - _yy) ** 2 + _xy ** 2); _q = np.sqrt(max(0.5 * (_xx + _yy) - _d, 0) / (0.5 * (_xx + _yy) + _d))
_W, _E = _k & (gpos[:, 0] > 2), _k & (gpos[:, 0] < -2)
print(f"  sim: projected dust inside 10 kpc: q = {_q:.2f}, PA = {np.degrees(0.5 * np.arctan2(-2 * _xy, _yy - _xx)) % 180:.0f} deg; "
      f"<v_los> (receding +) West {-np.average(gvel[_W, 2], weights=wd[_W]):+.0f}, East {-np.average(gvel[_E, 2], weights=wd[_E]):+.0f} km/s")
VLOS = -VL                                                               # positive = receding, the data's convention
dust_frac = np.clip(den / np.percentile(den[den > 0], 97), 0, 1) if np.any(den > 0) else den
nodust = (den / den.max()) < args.stream_min
VX = np.ma.masked_where(nodust, VX); VY = np.ma.masked_where(nodust, VY)
VLOS = np.where(nodust, 0.0, np.nan_to_num(VLOS))

# ── colours ─────────────────────────────────────────────────────────────────────────────────────────────────────
cm_dust = (mcolors.LinearSegmentedColormap.from_list("sb_dust", ["#000000", "#2a0800", "#6e1a02", "#b8420a", "#e8781f", "#ffb356", "#ffe6b0", "#ffffff"])
           if args.dust_cmap == "dust" else plt.get_cmap(args.dust_cmap))
cm_vel = plt.get_cmap("RdBu_r")
vnorm = mcolors.Normalize(-vmax, vmax)

def log_stretch(img, dyn, pmax=99.7):
    pos = img[np.isfinite(img) & (img > 0)]
    if len(pos) == 0:
        return np.zeros_like(img)
    top = np.log10(np.percentile(pos, pmax)); bot = top - dyn
    return np.clip((np.log10(np.where(img > 0, img, 10 ** (bot - 1))) - bot) / (top - bot), 0, 1)

sd_pc2 = sig_dust / 1e6                                                 # Msun / pc^2
_pos = sd_pc2[np.isfinite(sd_pc2) & (sd_pc2 > 0)]
sd_top = float(np.log10(np.percentile(_pos, args.dust_pmax))) if len(_pos) else 0.0; sd_bot = sd_top - args.dust_dyn
rgb_sim = cm_dust(log_stretch(sd_pc2, args.dust_dyn, args.dust_pmax))[..., :3]
print(f"  sim: log Sigma_dust colour range {sd_bot:.2f} .. {sd_top:.2f} Msun/pc^2")
rgb_kin = cm_vel(vnorm(np.nan_to_num(m1_msk)))[..., :3]
rgb_kin[~np.isfinite(m1_msk)] = 0.0                                      # black where the model has no CO (the moment-0 mask)

# ── the slices ──────────────────────────────────────────────────────────────────────────────────────────────────
order = [x.strip() for x in args.order.split(",")]
assert sorted(order) == ["kin", "obs", "sim"], "--order must name obs, kin and sim once"
def wedge(a0, a1, r=4.0):
    ang = np.linspace(np.radians(a0), np.radians(a1), 60)
    return np.vstack([[0, 0], np.column_stack([r * EXT * np.cos(ang), r * EXT * np.sin(ang)])])
bounds = [args.rot + 120.0 * k for k in range(4)]
NAMES = {"obs": "Optical+mid-IR observations", "kin": "Dynamics", "sim": "Cosmological simulations"}
images = {"obs": rgb_obs, "kin": rgb_kin, "sim": rgb_sim}

WHITE = args.page == "white"
PAGE, INK = ("white", "black") if WHITE else ("black", "white")
_pa_f = np.radians(pa_use if args.fade_pa == "obs" else float(args.fade_pa))
_u = np.array([-np.sin(_pa_f), np.cos(_pa_f)]); _v = np.array([np.cos(_pa_f), np.sin(_pa_f)])   # the major / minor axis on the sky (x West, y North)
def page_w(x, y):
    """The weight of the page colour at (x, y) [kpc]: 0 inside the fade ellipse, 1 beyond --fade-r1 x it (a smoothstep)."""
    if not WHITE:
        return np.zeros(np.broadcast(x, y).shape)
    xe = (x * _u[0] + y * _u[1]) / args.fade_a; ye = (x * _v[0] + y * _v[1]) / (args.fade_a * args.fade_q)
    t = np.clip((np.hypot(xe, ye) - 1.0) / max(args.fade_r1 - 1.0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)
if WHITE:
    _w = page_w(X, Y)[..., None]
    for _k in images:
        _im = np.asarray(images[_k], float); _im = _im / 255.0 if _im.max() > 1.5 else _im
        images[_k] = _im[..., :3] * (1 - _w) + _w                         # the sky fades into the white page
# the view: the bounding box of the outer fade ellipse (plus a margin), or the full square
R_TEXT = args.label_r if args.label_r > 0 else args.fade_r1 + 0.07
if args.no_crop:
    XL = YL = EXT
else:
    R_TEXT = args.label_r if args.label_r > 0 else args.fade_r1 + 0.07
    _rf = R_TEXT + 0.17                                                  # the names' ring plus a margin for the letters
    _A, _B = args.fade_a * _rf, args.fade_a * args.fade_q * _rf
    XL = min(EXT, np.hypot(_A * _u[0], _B * _v[0])); YL = min(EXT, np.hypot(_A * _u[1], _B * _v[1]))
FIGW = 9.6
fig = plt.figure(figsize=(FIGW, FIGW * YL / XL), facecolor=PAGE)
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_facecolor(PAGE)
ax.set_xlim(-XL, XL); ax.set_ylim(-YL, YL); ax.set_aspect("equal"); ax.axis("off")
print(f"  page: {PAGE}; fade ellipse a = {args.fade_a:g} kpc, q = {args.fade_q:g}, PA {np.degrees(_pa_f):.0f}; view ±{XL:.1f} x ±{YL:.1f} kpc")
CORNERS = {int(round(np.degrees(np.arctan2(sy_, sx_)) % 360)): (sx_, sy_) for sx_, sy_ in ((XL, YL), (-XL, YL), (-XL, -YL), (XL, -YL))}

def corners_in(a0, a1, margin=8.0):
    """The frame corners (their angles from the centre) that fall inside the slice [a0, a1], at least `margin` deg off its seams, upper ones first."""
    out = []
    for c in sorted(CORNERS, key=lambda c: -CORNERS[c][1]):
        d = (c - a0) % 360.0
        if margin <= d <= (a1 - a0) % 360.0 - margin:
            out.append(c)
    return out
used = set()
patches = {}
for k, name in enumerate(order):
    poly = Polygon(wedge(bounds[k], bounds[k + 1]), closed=True, transform=ax.transData)
    patches[name] = poly
    im = ax.imshow(images[name], extent=extent, origin="lower", interpolation="bilinear", zorder=1)
    im.set_clip_path(poly)
    if name == "kin" and args.vel_contours > 0:
        cs = ax.contour(X, Y, np.where(np.isfinite(m1_msk), m1, np.nan), levels=np.arange(-vmax, vmax + 1, args.vel_contours), colors="k", linewidths=0.5, alpha=0.55, zorder=2)
        for c in cs.collections:
            c.set_clip_path(poly)
    if name == "sim":
        lw = 0.45 + 1.5 * np.clip(dust_frac, 0, 1) ** 0.4
        sp = ax.streamplot(XV, YV, VX, VY, density=args.stream_density, color=VLOS, cmap=cm_vel, norm=vnorm,
                           linewidth=lw, arrowsize=0.8, arrowstyle="->", minlength=0.15, maxlength=2.5, zorder=3)
        sp.lines.set_clip_path(poly); sp.lines.set_alpha(0.95)
for a in ax.patches:                                                     # the streamplot arrows: clipped to the SIMBA slice
    if a.__class__.__name__ == "FancyArrowPatch":
        a.set_clip_path(patches["sim"])
from matplotlib.collections import LineCollection
for k in range(3):                                                       # the seams: plain lines from the centre, white on the sky, black on the page
    a = np.radians(bounds[k]); rr_ = np.linspace(0, 1.5 * EXT, 300)
    pts = np.column_stack([rr_ * np.cos(a), rr_ * np.sin(a)])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    mid = 0.5 * (pts[:-1] + pts[1:])
    g = 1.0 - page_w(mid[:, 0], mid[:, 1]) if WHITE else np.ones(len(segs))
    ax.add_collection(LineCollection(segs, colors=[(v, v, v, 1.0) for v in g], linewidths=1.3, zorder=5, capstyle="butt"))
# the names, wrapped around the galaxy along a ring outside the fade ellipse (each letter set on the ellipse, upright)
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
_fp = FontProperties(family="DejaVu Sans", weight="bold", size=args.label_size)
_ttp = TextToPath()
KPC_PER_PT = 2 * XL / (FIGW * 72.0)
def curved_text(text, theta_c_deg, r_scale, color, fontsize):
    """`text` along the ellipse r_scale x the fade ellipse, centred at polar angle theta_c_deg; reads left to right when the
    picture is upright: clockwise (letters standing on the ring) on the top half, counter-clockwise (hanging) on the bottom half."""
    a_t, b_t = args.fade_a * r_scale, args.fade_a * args.fade_q * r_scale
    phi = np.linspace(0, 2 * np.pi, 7201)[:-1]
    P = np.outer(np.cos(phi), a_t * _u) + np.outer(np.sin(phi), b_t * _v)
    ang = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360.0
    o = np.argsort(ang); P, ang = P[o], ang[o]                            # counter-clockwise order
    ds = np.hypot(*(np.roll(P, -1, axis=0) - P).T); sarc = np.concatenate([[0.0], np.cumsum(ds)[:-1]]); L = float(ds.sum())
    top = 0.0 <= (theta_c_deg % 360.0) < 180.0
    sgn = -1.0 if top else 1.0                                            # reading direction along increasing arc length
    s_c = float(np.interp(theta_c_deg % 360.0, ang, sarc))
    widths = np.array([_ttp.get_text_width_height_descent(ch, _fp, ismath=False)[0] for ch in text]) * KPC_PER_PT * 1.05
    total = widths.sum(); s0 = s_c - sgn * 0.5 * total
    s_ext = np.concatenate([sarc - L, sarc, sarc + L]); P_ext = np.vstack([P, P, P])
    for ch, w, off in zip(text, widths, np.cumsum(widths) - 0.5 * widths):
        si = s0 + sgn * off
        x, y = np.interp(si, s_ext, P_ext[:, 0]), np.interp(si, s_ext, P_ext[:, 1])
        dx, dy = (np.interp(si + 0.05, s_ext, P_ext[:, 0]) - np.interp(si - 0.05, s_ext, P_ext[:, 0]),
                  np.interp(si + 0.05, s_ext, P_ext[:, 1]) - np.interp(si - 0.05, s_ext, P_ext[:, 1]))
        rot = np.degrees(np.arctan2(sgn * dy, sgn * dx))
        if ch.strip():
            ax.text(x, y, ch, color=color, fontsize=fontsize, fontweight="bold", family="DejaVu Sans", ha="center", va="center",
                    rotation=rot, rotation_mode="anchor", zorder=6)
_custom = {}
for tok in [t for t in args.label_angles.split(",") if t.strip()]:
    k_, v_ = tok.split(":"); _custom[k_.strip()] = float(v_)
for k, name in enumerate(order):
    m = 0.5 * (bounds[k] + bounds[k + 1]) % 360.0
    theta_c = _custom.get(name, m + 0.15 * ((90.0 if m < 180.0 else 270.0) - m))
    curved_text(NAMES[name], theta_c, R_TEXT, INK, args.label_size)
# the scale bar: on the JWST slice's sky, below the East end of the disc, white with a thin black edge
if args.scale_kpc > 0:
    pc = 0.70 * args.fade_a * _u - 0.60 * args.fade_a * args.fade_q * _v
    x0_, x1_ = pc[0] - 0.5 * args.scale_kpc, pc[0] + 0.5 * args.scale_kpc
    ax.plot([x0_, x1_], [pc[1], pc[1]], color="w", lw=2.5, zorder=6, solid_capstyle="butt", path_effects=[withStroke(linewidth=4, foreground="black")])
    ax.text(pc[0], pc[1] + 0.03 * YL, f"{args.scale_kpc:g} kpc", color="w", fontsize=11, ha="center", va="bottom", zorder=6,
            path_effects=[withStroke(linewidth=2.5, foreground="black")])
for ext_ in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"peel_collage_{src}{args.tag}.{ext_}"), dpi=args.dpi, facecolor=PAGE)
print(f"  written peel_collage_{src}{args.tag}.png / .pdf")

# ── the uncut layers, for the eye ───────────────────────────────────────────────────────────────────────────────
fig2, axs = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="black")
for a, name in zip(axs, ("obs", "kin", "sim")):
    a.set_facecolor("black"); a.imshow(images[name], extent=extent, origin="lower", interpolation="bilinear")
    if name == "kin":
        a.contour(X, Y, np.where(np.isfinite(m1_msk), m1, np.nan), levels=np.arange(-vmax, vmax + 1, 50.0), colors="k", linewidths=0.5, alpha=0.55)
    if name == "sim":
        a.streamplot(XV, YV, VX, VY, density=args.stream_density, color=VLOS, cmap=cm_vel, norm=vnorm, linewidth=0.8, arrowsize=0.7)
    a.set_xlim(-EXT, EXT); a.set_ylim(-EXT, EXT); a.set_xticks([]); a.set_yticks([])
    a.set_title(NAMES[name], color="w", fontsize=11)
    for s in a.spines.values():
        s.set_edgecolor("0.3")
fig2.savefig(os.path.join(args.out, f"peel_collage_{src}{args.tag}_layers.png"), dpi=130, facecolor="black", bbox_inches="tight")
print(f"  written peel_collage_{src}{args.tag}_layers.png")
