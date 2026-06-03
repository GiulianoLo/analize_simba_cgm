#!/usr/bin/env python3
"""
generate_capabilities_figure.py
Produces  simbanator_map.png  (300 dpi) and  simbanator_map.pdf
for use in a PhD thesis.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ─── fonts ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Liberation Sans', 'DejaVu Sans'],
    'pdf.fonttype'      : 42,
    'ps.fonttype'       : 42,
    'figure.facecolor'  : 'white',
})

# ─── canvas ──────────────────────────────────────────────────────────────────
W, H = 14.0, 18.5
fig  = plt.figure(figsize=(W, H), dpi=300, facecolor='white')
ax   = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')

# ─── palette ─────────────────────────────────────────────────────────────────
BG     = '#F4F6F8'
TITLE  = '#17202A'
INP    = '#1A3A5C'   # dark navy  – input data
IO     = '#1F618D'   # steel blue – I/O layer
SEL    = '#0B5345'   # dark teal  – selection
PROG   = '#922B21'   # dark red   – progenitors
HIST   = '#C75A0E'   # burnt orange – histories
QUE    = '#B07D0B'   # amber      – quenching
MERG   = '#6E2C00'   # deep brown – mergers
EXT    = '#5B2C6F'   # purple     – extraction
PROF   = '#154360'   # dark blue  – profiles
VIZ    = '#145A32'   # dark green – imaging
SED    = '#78281F'   # deep red   – SED
EPO    = '#641E16'   # darker red – epochs
FLX    = '#4A0E07'   # darkest    – fluxes
GEOM   = '#4D5656'   # slate gray – utilities

# group tints
TSEL   = '#D5F5E3'   # light green tint  (selection group)
TTEMP  = '#FEF9E7'   # light yellow tint (temporal group)
TSPEC  = '#FDEDEC'   # light red tint    (spectral group)
TVIS   = '#EAFAF1'   # light green tint  (viz group)
TSTR   = '#EAF2FF'   # light blue tint   (structural group)

ARRW   = '#717D7E'
WT     = 'white'

# ─── helpers ─────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec='none', lw=0, radius=0.22, alpha=1.0, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0',
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder, clip_on=False)
    ax.add_patch(p)


def t(x, y, s, sz=10, c=WT, w='normal', ha='center', va='center',
      z=8, a=1.0, italic=False):
    ax.text(x, y, s, fontsize=sz, color=c, fontweight=w,
            fontstyle='italic' if italic else 'normal',
            ha=ha, va=va, zorder=z, alpha=a, multialignment='center')


def arr(x1, y1, x2, y2, col=ARRW, lw=1.7, rad=0.0, hw=0.22, hl=0.18):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=f'-|>,head_width={hw},head_length={hl}',
            color=col, lw=lw,
            connectionstyle=f'arc3,rad={rad}',
        ),
        zorder=9,
    )

# ─── canvas background ───────────────────────────────────────────────────────
rbox(0, 0, W, H, BG, zorder=0)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1  title
# ═══════════════════════════════════════════════════════════════════════════
rbox(0, 17.3, W, 1.2, TITLE, zorder=1)
t(7, 18.05, 'simbanator', sz=26, w='bold')
t(7, 17.57, 'Full analysis pipeline for SIMBA cosmological hydrodynamical simulations',
  sz=10.5, c='#AED6F1')

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2  simulation input
# ═══════════════════════════════════════════════════════════════════════════
rbox(0.35, 15.9, 13.3, 1.15, INP, radius=0.28)
t(7, 16.61, 'SIMBA  Cosmological  Simulation', sz=13.5, w='bold')
t(3.9, 16.18, 'Particle snapshots: positions, masses, SFR,\nvelocities, metallicities at each cosmic epoch', sz=9.5, a=0.88)
t(10.1, 16.18, 'Group catalogues: halo/galaxy membership\nand integrated galaxy properties', sz=9.5, a=0.88)
# divider
ax.plot([7, 7], [16.03, 16.37], color=WT, alpha=0.25, lw=1, zorder=9)

arr(7.0, 15.9, 7.0, 15.57)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 3  I/O & config
# ═══════════════════════════════════════════════════════════════════════════
rbox(1.4, 14.6, 11.2, 0.9, IO, radius=0.25)
t(7, 15.15, 'Simulation I/O  &  Configuration', sz=11.5, w='bold')
t(7, 14.82, 'Reads simulation files  ·  Manages output directories  ·  Loads photometric filter curves',
  sz=9.5, a=0.88)

arr(4.2, 14.6, 3.1, 14.3)    # IO → Selection
arr(9.8, 14.6, 10.9, 14.3)   # IO → Progenitor

# ═══════════════════════════════════════════════════════════════════════════
# ROW 4  Selection  |  gap  |  Progenitor Tracking
# ═══════════════════════════════════════════════════════════════════════════

# ── 4a: Galaxy Selection ─────────────────────────────────────────────────
rbox(0.35, 12.1, 5.5, 2.0, SEL, radius=0.28)
t(3.1, 13.77, 'Galaxy Sample Selection', sz=11.5, w='bold')
t(3.1, 13.40, 'Filter galaxies by stellar mass, specific SFR,', sz=9.5, a=0.9)
t(3.1, 13.07, 'dust-to-stellar ratio, and environment', sz=9.5, a=0.9)
t(3.1, 12.74, '(field · satellite · cluster)', sz=9.5, a=0.88)
t(3.1, 12.41, 'at any redshift snapshot', sz=9.5, a=0.85)

# ── 4b: Progenitor Tracking ──────────────────────────────────────────────
rbox(8.15, 12.1, 5.5, 2.0, PROG, radius=0.28)
t(10.9, 13.77, 'Progenitor Tracking', sz=11.5, w='bold')
t(10.9, 13.40, 'Identify each galaxy\'s most-massive progenitor', sz=9.5, a=0.9)
t(10.9, 13.07, 'at every prior simulation output', sz=9.5, a=0.9)
t(10.9, 12.74, 'by walking the merger tree backward in time', sz=9.5, a=0.88)
t(10.9, 12.41, 'with periodic-boundary corrections', sz=9.5, a=0.85)

# arrows Row4 → Row5
arr(10.9, 12.1, 2.45, 11.73, rad=0.15)   # Progenitor → Histories (long)
arr(10.9, 12.1, 7.0,  11.73, rad=0.1)    # Progenitor → Quenching
arr(10.9, 12.1, 11.55, 11.73)             # Progenitor → Merger

# Selection → Extraction (curved around left margin, below history box)
arr(0.7, 12.1, 0.7, 8.88, rad=-0.18, col='#1A6B55', lw=2.0)   # curved left rail
ax.text(0.20, 10.5, 'target\nsample', fontsize=7.5, color='#1A6B55',
        ha='center', va='center', rotation=90, alpha=0.8, zorder=9,
        fontfamily='Liberation Sans', fontstyle='italic')

# ═══════════════════════════════════════════════════════════════════════════
# ROW 5  Analysis triptych
# ═══════════════════════════════════════════════════════════════════════════
ROW5_Y  = 9.6
ROW5_H  = 2.0
COL1_X  = 0.35
COL2_X  = 4.85
COL3_X  = 9.35
COL_W   = 4.15

# ── 5a: Property Histories ───────────────────────────────────────────────
rbox(COL1_X, ROW5_Y, COL_W, ROW5_H, HIST, radius=0.28)
t(2.43, 11.27, 'Property Histories', sz=10.5, w='bold')
t(2.43, 10.92, 'Reconstruct continuous time evolution of', sz=9.5, a=0.9)
t(2.43, 10.59, 'SFR  ·  M*  ·  M_mol  ·  M_dust  ·  Z', sz=9.5, a=0.9)
t(2.43, 10.26, 'along the progenitor chain;', sz=9.5, a=0.88)
t(2.43, 9.93, 'interpolated as smooth continuous tracks', sz=9.5, a=0.85)

# ── 5b: Quenching Analysis ───────────────────────────────────────────────
rbox(COL2_X, ROW5_Y, COL_W, ROW5_H, QUE, radius=0.28)
t(7.0, 11.27, 'Quenching Analysis', sz=10.5, w='bold')
t(7.0, 10.92, 'Pinpoint key evolutionary epochs from', sz=9.5, a=0.9)
t(7.0, 10.59, 'the interpolated SFR track:', sz=9.5, a=0.9)
t(7.0, 10.26, 'SFR peak  ·  quench onset  ·  quench end', sz=9.5, a=0.88)
t(7.0, 9.93,  'gas exhaustion  ·  rejuvenation', sz=9.5, a=0.85)

# ── 5c: Merger Detection ─────────────────────────────────────────────────
rbox(COL3_X, ROW5_Y, COL_W, ROW5_H, MERG, radius=0.28)
t(11.43, 11.27, 'Merger Detection', sz=10.5, w='bold')
t(11.43, 10.92, 'Find companion galaxies per snapshot;', sz=9.5, a=0.9)
t(11.43, 10.59, 'classify events by stellar-mass ratio:', sz=9.5, a=0.9)
t(11.43, 10.26, 'Major merger  (mass ratio ≥ 1:4)', sz=9.5, a=0.88)
t(11.43, 9.93,  'Minor merger  (1:4 – 1:10)', sz=9.5, a=0.85)

# arrows Row5 → Extraction
arr(2.43, ROW5_Y, 3.5, 8.88)     # Histories → Extraction
arr(7.0,  ROW5_Y, 7.0, 8.88)     # Quenching → Extraction
# Merger Detection is a terminal product — no downward arrow

# ═══════════════════════════════════════════════════════════════════════════
# ROW 6  Particle Extraction  (central junction)
# ═══════════════════════════════════════════════════════════════════════════
rbox(1.3, 7.65, 11.4, 1.1, EXT, radius=0.28)
t(7.0, 8.32, 'Particle  Extraction', sz=13, w='bold')
t(3.6, 7.93, 'Isolate per-galaxy HDF5 subsets', sz=9.5, a=0.88)
t(7.0, 7.93, 'Gas  ·  Stars  ·  Dust  ·  Dark matter', sz=9.5, a=0.88)
t(10.4, 7.93, 'Galaxy  ·  Halo  ·  Aperture modes', sz=9.5, a=0.88)

# arrows Extraction → products
arr(2.85, 7.65, 2.43, 7.25)     # → Profiles
arr(7.0,  7.65, 7.0,  7.25)     # → Imaging
arr(11.15, 7.65, 11.43, 7.25)   # → SED

# ═══════════════════════════════════════════════════════════════════════════
# ROW 7  Output triptych
# ═══════════════════════════════════════════════════════════════════════════
ROW7_Y = 4.55
ROW7_H = 2.6

# ── 7a: Radial Profiles ──────────────────────────────────────────────────
rbox(COL1_X, ROW7_Y, COL_W, ROW7_H, PROF, radius=0.28)
t(2.43, 6.82, 'Radial Profiles', sz=10.5, w='bold')
t(2.43, 6.47, 'Surface-density and mean-property', sz=9.5, a=0.9)
t(2.43, 6.14, 'radial profiles for gas, stars, and dust', sz=9.5, a=0.9)
t(2.43, 5.81, 'Galaxy aligned to face-on orientation', sz=9.5, a=0.88)
t(2.43, 5.48, 'via iterative shrinking-sphere centering', sz=9.5, a=0.88)
t(2.43, 5.15, 'and principal-axis decomposition', sz=9.5, a=0.85)

# ── 7b: Imaging & Animation ──────────────────────────────────────────────
rbox(COL2_X, ROW7_Y, COL_W, ROW7_H, VIZ, radius=0.28)
t(7.0, 6.82, 'Imaging  &  Animation', sz=10.5, w='bold')
t(7.0, 6.47, 'SPH-smoothed multi-component images', sz=9.5, a=0.9)
t(7.0, 6.14, 'RGB blend: gas  ·  stars  ·  dust', sz=9.5, a=0.9)
t(7.0, 5.81, 'Multi-scale zoom  (Mpc → kpc)', sz=9.5, a=0.88)
t(7.0, 5.48, '3-projection views  (xy · xz · yz)', sz=9.5, a=0.88)
t(7.0, 5.15, 'Time-lapse GIF animations', sz=9.5, a=0.85)

# ── 7c: SED Modeling ─────────────────────────────────────────────────────
rbox(COL3_X, ROW7_Y, COL_W, ROW7_H, SED, radius=0.28)
t(11.43, 6.82, 'SED Modeling', sz=10.5, w='bold')
t(11.43, 6.47, '3D Monte-Carlo dust radiative transfer', sz=9.5, a=0.9)
t(11.43, 6.14, 'coupled with stellar population synthesis', sz=9.5, a=0.9)
t(11.43, 5.81, 'Full SED with and without dust attenuation', sz=9.5, a=0.88)
t(11.43, 5.48, 'Face-on  ·  edge-on inclinations', sz=9.5, a=0.88)
t(11.43, 5.15, 'Submitted as cluster batch jobs', sz=9.5, a=0.85)

# arrows SED → epoch + flux
arr(10.43, ROW7_Y, 10.2, 4.22)
arr(12.43, ROW7_Y, 12.66, 4.22)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 8  sub-products under SED
# ═══════════════════════════════════════════════════════════════════════════
ROW8_Y = 1.95
ROW8_H = 2.1

# ── 8a: Critical Epoch SEDs ──────────────────────────────────────────────
rbox(COL3_X, ROW8_Y, 2.0, ROW8_H, EPO, radius=0.24)
t(10.35, 3.72, 'Critical\nEpoch  SEDs', sz=9.5, w='bold')
t(10.35, 3.18, 'SEDs computed at each', sz=8.8, a=0.9)
t(10.35, 2.88, 'key evolutionary epoch:', sz=8.8, a=0.9)
t(10.35, 2.58, 'SFR peak  ·  quench start', sz=8.5, a=0.88)
t(10.35, 2.28, 'quench end  ·  gas depletion', sz=8.5, a=0.85)

# ── 8b: Photometric Flux Extraction ──────────────────────────────────────
rbox(COL3_X + 2.15, ROW8_Y, 2.0, ROW8_H, FLX, radius=0.24)
t(12.50, 3.72, 'Photometric\nFlux  Extraction', sz=9.5, w='bold')
t(12.50, 3.18, 'Convolve SEDs with filter curves', sz=8.5, a=0.9)
t(12.50, 2.88, 'GALEX · HST · JWST', sz=8.5, a=0.9)
t(12.50, 2.58, 'SDSS · 2MASS · WISE', sz=8.5, a=0.88)
t(12.50, 2.28, 'Spitzer · Herschel · JCMT', sz=8.5, a=0.85)

# ═══════════════════════════════════════════════════════════════════════════
# Decorative group labels (subtle background panels)
# ═══════════════════════════════════════════════════════════════════════════

# "Temporal Analysis" label spanning rows 4–5 (right side)
rbox(7.95, 9.5, 5.8, 4.95, TTEMP, alpha=0.18, zorder=1)
ax.text(13.55, 11.98, 'TEMPORAL\nANALYSIS', fontsize=7.5,
        color='#7D6608', ha='right', va='center', rotation=90,
        fontweight='bold', alpha=0.55, zorder=1,
        fontfamily='Liberation Sans')

# "Sample Selection" label (left column rows 4)
rbox(0.15, 12.0, 5.9, 2.3, TSEL, alpha=0.18, zorder=1)
ax.text(0.25, 13.15, 'SELECTION', fontsize=7.5,
        color='#1A5276', ha='left', va='center', rotation=90,
        fontweight='bold', alpha=0.55, zorder=1,
        fontfamily='Liberation Sans')

# "Structural Analysis" panel (row 7 left)
rbox(0.15, 4.4, 4.5, 3.5, TSTR, alpha=0.18, zorder=1)
ax.text(0.25, 6.1, 'STRUCTURAL', fontsize=7.0,
        color='#1A5276', ha='left', va='center', rotation=90,
        fontweight='bold', alpha=0.55, zorder=1,
        fontfamily='Liberation Sans')

# "Spectral Modeling" panel (rows 7–8 right)
rbox(9.15, 1.8, 4.6, 5.6, TSPEC, alpha=0.18, zorder=1)
ax.text(13.55, 4.6, 'SPECTRAL\nMODELING', fontsize=7.0,
        color='#922B21', ha='right', va='center', rotation=90,
        fontweight='bold', alpha=0.55, zorder=1,
        fontfamily='Liberation Sans')

# "Visualization" panel (row 7 center)
rbox(4.65, 4.4, 4.6, 3.5, TVIS, alpha=0.18, zorder=1)
ax.text(5.0, 6.1, 'VISUAL.', fontsize=7.0,
        color='#145A32', ha='left', va='center', rotation=90,
        fontweight='bold', alpha=0.55, zorder=1,
        fontfamily='Liberation Sans')

# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════
ax.text(7, 0.35, 'v 0.2.0  ·  MIT License  ·  github.com/GiulianoLo/analize_simba_cgm',
        fontsize=8, color='#AAB7B8', ha='center', va='center', zorder=9,
        fontfamily='Liberation Sans')

# ─── save ────────────────────────────────────────────────────────────────────
for ext in ('png', 'pdf'):
    out = f'simbanator_map.{ext}'
    fig.savefig(out, dpi=300 if ext == 'png' else None,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved  {out}')
