#!/usr/bin/env python
"""SLURM array worker — write LEAN reduced particle files for flexible Σ / profile work.

Motivation
----------
The fixed-binning products from ``build_profiles_job.py`` bake in one radial grid and select gas
by the CAESAR *galaxy member list* (``gal.glist``, ISM only). This worker instead stores the raw
ingredients so profiles and surface densities can be re-binned at will in the notebook, and selects
particles by a **100 kpc physical aperture around the galaxy** — which captures ISM *and* CGM, not
just the listed galaxy particles.

Loading every gas particle of an m100n1024 snapshot is infeasible (~10^9 particles), so the
candidate set is the **parent halo** gas+star lists (``halo.glist`` / ``halo.slist`` — these already
include the diffuse CGM that ``gal.glist`` omits), spatially cut to ``REDUCED_RMAX`` physical kpc of
the galaxy centre with periodic minimum-image wrapping. For a central galaxy the 100 kpc sphere lies
well inside its FOF halo, so the halo candidate set loses nothing relative to a global spatial query.

For each unique (snapshot, galaxy) in the shared plan (the SAME dust_profile_plan_<tag>.hdf5 used by
build_profiles_job.py) it writes ONE lean HDF5 per galaxy:

  output/<sim>/reduced_particles/snap_NNN/<prefix>_snap<NNN>_gal<GX>.h5
    attrs : sim_name, snap, gx, redshift, a, hub, rmax_kpc,
            center_kpc(3), evecs(3,3)              # stellar principal frame, for face-on projection
    gas/  : idx(n) [GLOBAL snapshot index], pos(n,3) [kpc, RELATIVE to centre],
            m_gas, m_dust, m_HI, m_H2, sfr, temp [Msun, Msun/yr, K], member(bool),
            vel(n,3) [km/s peculiar = snapshot Velocities x sqrt(a); NaN if absent]
    star/ : idx(m), pos(m,3) [kpc, relative], m_star [Msun], member(bool),
            tform [scale factor a_form = PartType4/StellarFormationTime; NaN if absent],
            vel(m,3) [km/s peculiar, as for the gas]

The `member` boolean marks each particle as belonging to the CAESAR galaxy member list
(``gal.glist`` / ``gal.slist``). Member particles are ALWAYS kept (even if just outside the aperture),
so R20/R80 computed over ``member==True`` uses the SAME particles CAESAR does — a like-for-like check
that our extraction/centring reproduce the catalog radii (the full-aperture set adds the CGM).

Extensible schema (add fields WITHOUT rewriting files)
------------------------------------------------------
Extraction is split into two passes:

  1. **Geometry pass** (needs the snapshot): stellar frame + aperture selection -> the kept particles'
     GLOBAL snapshot indices ``idx`` and relative ``pos``. This is the expensive part and is done
     ONCE per galaxy.
  2. **Field producers** (``GAS_PRODUCERS`` / ``STAR_PRODUCERS``): each per-particle field is computed
     from the stored ``idx`` — reading only what that field needs (a snapshot column, or just the
     catalog for ``member``). Fields are written **additively** (HDF5 append mode), so a NEW field is
     backfilled into existing files by computing only that field at the stored ``idx`` — no geometry
     redo, no rewrite of the datasets already there.

To add a field later: add a producer to the registry (with its dependency tag) and add its name to
``GAS_FIELDS`` / ``STAR_FIELDS``; re-run the job. Files missing it get *only that dataset* appended
(catalog-only fields don't even open the snapshot). History of such additions: ``tform`` (2026-08-27),
``vel`` (2026-08-28) — every reduced file built before them (any plan, cis25 and cis100 alike) is
"partial" and is completed the next time ITS plan is submitted. Files missing ``idx``/``pos`` (or absent/corrupt)
fall back to a full geometry rebuild. ``REDUCED_OVERWRITE=1`` forces a full rebuild of everything.
Files are stamped with the HI/H2 split they were built with (attr ``h2_recipe`` = build_profiles_job
``H2_RECIPE``); files carrying an older/absent stamp get ``m_HI``/``m_H2`` recomputed at the stored
``idx`` on the next run (same backfill path, no geometry redo) — so re-running the same sbatch command
after a recipe change refreshes every file (2026-08-27: the old m_H*nh*fH2 split lost up to 97% of the
H2 in SIMBA's star-forming gas; now the caesar split, see build_profiles_job._components).

Files are keyed globally by (snapshot, galaxy id), so running this per anchor is naturally
idempotent: a galaxy already complete for one anchor is skipped (before any load) when it recurs in
another anchor's plan.

Performance
-----------
All snapshot I/O is batched per snapshot: a catalog pass collects every planned galaxy's candidate
indices (halo lists cached across galaxies sharing a halo), then each needed snapshot column is
slab-streamed ONCE at the sorted union of all candidates (sequential reads at disk bandwidth; slabs
holding no wanted row are skipped) and galaxies are served by in-memory slicing. This replaces the
per-galaxy-per-field h5py fancy reads that dominated the old runtime. Datasets are lzf-compressed
(much faster than gzip to write AND to read back in the notebook cache build; h5py reads both
transparently, so old gzip files coexist).

Env: DUST_PLAN (plan, shared with build_profiles_job; REQUIRED — the plan carries the sim name),
     REDUCED_RMAX_KPC (default 100), REDUCED_PREFIX (default: derived from the plan's sim
     file_format, e.g. m100n1024 / m50n512), REDUCED_OVERWRITE (default 0),
     REDUCED_GATHER_MB (slab budget per streamed read, default 256).
"""
import os
import gc
import numpy as np
import h5py

from simbanator.io.simba import Simulation
from simbanator.utils.geometry import shrink_center, principal_axes
# reuse the EXACT unit/field recipes the profile job is validated against
from build_profiles_job import (header_units, _to_kpc, _to_msun, _detect, _components, _halo_of,
                                _temperature, _XH, _nH, H2_RECIPE)

PLAN_PATH = os.environ.get("DUST_PLAN")                     # REQUIRED — the plan carries the sim
RMAX = float(os.environ.get("REDUCED_RMAX_KPC", 100.0))     # aperture [physical kpc]
PREFIX = os.environ.get("REDUCED_PREFIX") or None           # None -> derived from the plan's sim in main()
OVERWRITE = int(os.environ.get("REDUCED_OVERWRITE", 0)) == 1


# ── reduced-file schema ──────────────────────────────────────────────────────
# Written by the geometry pass (never backfillable — defines the particle set):
GEOM_DS = ("idx", "pos")
# Per-particle fields produced from `idx` (order here = write order in a fresh file):
GAS_FIELDS = ("m_gas", "m_dust", "m_HI", "m_H2", "sfr", "temp", "member", "vel")
STAR_FIELDS = ("m_star", "member", "tform", "vel")
_DEP_SNAP, _DEP_CAT = "snapshot", "catalog"       # what a producer needs to run


def _box_kpc(f, a, hub):
    bs = float(f["Header"].attrs.get("BoxSize", 0.0))
    return _to_kpc(bs, a, hub) if bs > 0 else None


def _min_image(d, L):
    """Periodic minimum-image displacement (kpc); L None -> no wrapping."""
    return d - L * np.round(d / L) if L else d


def _frame(spos, smass, gpos):
    """Stellar principal frame (centre, eigenvectors); fall back to gas median."""
    if len(spos) >= 10 and np.sum(smass) > 0:
        center = shrink_center(spos, masses=smass)
        _, _, evecs, _ = principal_axes(spos - center, masses=smass)
    elif len(gpos):
        center = np.median(gpos, axis=0)
        evecs = np.eye(3)
    else:
        return None, None
    return center, evecs


# ── snapshot column store + field producers ───────────────────────────────────────────────────────
# h5py fancy indexing with >~1e5 scattered indices on a compressed 1e9-row dataset is pathologically
# slow (each galaxy triggers a huge hyperslab selection and re-decompresses chunks its neighbours
# already touched). Instead, every column is gathered ONCE per snapshot at the sorted union of ALL
# candidate indices via sequential slab streaming (disk-bandwidth reads; slabs holding no wanted row
# are skipped entirely), and each galaxy is then served by in-memory slicing.
GATHER_MB = float(os.environ.get("REDUCED_GATHER_MB", 256))  # slab budget per read [MB]


def _gather(dset, uidx):
    """dset[uidx] (uidx sorted unique int64) via sequential slab streaming."""
    rowbytes = max(int(np.prod(dset.shape[1:], dtype=int)), 1) * dset.dtype.itemsize
    slab = max(int(GATHER_MB * 1e6 // rowbytes), 1_000_000)
    out = np.empty((len(uidx),) + dset.shape[1:], dset.dtype)
    n, j0 = dset.shape[0], 0
    while j0 < len(uidx):
        start = (int(uidx[j0]) // slab) * slab                 # jump straight to the next needed slab
        stop = min(start + slab, n)
        j1 = np.searchsorted(uidx, stop, side="left")
        block = dset[start:stop]
        out[j0:j1] = block[uidx[j0:j1] - start]
        del block
        j0 = j1
    return out


class _Ctx:
    """Per-snapshot context: lazy union-gathered snapshot columns + per-galaxy catalog object + units.
    `take(part, name, idx)` returns column values at global indices idx (idx ⊆ the part's union);
    the column is streamed from disk on first use and cached for every later galaxy."""
    __slots__ = ("_src", "_u", "_cache", "gal", "a", "hub", "fld")

    def __init__(self, f, u_gas, u_star, a, hub, fld):
        self._src = {"gas": (f["PartType0"] if f is not None else None),
                     "star": (f["PartType4"] if (f is not None and "PartType4" in f) else None)}
        self._u = {"gas": u_gas, "star": u_star}
        self._cache = {}
        self.gal, self.a, self.hub, self.fld = None, a, hub, fld

    def has(self, part):
        return self._src[part] is not None

    def has_field(self, part, name):
        src = self._src[part]
        return src is not None and name in src

    def take(self, part, name, idx):
        key = (part, name)
        if key not in self._cache:
            self._cache[key] = _gather(self._src[part][name], self._u[part])
        return self._cache[key][np.searchsorted(self._u[part], idx)]


def _gas_components(ctx, idx):
    """m_gas + the dust/HI/H2 split (caesar recipe, build_profiles_job._components: H2 = 0.76*m*fH2 with
    the n_H >= 0.13 cut, HI = 0.76*m*min(nh, 1-fH2)), all from one read."""
    fld, hub = ctx.fld, ctx.hub
    mgas = _to_msun(ctx.take("gas", "Masses", idx), hub)
    m_dust, m_HI, m_H2 = _components(
        mgas,
        _to_msun(ctx.take("gas", fld["dust"], idx), hub) if fld["dust"] else None,
        None,                                                 # Z unused by the caesar split (skip the 11-col stream)
        ctx.take("gas", fld["fneut"], idx) if fld["fneut"] else None,
        ctx.take("gas", fld["fmol"], idx) if fld["fmol"] else None,
        _nH(ctx.take("gas", fld["rho"], idx), ctx.a, hub) if fld.get("rho") else None)
    return {"m_gas": mgas.astype(np.float32), "m_dust": np.asarray(m_dust, np.float32),
            "m_HI": np.asarray(m_HI, np.float32), "m_H2": np.asarray(m_H2, np.float32)}


def _gas_sfr(ctx, idx):
    fld = ctx.fld
    sfr = (np.asarray(ctx.take("gas", fld["sfr"], idx), np.float32) if fld["sfr"]
           else np.zeros(len(idx), np.float32))
    return {"sfr": sfr}


def _gas_temp(ctx, idx):
    """Per-particle temperature [K] — same recipe as build_profiles_job._temperature."""
    fld = ctx.fld
    Zc = ctx.take("gas", fld["Z"], idx) if fld["Z"] else None
    if fld["Tdir"]:
        T = np.asarray(ctx.take("gas", fld["Tdir"], idx), np.float64)
    elif fld["u"] and fld["ne"]:
        T = _temperature(np.asarray(ctx.take("gas", fld["u"], idx), np.float64),
                         np.asarray(ctx.take("gas", fld["ne"], idx), np.float64), _XH(Zc, len(idx)))
    else:
        T = np.full(len(idx), np.nan)
    return {"temp": T.astype(np.float32)}


def _gas_member(ctx, idx):
    gml = np.unique(np.asarray(ctx.gal.glist, dtype=np.int64))
    return {"member": np.isin(idx, gml)}


def _star_mass(ctx, idx):
    return {"m_star": _to_msun(ctx.take("star", "Masses", idx), ctx.hub).astype(np.float32)}


def _star_member(ctx, idx):
    gsl = np.unique(np.asarray(getattr(ctx.gal, "slist", []), dtype=np.int64))
    return {"member": np.isin(idx, gsl)}


def _star_tform(ctx, idx):
    """Formation epoch a_form = PartType4/StellarFormationTime (scale factor in cosmological
    GIZMO/SIMBA runs; the consumer converts it to cosmic time). NaN if the snapshot lacks it.
    Added 2026-08-27 for the KS-track notebook (archaeological SFR windows from slist members)."""
    if not ctx.has_field("star", "StellarFormationTime"):
        return {"tform": np.full(len(idx), np.nan, np.float32)}
    return {"tform": np.asarray(ctx.take("star", "StellarFormationTime", idx), np.float32)}


def _velocity(ctx, part, idx):
    """Peculiar velocity [km/s] (n, 3) = snapshot ``Velocities`` x sqrt(a): GIZMO cosmological runs store
    v_pec / sqrt(a) in km/s (yt/caesar apply the same factor). NaN if the snapshot lacks the column.
    Added 2026-08-28 for the KS-track notebook (kappa_rot of the H2 / stars in the core and outskirt at
    every critical epoch, ks_tracks_lib.measure_zone_kinematics); kappa_rot itself is unit-free."""
    if not ctx.has_field(part, "Velocities"):
        return {"vel": np.full((len(idx), 3), np.nan, np.float32)}
    v = np.asarray(ctx.take(part, "Velocities", idx), np.float64) * np.sqrt(float(ctx.a))
    return {"vel": v.astype(np.float32)}


def _gas_vel(ctx, idx):
    return _velocity(ctx, "gas", idx)


def _star_vel(ctx, idx):
    return _velocity(ctx, "star", idx)


GAS_PRODUCERS = [
    (("m_gas", "m_dust", "m_HI", "m_H2"), _DEP_SNAP, _gas_components),
    (("sfr",),    _DEP_SNAP, _gas_sfr),
    (("temp",),   _DEP_SNAP, _gas_temp),
    (("member",), _DEP_CAT,  _gas_member),
    (("vel",),    _DEP_SNAP, _gas_vel),
]
STAR_PRODUCERS = [
    (("m_star",), _DEP_SNAP, _star_mass),
    (("member",), _DEP_CAT,  _star_member),
    (("tform",),  _DEP_SNAP, _star_tform),
    (("vel",),    _DEP_SNAP, _star_vel),
]
_PRODUCERS = {"gas": GAS_PRODUCERS, "star": STAR_PRODUCERS}


def _produce_into(rec, ctx, want):
    """Fill rec[grp][field] for each requested field, from rec[grp]['idx']. `want` = {grp: set}."""
    for grp, producers in _PRODUCERS.items():
        idx = rec[grp].get("idx")
        w = want.get(grp, set())
        if idx is None or not len(idx) or not w:
            continue
        for names, _dep, fn in producers:
            if w & set(names):
                for k, v in fn(ctx, idx).items():
                    if k in w:
                        rec[grp][k] = v


def _needs_snapshot(miss):
    """True if any missing field needs a snapshot read (catalog-only fields don't)."""
    for grp, producers in _PRODUCERS.items():
        w = miss.get(grp, set())
        for names, dep, _fn in producers:
            if dep == _DEP_SNAP and (w & set(names)):
                return True
    return False


# ── catalog pass + geometry ──────────────────────────────────────────────────
def _catalog_pass(cs, gxs):
    """Per-galaxy member + candidate index lists, collected BEFORE any snapshot read so the snapshot
    columns can be gathered once at the union. The candidate set is the parent halo gas+stars (incl.
    CGM) UNION the galaxy members; halo lists are cached, so several galaxies sharing one halo read
    its (possibly huge) lists once."""
    plans, halo_cache = {}, {}
    for gx in gxs:
        try:
            gal = cs.galaxies[gx]
        except (IndexError, KeyError):
            continue
        gsl = np.unique(np.asarray(getattr(gal, "slist", []), dtype=np.int64))
        gml = np.unique(np.asarray(gal.glist, dtype=np.int64))
        halo = _halo_of(gal, cs)
        hkey = int(getattr(halo, "GroupID", -1)) if halo is not None else None
        if hkey not in halo_cache:
            halo_cache[hkey] = (
                (np.unique(np.asarray(getattr(halo, "glist", []), dtype=np.int64)),
                 np.unique(np.asarray(getattr(halo, "slist", []), dtype=np.int64)))
                if halo is not None else (None, None))
        hg, hs = halo_cache[hkey]
        plans[int(gx)] = dict(gal=gal, gsl=gsl, gml=gml,
                              cand_g=np.union1d(hg, gml) if hg is not None else gml,
                              cand_s=np.union1d(hs, gsl) if hs is not None else gsl)
    return plans


def _extract_full(ctx, plan, L):
    """Full record for a fresh galaxy file: stellar frame + aperture selection + all fields, served
    entirely from the ctx column store. idx stays GLOBAL & ascending (union1d + boolean masks)."""
    gal, gsl, gml = plan["gal"], plan["gsl"], plan["gml"]
    cand_g, cand_s = plan["cand_g"], plan["cand_s"]
    ctx.gal = gal
    a, hub = ctx.a, ctx.hub

    # stellar principal frame from the galaxy's OWN member stars
    if ctx.has("star") and len(gsl):
        fpos = _to_kpc(ctx.take("star", "Coordinates", gsl), a, hub)
        fmass = _to_msun(ctx.take("star", "Masses", gsl), hub)
    else:
        fpos, fmass = np.empty((0, 3)), np.empty(0)
    gmpos = _to_kpc(ctx.take("gas", "Coordinates", gml), a, hub) if len(gml) else np.empty((0, 3))
    center, evecs = _frame(fpos, fmass, gmpos)
    if center is None:
        return None

    rec = dict(gx=int(plan["gx"]), z=float(1.0 / a - 1.0), a=float(a), hub=float(hub),
               center=np.asarray(center, float), evecs=np.asarray(evecs, float),
               gas={}, star={})
    # aperture cut; member particles are ALWAYS kept (aperture OR member) so the member subset ==
    # CAESAR's galaxy list.
    if len(cand_g):
        d = _min_image(_to_kpc(ctx.take("gas", "Coordinates", cand_g), a, hub) - center, L)
        keep = (np.sum(d * d, axis=1) < RMAX * RMAX) | np.isin(cand_g, gml)
        if np.any(keep):
            rec["gas"]["idx"] = cand_g[keep].astype(np.int64)
            rec["gas"]["pos"] = d[keep].astype(np.float32)
    if ctx.has("star") and len(cand_s):
        d = _min_image(_to_kpc(ctx.take("star", "Coordinates", cand_s), a, hub) - center, L)
        keep = (np.sum(d * d, axis=1) < RMAX * RMAX) | np.isin(cand_s, gsl)
        if np.any(keep):
            rec["star"]["idx"] = cand_s[keep].astype(np.int64)
            rec["star"]["pos"] = d[keep].astype(np.float32)
    _produce_into(rec, ctx, {"gas": set(GAS_FIELDS), "star": set(STAR_FIELDS)})
    return rec


# ── I/O ──────────────────────────────────────────────────────────────────────
def _outname(snap, gx):
    """Per-galaxy output filename (snap, gx) — the global dedup key shared across anchors."""
    return f"{PREFIX}_snap{int(snap):03d}_gal{int(gx):06d}.h5"


def _h2_recipe_of(f):
    v = f.attrs.get("h2_recipe", "")
    return v.decode() if isinstance(v, bytes) else str(v)


def _missing_fields(path):
    """For an existing file, {'gas': set_missing, 'star': set_missing} of current-schema fields that
    can be BACKFILLED (idx/pos present). Returns None if the file needs a full geometry rebuild
    (absent, corrupt, or missing idx/pos). An empty group (no particles) contributes an empty set.
    A non-empty gas group whose ``h2_recipe`` stamp is not the current H2_RECIPE has m_HI/m_H2 marked
    missing so they are recomputed (stale split)."""
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            res = {}
            for grp, fields in (("gas", GAS_FIELDS), ("star", STAR_FIELDS)):
                if grp not in f:
                    return None                                   # unexpected layout -> rebuild
                keys = set(f[grp].keys())
                if not keys:
                    res[grp] = set()                              # genuinely empty group -> nothing to add
                    continue
                if not set(GEOM_DS).issubset(keys):               # no idx/pos -> cannot backfill
                    return None
                res[grp] = set(fields) - keys
                if grp == "gas" and _h2_recipe_of(f) != H2_RECIPE:
                    res[grp] |= {"m_HI", "m_H2"} & set(fields)
            return res
    except OSError:
        return None


def _write_full(rec, snap, out_dir):
    """Write a fresh file (mode 'w'): attrs + geometry + all fields."""
    fpath = os.path.join(out_dir, _outname(snap, rec["gx"]))
    with h5py.File(fpath, "w") as o:
        o.attrs["sim_name"] = PREFIX
        o.attrs["snap"] = int(snap)
        o.attrs["gx"] = int(rec["gx"])
        o.attrs["redshift"] = rec["z"]
        o.attrs["a"] = rec["a"]
        o.attrs["hub"] = rec["hub"]
        o.attrs["rmax_kpc"] = RMAX
        o.attrs["h2_recipe"] = H2_RECIPE
        o.attrs["center_kpc"] = rec["center"]
        o.attrs["evecs"] = rec["evecs"]
        for grp in ("gas", "star"):
            gg = o.require_group(grp)
            for k, v in rec[grp].items():
                gg.create_dataset(k, data=v, compression="lzf")   # lzf: ~5-10x faster than gzip
    return fpath


def _append_fields(path, add):
    """Append ONLY the datasets in `add` = {'gas': {k: v}, 'star': {k: v}} to an existing file,
    leaving every other dataset intact (HDF5 append mode — no rewrite of the rest)."""
    with h5py.File(path, "a") as o:
        for grp in ("gas", "star"):
            for k, v in add.get(grp, {}).items():
                if k in o[grp]:
                    del o[grp][k]                                 # replace a stray/partial/stale one
                o[grp].create_dataset(k, data=v, compression="lzf")
        if {"m_HI", "m_H2"} <= set(add.get("gas", {})):
            o.attrs["h2_recipe"] = H2_RECIPE                     # HI/H2 now on the current split


def _stored_idx(path, miss):
    """{'gas': idx|None, 'star': idx|None} stored in an existing file, for the groups being backfilled."""
    out = {"gas": None, "star": None}
    with h5py.File(path, "r") as o:
        for grp in ("gas", "star"):
            if miss.get(grp) and grp in o and "idx" in o[grp]:
                out[grp] = np.asarray(o[grp]["idx"][:], np.int64)
    return out


def _backfill_one(path, gx, miss, idx_of, cs, ctx):
    """Compute only the missing fields from the file's stored idx (already read into idx_of) and
    append them. Returns True if anything was written."""
    try:
        ctx.gal = cs.galaxies[gx]
    except (IndexError, KeyError):
        return False
    rec = dict(gx=int(gx), gas={}, star={})
    for grp in ("gas", "star"):
        if idx_of.get(grp) is not None:
            rec[grp]["idx"] = idx_of[grp]
    _produce_into(rec, ctx, miss)
    add = {grp: {k: v for k, v in rec[grp].items() if k in miss.get(grp, set())}
           for grp in ("gas", "star")}
    if not add["gas"] and not add["star"]:
        return False
    _append_fields(path, add)
    return True


# ── driver ───────────────────────────────────────────────────────────────────
def _plan_gxs(gx, snap, sn):
    return np.unique(gx[snap == sn]).astype(np.int64).tolist()


def _union(arrays):
    arrays = [a for a in arrays if a is not None and len(a)]
    return (np.unique(np.concatenate(arrays)) if arrays else np.empty(0, np.int64))


def process_snapshot(sim, snap, gxs, out_dir):
    """Full + incremental processing of `gxs` in `snap`. Returns (n_full, n_backfill, n_skipped).
    All snapshot I/O is batched: the catalog pass collects every galaxy's candidate indices first,
    then each needed column is slab-streamed ONCE at the union and galaxies are served from memory."""
    todo_full, todo_part, n_skip = [], {}, 0
    for gx in gxs:
        path = os.path.join(out_dir, _outname(snap, gx))
        if OVERWRITE:
            todo_full.append(int(gx)); continue
        miss = _missing_fields(path)
        if miss is None:
            todo_full.append(int(gx))
        elif miss["gas"] or miss["star"]:
            todo_part[int(gx)] = miss
        else:
            n_skip += 1
    if not todo_full and not todo_part:
        return 0, 0, n_skip

    need_snap = bool(todo_full) or any(_needs_snapshot(m) for m in todo_part.values())
    cs = sim.load_catalog(snap=snap)                              # catalog is cheap; needed either way
    n_full = n_part = 0
    try:
        plans = _catalog_pass(cs, todo_full)
        for gx, p in plans.items():
            p["gx"] = gx
        bf_idx = {gx: _stored_idx(os.path.join(out_dir, _outname(snap, gx)), miss)
                  for gx, miss in todo_part.items()}
        if need_snap:
            with h5py.File(sim.get_snapshot_file(snap), "r") as f:
                a, hub = header_units(f)
                L = _box_kpc(f, a, hub)
                u_gas = _union([p["cand_g"] for p in plans.values()]
                               + [ix["gas"] for ix in bf_idx.values()])
                u_star = _union([p["cand_s"] for p in plans.values()]
                                + [ix["star"] for ix in bf_idx.values()])
                ctx = _Ctx(f, u_gas, u_star, a, hub, _detect(f))
                for gx, p in plans.items():
                    rec = _extract_full(ctx, p, L)
                    if rec is not None:
                        _write_full(rec, snap, out_dir); n_full += 1
                for gx, miss in todo_part.items():
                    if _backfill_one(os.path.join(out_dir, _outname(snap, gx)), gx, miss,
                                     bf_idx[gx], cs, ctx):
                        n_part += 1
        else:                                                    # only catalog-derived fields missing
            ctx = _Ctx(None, None, None, None, None, None)
            for gx, miss in todo_part.items():
                if _backfill_one(os.path.join(out_dir, _outname(snap, gx)), gx, miss,
                                 bf_idx[gx], cs, ctx):
                    n_part += 1
    finally:
        del cs
        gc.collect()
    return n_full, n_part, n_skip


def main():
    global PREFIX
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_task = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    if not PLAN_PATH:
        raise SystemExit("DUST_PLAN is not set — run  sbatch submit_reduced_particles.sh "
                         "output/<sim>/caesar_sfh/prof_<tag>/dust_profile_plan_<tag>.hdf5")
    with h5py.File(PLAN_PATH, "r") as f:
        sim_name = str(f.attrs["sim_name"])
        gx = f["entry_gx"][:]
        snap = f["entry_snap"][:]

    sim = Simulation(sim_name)
    if PREFIX is None:
        PREFIX = sim.file_format.split("_{")[0]    # matches the notebook loader's REDUCED_PREFIX
    out_root = os.path.join(os.getcwd(), "output", sim_name, "reduced_particles")

    snaps_all = np.sort(np.unique(snap))[::-1]                    # newest first
    my_snaps = np.array_split(snaps_all, n_task)[task_id] if n_task > 1 else snaps_all
    print(f"[task {task_id}/{n_task}] {len(my_snaps)} snapshots; aperture={RMAX:g} kpc "
          f"prefix='{PREFIX}': {list(map(int, my_snaps))}", flush=True)

    n_full = n_part = n_skip = 0
    for k, sn in enumerate(my_snaps):
        sn = int(sn)
        gxs = _plan_gxs(gx, snap, sn)                             # one file per (snap, galaxy)
        out_dir = os.path.join(out_root, f"snap_{sn:03d}")
        os.makedirs(out_dir, exist_ok=True)
        try:
            nf, np_, ns = process_snapshot(sim, sn, gxs, out_dir)
        except OSError as e:
            print(f"  [skip] snapshot {sn}: {e}", flush=True)
            continue
        n_full += nf; n_part += np_; n_skip += ns
        print(f"  [task {task_id}] {k + 1}/{len(my_snaps)} snap {sn}: {len(gxs)} planned -> "
              f"{nf} extracted, {np_} backfilled, {ns} already complete "
              f"(totals: {n_full}/{n_part}/{n_skip})", flush=True)

    print(f"[task {task_id}] done: {n_full} extracted, {n_part} backfilled, {n_skip} complete "
          f"-> {out_root}", flush=True)


if __name__ == "__main__":
    main()
