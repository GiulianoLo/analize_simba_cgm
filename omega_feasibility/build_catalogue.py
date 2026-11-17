"""Build the OMEGA feasibility catalogue (the only step that reads particles).

Per box (cis25 = m25n512, cis100 = m100n1024):
  highz  every galaxy at the z~8/7/6 snapshots: catalogue quantities + star-particle ages
         (SFR10, SFR100, N_young) + true gas-phase O/H, C/O, N/O of the star-forming gas
         -> output/<sim>/omega_feasibility/omega_highz.{parquet|h5} (+ .csv)
  lowz   every galaxy at z~0.4 (catalogue only), the quiescent flag and the compact / disky class
         -> omega_lowz.{parquet|h5}
  trees  main branches of every z~0.4 galaxy back to z~8 (catalogue columns only, one read per
         catalogue): the progenitor index at the z~6-8 snapshots for every galaxy (forward test) and
         the full tracks of the quiescent sample (figure a)
         -> omega_progenitors.{parquet|h5}, omega_tracks.npz

Run from the repository root (the m25 merger-tree sidecars are found under output/<sim>/progen_links):
    python -m omega_feasibility.build_catalogue --sims cis25 cis100 --stages highz lowz trees

Local test through the gvfs mount: set SIMBANATOR_CONFIG to the cluster config and OMEGA_PATH_PREFIX
to the mount root (prepended to the data / catalogue paths; the sidecars are then read from
$OMEGA_PATH_PREFIX/<repo on the cluster>/output via OMEGA_PROGEN_DIR).
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omega_feasibility import config as C  # noqa: E402

import simbanator as sb  # noqa: E402
from simbanator.io.catalogs import read_galaxy_columns, make_a_to_t  # noqa: E402
from simbanator.histories.trees import main_progenitors  # noqa: E402
from simbanator.histories.properties import trace_main_branches, gas_oh12  # noqa: E402
from simbanator.selection import quenched_sample  # noqa: E402

CAT_COLS = {
    "mstar": "masses.stellar", "mgas": "masses.gas", "mh2": "masses.H2", "mdust": "masses.dust",
    "sfr": "sfr", "sfr_100_caesar": "sfr_100", "central": "central", "nstar": "nstar", "ngas": "ngas",
    "z_sfr": "metallicities.sfr_weighted", "z_mw": "metallicities.mass_weighted",
    "r20_star": "radii.stellar_r20", "r50_star": "radii.stellar_half_mass", "r80_star": "radii.stellar_r80",
}
TRACK_COLS = {"mstar": "masses.stellar", "sfr": "sfr", "mgas": "masses.gas",
              "z_sfr": "metallicities.sfr_weighted", "nstar": "nstar", "central": "central"}


# ─────────────────────────────────────────────────────────────────────────────
# simulation handle, cosmology
# ─────────────────────────────────────────────────────────────────────────────

def simulation(name):
    s = sb.Simulation(name)
    pre = os.environ.get("OMEGA_PATH_PREFIX")
    if pre:
        s.data_dir = pre + s.data_dir
        s.catalog_dir = pre + s.catalog_dir
    return s


def progenitors(s, snap):
    p = main_progenitors(s, snap)
    if p is None and os.environ.get("OMEGA_PROGEN_DIR"):
        side = os.path.join(os.environ["OMEGA_PROGEN_DIR"], s.name, "progen_links",
                            os.path.basename(s.get_caesar_file(snap)))
        if os.path.exists(side):
            with h5py.File(side, "r") as f:
                q = f["tree_data/progen_galaxy_star"][:]
            p = (q[:, 0] if q.ndim == 2 else q).astype(np.int64)
    return p


def sim_cosmology(cat):
    """The run's own cosmology from the catalogue attributes (SIMBA: h=0.68, Om=0.3, OL=0.7)."""
    from astropy.cosmology import FlatLambdaCDM
    with h5py.File(cat, "r") as f:
        a = f["simulation_attributes"].attrs
        h, om, ob, z = float(a["hubble_constant"]), float(a["omega_matter"]), float(a["omega_baryon"]), float(a["redshift"])
    return FlatLambdaCDM(H0=100 * h, Om0=om, Ob0=ob), z


def catalogue(s, snap):
    path = s.get_caesar_file(snap)
    cols = read_galaxy_columns(path, CAT_COLS)
    miss = cols.pop("_missing")
    if miss:
        print(f"  [{s.name} {snap}] missing catalogue columns: {miss}", flush=True)
    with h5py.File(path, "r") as f:
        g = f["galaxy_data"]
        lists = {k: (g[f"{k}_start"][:].astype(np.int64), g[f"{k}_end"][:].astype(np.int64), g[f"lists/{k}"][:].astype(np.int64))
                 for k in ("slist", "glist")}
    return path, cols, lists


# ─────────────────────────────────────────────────────────────────────────────
# particle reads: only the rows we need
# ─────────────────────────────────────────────────────────────────────────────

def read_rows(ds, idx, cols=None, window=1 << 22, dense=0.02):
    """``ds[idx]`` (optionally ``[:, cols]``) for sorted unique ``idx`` without loading the whole array.

    Windows of ``window`` rows that contain indices are read as one slab when the indices are dense
    (> ``dense`` of the window), otherwise by point selection.
    """
    idx = np.asarray(idx, np.int64)
    shape = (len(idx),) if ds.ndim == 1 else (len(idx), ds.shape[1] if cols is None else len(cols))
    out = np.empty(shape, ds.dtype)
    if len(idx) == 0:
        return out
    win = idx // window
    cuts = np.flatnonzero(np.diff(win)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(idx)]):
        ii = idx[a:b]
        if (b - a) > dense * window:
            lo, hi = int(ii[0]), int(ii[-1]) + 1
            slab = ds[lo:hi] if ds.ndim == 1 else ds[lo:hi, :]
            v = slab[ii - lo]
        else:
            v = ds[ii] if ds.ndim == 1 else ds[ii, :]
        if ds.ndim == 2 and cols is not None:
            v = v[:, cols]
        out[a:b] = v
    return out


def members(start, end, flat):
    """Concatenated particle indices of every galaxy + the galaxy each belongs to."""
    n = (end - start).clip(0)
    owner = np.repeat(np.arange(len(start)), n)
    pos = np.concatenate([np.arange(s, e) for s, e in zip(start, end)]) if n.sum() else np.empty(0, np.int64)
    return flat[pos], owner


def number_ratio(num, den):
    with np.errstate(all="ignore"):
        return np.where((num > 0) & (den > 0), np.log10(num / den), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# stage highz
# ─────────────────────────────────────────────────────────────────────────────

def build_highz_snapshot(s, zt, snap):
    t0 = time.time()
    cat, cols, lists = catalogue(s, snap)
    cosmo, z = sim_cosmology(cat)
    ngal = len(cols["mstar"])
    t_now = cosmo.age(z).value * 1e3  # Myr
    a_to_t = make_a_to_t(cosmo, a_min=0.01)
    snapf = s.get_snapshot_file(snap)

    # stars: ages -> SFR10, SFR100, N_young
    sidx, sown = members(*lists["slist"])
    order = np.argsort(sidx); sidx, sown = sidx[order], sown[order]
    with h5py.File(snapf, "r") as f:
        aform = read_rows(f["PartType4/StellarFormationTime"], sidx)
        mst = read_rows(f["PartType4/Masses"], sidx).astype(float) * 1e10 / cosmo.h
    age = t_now - a_to_t(aform) * 1e3
    out = {"n_young10": np.zeros(ngal, int), "n_young100": np.zeros(ngal, int)}
    for T in C.T_SFR_MYR:
        young = age < T
        out[f"sfr{int(T)}"] = np.bincount(sown[young], weights=mst[young], minlength=ngal) / (T * 1e6)
        out[f"n_young{int(T)}"] = np.bincount(sown[young], minlength=ngal)
    out[f"n_young{int(C.T_YOUNG_MYR)}"] = np.bincount(sown[age < C.T_YOUNG_MYR], minlength=ngal)
    print(f"  [{s.name} {snap}] {len(sidx)} star particles in galaxies, {time.time() - t0:.0f} s", flush=True)

    # gas: star-forming members only -> O/H, C/O, N/O (SFR- and mass-weighted)
    gidx, gown = members(*lists["glist"])
    order = np.argsort(gidx); gidx, gown = gidx[order], gown[order]
    with h5py.File(snapf, "r") as f:
        gsfr = read_rows(f["PartType0/StarFormationRate"], gidx).astype(float)
        sf = gsfr > 0
        gidx_sf, gown_sf, gsfr_sf = gidx[sf], gown[sf], gsfr[sf]
        mc = list(C.METAL_COLS.values())
        met = read_rows(f["PartType0/Metallicity"], gidx_sf, cols=mc).astype(float)
        mg = read_rows(f["PartType0/Masses"], gidx_sf).astype(float) * 1e10 / cosmo.h
    X = {k: met[:, i] for i, k in enumerate(C.METAL_COLS)}
    xh = 1.0 - X["Z"] - X["He"]
    nh, no, nc, nn = (xh / C.ATOMIC_MASS["H"], X["O"] / C.ATOMIC_MASS["O"],
                      X["C"] / C.ATOMIC_MASS["C"], X["N"] / C.ATOMIC_MASS["N"])
    out["n_sfgas"] = np.bincount(gown_sf, minlength=ngal)
    out["sfr_gas"] = np.bincount(gown_sf, weights=gsfr_sf, minlength=ngal)
    for tag, w in (("sfr", gsfr_sf), ("mw", mg)):
        S = {k: np.bincount(gown_sf, weights=w * v, minlength=ngal)
             for k, v in (("H", nh), ("O", no), ("C", nc), ("N", nn))}
        out[f"oh12_{tag}"] = 12 + number_ratio(S["O"], S["H"])
        out[f"logCO_{tag}"] = number_ratio(S["C"], S["O"])
        out[f"logNO_{tag}"] = number_ratio(S["N"], S["O"])
    print(f"  [{s.name} {snap}] {len(gidx)} gas particles in galaxies, {sf.sum()} star-forming, "
          f"{time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame({k: np.asarray(v) for k, v in cols.items()})
    for k, v in out.items():
        df[k] = v
    df.insert(0, "galaxy_id", np.arange(ngal))
    df.insert(0, "z", z); df.insert(0, "z_target", zt); df.insert(0, "snap", snap)
    df.insert(0, "run", C.SIMS[s.name]); df.insert(0, "sim", s.name)
    df["central"] = df["central"].astype(bool)
    with np.errstate(all="ignore"):
        df["log_mstar"] = np.log10(df["mstar"].where(df["mstar"] > 0))
        df["f_gas"] = df["mgas"] / (df["mgas"] + df["mstar"])
        df["oh12_caesar"] = gas_oh12(df["z_sfr"].to_numpy())
    df["resolved_star"] = df["nstar"] >= C.N_STAR_MIN
    df["resolved_young"] = df[f"n_young{int(C.T_YOUNG_MYR)}"] >= C.N_YOUNG_MIN
    df["resolved"] = df["resolved_star"] & df["resolved_young"]
    return df


# NOTE on the weights: SFR-weighted = sum_i SFR_i x_i / sum_i SFR_i (x = per-particle number density
# per unit mass, so the ratio of the weighted sums is the SFR-weighted abundance ratio); mass-weighted =
# sum_i m_i x_i / sum_i m_i, i.e. the ratio of the total numbers of atoms in the star-forming gas.


# ─────────────────────────────────────────────────────────────────────────────
# stage lowz
# ─────────────────────────────────────────────────────────────────────────────

def build_lowz(s):
    snap = C.LOWZ_SNAP
    cat = s.get_caesar_file(snap)
    cols = read_galaxy_columns(cat, CAT_COLS); cols.pop("_missing")
    cosmo, z = sim_cosmology(cat)
    df = pd.DataFrame({k: np.asarray(v) for k, v in cols.items()})
    df.insert(0, "galaxy_id", np.arange(len(df)))
    df.insert(0, "z", z); df.insert(0, "snap", snap); df.insert(0, "run", C.SIMS[s.name]); df.insert(0, "sim", s.name)
    df["central"] = df["central"].astype(bool)
    with np.errstate(all="ignore"):
        df["log_mstar"] = np.log10(df["mstar"].where(df["mstar"] > 0))
        df["C_r80_r20"] = df["r80_star"] / df["r20_star"]
    mask, _ = quenched_sample(df["mstar"], df["sfr"], cosmo.age(z).value * 1e9, central=df["central"],
                              mass_floor=C.Q_MASS_FLOOR, passive_factor=C.Q_PASSIVE_FACTOR,
                              ngas_min=None, nstar_min=None, dust_to_h2_min=None, centrals_only=True)
    df["quiescent"] = mask
    cut = np.nanmedian(df.loc[mask, "C_r80_r20"])
    df["qclass"] = ""
    df.loc[mask & (df["C_r80_r20"] >= cut), "qclass"] = "compact"
    df.loc[mask & (df["C_r80_r20"] < cut), "qclass"] = "disky"
    df.attrs["C_split"] = float(cut)
    print(f"  [{s.name} {snap}] z={z:.3f}: {mask.sum()} quiescent centrals (log M*>{C.Q_MASS_FLOOR}), "
          f"C split {cut:.2f}: {(df.qclass == 'compact').sum()} compact / {(df.qclass == 'disky').sum()} disky", flush=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# stage trees
# ─────────────────────────────────────────────────────────────────────────────

def build_trees(s, lowz):
    snap_lo = min(min(C.HIGHZ_SNAPS.values()), C.TRACK_SNAP_MIN)

    def read_columns(snap):
        cat = s.get_caesar_file(snap)
        cols = read_galaxy_columns(cat, TRACK_COLS); cols.pop("_missing")
        with h5py.File(cat, "r") as f:
            z = float(f["simulation_attributes"].attrs["redshift"])
        cols["gid"] = np.arange(len(cols["mstar"]), dtype=float)
        return cols, z

    anchors = {C.LOWZ_SNAP: lowz["galaxy_id"].to_numpy()}
    res = trace_main_branches(anchors, C.LOWZ_SNAP, snap_lo, read_columns, lambda q: progenitors(s, q))[C.LOWZ_SNAP]
    snaps = list(res["snaps"])
    prog = pd.DataFrame({"sim": s.name, "galaxy_id_lowz": res["ids"]})
    for zt, sn in C.HIGHZ_SNAPS.items():
        g = res["data"]["gid"][snaps.index(sn)]
        prog[f"prog_id_z{zt}"] = np.where(np.isfinite(g), g, -1).astype(int)
    q = lowz.set_index("galaxy_id").loc[res["ids"]]
    prog["quiescent"] = q["quiescent"].to_numpy(); prog["qclass"] = q["qclass"].to_numpy()
    prog["log_mstar_lowz"] = q["log_mstar"].to_numpy()
    sel = prog["quiescent"].to_numpy()
    tracks = {"snaps": res["snaps"], "z": res["z"], "ids": res["ids"][sel], "qclass": prog["qclass"].to_numpy()[sel].astype("U8")}
    for k, v in res["data"].items():
        tracks[k] = v[:, sel]
    return prog, tracks


# ─────────────────────────────────────────────────────────────────────────────

def write_table(df, path_noext):
    df.to_csv(path_noext + ".csv", index=False)
    try:
        df.to_parquet(path_noext + ".parquet", index=False)
        print("  ->", path_noext + ".parquet (+ .csv)", flush=True)
    except Exception as e:  # pyarrow / fastparquet absent on the kernel
        with h5py.File(path_noext + ".h5", "w") as f:
            for c in df.columns:
                v = df[c].to_numpy()
                f[c] = v.astype("S") if v.dtype == object else v
        print(f"  -> {path_noext}.h5 (+ .csv; parquet unavailable: {type(e).__name__})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="+", default=list(C.SIMS))
    ap.add_argument("--stages", nargs="+", default=["highz", "lowz", "trees"])
    ap.add_argument("--zt", nargs="+", type=int, default=list(C.HIGHZ_SNAPS), help="target redshifts (highz stage)")
    a = ap.parse_args(argv)
    for name in a.sims:
        s = simulation(name); od = C.out_dir(name)
        print(f"== {name} ({C.SIMS[name]})", flush=True)
        if "highz" in a.stages:
            df = pd.concat([build_highz_snapshot(s, zt, C.HIGHZ_SNAPS[zt]) for zt in a.zt], ignore_index=True)
            write_table(df, os.path.join(od, "omega_highz"))
        if "lowz" in a.stages or "trees" in a.stages:
            lowz = build_lowz(s)
            write_table(lowz, os.path.join(od, "omega_lowz"))
        if "trees" in a.stages:
            prog, tracks = build_trees(s, lowz)
            write_table(prog, os.path.join(od, "omega_progenitors"))
            np.savez(os.path.join(od, "omega_tracks.npz"), **tracks)
            print("  ->", os.path.join(od, "omega_tracks.npz"), flush=True)


if __name__ == "__main__":
    main()
