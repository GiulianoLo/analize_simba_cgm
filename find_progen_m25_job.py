#!/usr/bin/env python
"""Cluster job — build merger-tree links for the SIMBA_25 catalogs as SIDECAR files.

Motivation
----------
The shared cis25 (m25n512) Caesar catalogs in ``/mnt/home/share/simbas/SIMBA_25/Groups/`` were
never given the ``tree_data`` group that the cis50/cis100 catalogs ship with. Without it,
``caesar_read_progen`` produces an empty progenitor FITS and ``HDF5BuildHistory.get_history_indx``
cannot trace any galaxy backward — every history build for this box fails (this is exactly the
``[pre-select] 0/5382`` -> ``_unwrap_positions`` crash in ``powderday_flux_quenched_m25.ipynb``).

The catalogs are READ-ONLY at the file level (caesar's in-place ``run_progen`` dies with
errno 13), so this job writes each snapshot's links to a **sidecar** HDF5 you own instead:

  output/cis25/progen_links/m25n512_NNN.hdf5
    tree_data/progen_galaxy_star : (ngal, 2) int32 — [most massive, second most massive]
                                   progenitor catalog index at snap NNN-1; -1 = none

— the same dataset name/layout as in-catalog trees, so ``[:, 0]`` consumers are unchanged.
simbanator's readers (``caesar_read_progen`` / ``get_history_indx``) check the catalog first and
fall back to the sidecar automatically (``progen_tree_file``).

Matching needs BOTH snapshot files of each pair — the catalogs store galaxy star members as
snapshot *indices* (``slist``), and caesar's ``progen_finder`` reads ``PartType4/ParticleIDs``
via pygadgetreader to match galaxies across epochs. All snapshots (019–151) are on the share.
Consecutive pairs share a Caesar object (each catalog is loaded once per task), and finished
sidecars are skipped, so a killed task resumes where it stopped.

Run ON the cluster (pd39 has caesar 0.2b0 + pyGadgetReader), via ``submit_find_progen_m25.sh``
(4-task array over disjoint sidecar ranges) or manually::

    python find_progen_m25_job.py             # full run: sidecars for snaps 20..151
    python find_progen_m25_job.py 134 133     # single-pair smoke test (sidecar for snap 134)

Afterwards delete the stale empty ``output/cis25/progenitors/progenitors_anchor_*.fits`` and
re-run the notebook's BUILD_MULTI_Z cell (expect ~97/5382 pre-selected at z=0.31).
"""
import os
import sys

import h5py
import numpy as np
import caesar
from caesar.progen import progen_finder

SNAPDIR = "/mnt/home/share/simbas/SIMBA_25"
CATDIR  = os.path.join(SNAPDIR, "Groups")
LINKDIR = os.path.join(os.getcwd(), "output", "cis25", "progen_links")


def _cat(s):
    return os.path.join(CATDIR, f"m25n512_{s:03d}.hdf5")


def _side(s):
    return os.path.join(LINKDIR, f"m25n512_{s:03d}.hdf5")


# pair (s, s-1) -> sidecar for snap s, covering snaps lo+1..hi; the default covers
# 20..151 — deeper than any anchor's tracking window (end_snap >= 36)
hi = int(sys.argv[1]) if len(sys.argv) > 1 else 151
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 19

os.makedirs(LINKDIR, exist_ok=True)

cur = None                        # caesar object at snap s, chained down the range
for s in range(hi, lo, -1):
    done = False
    if os.path.exists(_side(s)):
        with h5py.File(_side(s), "r") as f:
            done = "tree_data/progen_galaxy_star" in f
    if done:
        print(f"[{s}] sidecar exists -> skip", flush=True)
        cur = None                # chain broken; reload at the next unfinished pair
        continue
    if cur is None:
        cur = caesar.load(_cat(s))
    prev = caesar.load(_cat(s - 1))
    # save=False: caesar never touches the (read-only) catalog; we write the sidecar ourselves
    prog = progen_finder(cur, prev, _cat(s), snap_dir=SNAPDIR, save=False, n_most=2)
    if prog is None:              # no galaxies on one side (earliest epochs)
        prog = np.full((0, 2), -1, dtype=np.int32)
        print(f"[{s}] no galaxies to match -> empty sidecar", flush=True)
    prog = np.asarray(prog, dtype=np.int32)
    tmp = _side(s) + ".part"      # atomic publish: a killed write never looks 'done'
    with h5py.File(tmp, "w") as f:
        f.attrs["caesar_catalog"] = _cat(s)
        f.attrs["snap"] = s
        f.attrs["z_current"] = float(cur.simulation.redshift)
        f.attrs["z_progen"] = float(prev.simulation.redshift)
        f.create_dataset("tree_data/progen_galaxy_star", data=prog, compression=1)
    os.replace(tmp, _side(s))
    print(f"[{s}] progen({s}->{s - 1}): {int((prog[:, 0] >= 0).sum())}/{len(prog)} matched "
          f"-> {_side(s)}", flush=True)
    cur = prev
