#!/usr/bin/env python
"""Cluster job — add merger-tree links (``tree_data/progen_galaxy_star``) to the SIMBA_25 catalogs.

Motivation
----------
The shared cis25 (m25n512) Caesar catalogs in ``/mnt/home/share/simbas/SIMBA_25/Groups/`` were
never given the ``tree_data`` group that the cis50/cis100 catalogs ship with. Without it,
``caesar_read_progen`` produces an empty progenitor FITS and ``HDF5BuildHistory.get_history_indx``
cannot trace any galaxy backward — every history build for this box fails (this is exactly the
``[pre-select] 0/5382`` -> ``_unwrap_positions`` crash in ``powderday_flux_quenched_m25.ipynb``).

This is the same one-time fix previously applied to the m50 box (``~/find_progen.py`` on the
cluster): run caesar's progen matcher over consecutive snapshot pairs. It needs BOTH snapshot
files of each pair — the catalogs store galaxy star members as snapshot *indices* (``slist``),
and progen reads ``PartType4/ParticleIDs`` via pygadgetreader to match galaxies across epochs.
All required snapshots (019–151) are present on the share.

Layout gotcha: caesar builds the catalog path as ``<snapdir>/Groups/<prefix><snapname minus
'snap_'>NNN.hdf5``. The m50 catalogs were named ``caesar_m50n512_*`` (prefix='caesar_'); the m25
ones are plain ``m25n512_*.hdf5`` -> **prefix=''**.

Run ON the cluster (env with caesar + pygadgetreader), ideally as a SLURM job (~130 pairs, a few
hours). ``write_progens`` opens the shared catalogs in r+ mode: check file-level write permission
first (``test -w .../Groups/m25n512_134.hdf5``). The write is additive (only adds ``tree_data``,
standard caesar layout). Decreasing snapshot order -> progenitors; the range is resumable /
splittable across jobs as long as consecutive ranges overlap by one snapshot, e.g.::

    python find_progen_m25_job.py             # full run, snaps 151 -> 20
    python find_progen_m25_job.py 151 100     # then: 100 50, then: 50 19
    python find_progen_m25_job.py 134 133     # single-pair smoke test first

Afterwards delete the stale empty ``output/cis25/progenitors/progenitors_anchor_*.fits`` and
re-run the notebook's BUILD_MULTI_Z cell (expect ~97/5382 pre-selected at z=0.31).
"""
import sys

import caesar

SNAPDIR  = "/mnt/home/share/simbas/SIMBA_25"
FILEROOT = "snap_m25n512_"

# pairs (hi, hi-1) ... (lo+1, lo): tree_data is written into the HIGHER snap of each pair,
# so the default covers snaps 20..151 — deeper than any anchor's tracking window (end_snap >= 36)
hi = int(sys.argv[1]) if len(sys.argv) > 1 else 151
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 19

caesar.progen.run_progen(SNAPDIR, FILEROOT, list(range(hi, lo, -1)), prefix='')
