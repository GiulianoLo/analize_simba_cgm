"""ks_tracks_lib — pure numpy/h5py helpers for ks_tracks_quenched_m25.ipynb.

Kennicutt–Schmidt (KS) evolution tracks of the cis25 quenched sample: surface densities of
molecular gas and star formation measured from the CAESAR member particles (glist/slist) stored in
the reduced particle files written by ``build_reduced_particles_job.py``, at each galaxy's critical
epochs (sf_peak → sft → qt → post_quench → gas_min → anchor, plus ``end`` = the anchor or, when the
anchor holds no measurable H2, the last snapshot with M_H2/M_star > ``fh2_min``; plus the AGN-feedback
epochs ``agn_ign`` / ``jet_on`` attached from the m25 Part 2d windows table by ``attach_bh_stages``).

**Moved into simbanator (2026-10-22).** The code now lives in the task modules and this file only
re-exports it, so ``import ks_tracks_lib as kl`` keeps working unchanged:

- ``simbanator.particles.ks``          apertures, He factor, sfr_window, measure_ks, ks_columns
- ``simbanator.particles.kinematics``  kappa_rot, measure_zone_kinematics, KIN_ZONES
- ``simbanator.particles.grains``      measure_zone_budget, nfw_concentration, nfw_vesc_kms
- ``simbanator.particles.cutouts``     reduced_path, load_reduced
- ``simbanator.particles.frames``      face_on_R;  ``simbanator.particles.profiles``  half_mass_radius
- ``simbanator.histories.events``      the stages, build_stage_records, stage_time_order, nearest_row,
                                       interp_track;  ``simbanator.agn``  attach_bh_stages
- ``simbanator.obs``                   RELATIONS, relation_y, tdep_ms_gyr
- ``simbanator.stats``                 grid_stats, ecdf;  ``simbanator.io.catalogs``  make_a_to_t

``simbanator`` no longer needs yt to import, so these are still unit-tested locally
(``tests/test_ks_tracks.py``). The notebook still passes ``find_quenching_times`` and the ``a -> t``
callable explicitly.

Conventions (all documented in the notebook's closing cell as well)
-------------------------------------------------------------------
* Face-on projection: ``principal_axes`` (simbanator.utils.geometry) returns the principal axes as the
  COLUMNS of ``evecs`` and ``rotate_to_frame`` uses ``pos @ evecs``; the reduced job stores that
  matrix verbatim, so the face-on cylindrical radius is ``hypot(*(pos @ evecs)[:, :2].T)``.
  (``pos @ evecs.T`` — used by two older notebooks — projects onto the wrong frame.)
* Simulated ``m_H2`` is hydrogen-only (``build_profiles_job._components``); the observed M_H2
  (alpha_CO = 4.36) includes helium, hence ``HE_FACTOR = 1.36`` on the simulated H2 for the comparison
  columns only (raw masses are kept in the tables).
* Half-mass apertures: the mean surface density inside the face-on half-mass radius,
  ``M(<R50)/(pi R50^2)``, is IDENTICALLY ``0.5 M_tot/(pi R50^2)`` — the observed convention — so no
  separate 0.5 factor is ever multiplied in. For the SFR the observed convention assumes the star
  formation follows the CO (``0.5 SFR_tot / (pi R_CO^2)``); ``ks_columns`` provides both that
  (``logSigmaSFR_obs``, used on the R50_H2 rows) and the literal SFR(<R)/area (``logSigmaSFR_inside``).
  Fixed apertures never carry a 0.5.
* SFR = 0 inside an aperture over a window is a censored value: the one-particle floor
  ``m_star_particle / window / area`` is stored as an upper limit (``is_ul``).
* Rotation support: ``kappa_rot`` is the Sales+12 K_rot/K estimator of the m25 notebook's Part 8j0
  (``_kin``), verbatim: velocities about the weighted mean of the subset (its bulk motion), spin
  axis = the subset's angular momentum, so a constant velocity unit cancels. ``measure_zone_kinematics``
  evaluates it for all gas / H2-weighted gas / stars in SPHERICAL zones about the stored centre
  (``KIN_ZONES`` = the m25 ladder's core ``ap3kpc`` (r < 3.16 kpc), outskirt shell ``ann10kpc``
  (3.16 < r < 10 kpc) and the ``ap10kpc`` rung) together with the mass-weighted stellar age of the
  same zone — the reduced files must carry ``vel`` (build_reduced_particles_job, 2026-08-28+;
  older files are backfilled by re-running the plan); without it the kappas are NaN, the ages not.
"""
from simbanator.obs import RELATIONS, relation_y, tdep_ms_gyr  # noqa: F401
from simbanator.particles.cutouts import reduced_path, load_reduced  # noqa: F401
from simbanator.particles.frames import face_on_R  # noqa: F401
from simbanator.particles.profiles import half_mass_radius  # noqa: F401
from simbanator.io.catalogs import make_a_to_t  # noqa: F401
from simbanator.particles.ks import FIXED_AP_KPC, FIXED_AP_LABELS, ADAPTIVE_AP, AP_LABELS, HE_FACTOR, sfr_window, MEASURE_COLUMNS, _f64, measure_ks, ks_columns  # noqa: F401
from simbanator.particles.kinematics import KIN_ZONES, KIN_COMPONENTS, ZONE_KIN_COLUMNS, kappa_rot, measure_zone_kinematics  # noqa: F401
from simbanator.particles.grains import G_KPC_KMS2, BUDGET_ZONES, BUDGET_COLUMNS, nfw_concentration, nfw_vesc_kms, measure_zone_budget  # noqa: F401
from simbanator.histories.events import STAGES_BH, STAGES_KS, STAGES_PLOT, FH2_MIN_END, _smooth, _ordered_positive, _peak_time, _trough_before_floor, build_stage_records, stage_time_order, nearest_row, interp_track  # noqa: F401
from simbanator.agn import attach_bh_stages  # noqa: F401
from simbanator.stats import grid_stats, ecdf  # noqa: F401

__all__ = [
    "STAGES_KS",
    "STAGES_PLOT",
    "FH2_MIN_END",
    "FIXED_AP_KPC",
    "FIXED_AP_LABELS",
    "ADAPTIVE_AP",
    "AP_LABELS",
    "HE_FACTOR",
    "RELATIONS",
    "tdep_ms_gyr",
    "relation_y",
    "BUDGET_ZONES",
    "BUDGET_COLUMNS",
    "measure_zone_budget",
    "nfw_concentration",
    "nfw_vesc_kms",
    "reduced_path",
    "load_reduced",
    "face_on_R",
    "half_mass_radius",
    "make_a_to_t",
    "sfr_window",
    "measure_ks",
    "MEASURE_COLUMNS",
    "ks_columns",
    "build_stage_records",
    "stage_time_order",
    "STAGES_BH",
    "nearest_row",
    "attach_bh_stages",
    "interp_track",
    "grid_stats",
    "ecdf",
    "KIN_ZONES",
    "KIN_COMPONENTS",
    "ZONE_KIN_COLUMNS",
    "kappa_rot",
    "measure_zone_kinematics",
]
