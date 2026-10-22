"""Everything along a galaxy's branch through time.

- ``trees``       progenitor tracks from the caesar merger trees
- ``mergers``     merger detection and classification along the tracks
- ``properties``  per-snapshot property histories (mass, gas, dust, H2, Z, ...)
- ``sfh``         star-formation histories: smoothing, archaeological / region SFHs, delayed-bq fits
- ``sfh_fsps``    FSPS-based SFH recovery (needs yt + fsps; import it explicitly)
- ``events``      quenching events (SFT / QT), critical points and clocks
"""
from .trees import (caesar_read_progen, read_progen, progen_tree_file, main_progenitors, main_descendants,
                    load_anchor_history, progenitor_index_matrix, align_by_ids)
from .mergers import Progenitor, Galaxy, process_galaxies_with_tracks, analyze_mergers
from .properties import (HDF5BuildHistory, find_property_threshold_crossings_from_hdf5, safe_ratio, gas_oh12,
                         dust_to_metal_devis, time_derivative, trace_main_branches)
from .sfh import (smooth_resample_sfh, recent_sfr, sfr_delayed_bq, fit_delayed_bq,
                  build_mfrac_lookup, mfrac_of, archaeological_sfh, projected_region_sfh,
                  formed_mass_sfh, cumulative_ssfr, load_sfh_archive, sfh_archive_attrs, sfh_shape_stats)
from .events import (find_quenching_times, load_quenching_events, ssfr_track, quench_event, quench_records,
                     track_start, first_downward_crossing, values_at_rows, ratio_at_rows, regrid, log_track,
                     aligned_stack, binned_event_profile)

__all__ = [
    "caesar_read_progen", "read_progen", "progen_tree_file", "main_progenitors", "main_descendants",
    "load_anchor_history", "progenitor_index_matrix", "align_by_ids",
    "Progenitor", "Galaxy", "process_galaxies_with_tracks", "analyze_mergers",
    "HDF5BuildHistory", "find_property_threshold_crossings_from_hdf5", "safe_ratio", "gas_oh12",
    "dust_to_metal_devis", "time_derivative", "trace_main_branches",
    "smooth_resample_sfh", "recent_sfr", "sfr_delayed_bq", "fit_delayed_bq",
    "build_mfrac_lookup", "mfrac_of", "archaeological_sfh", "projected_region_sfh",
    "formed_mass_sfh", "cumulative_ssfr", "load_sfh_archive", "sfh_archive_attrs", "sfh_shape_stats",
    "find_quenching_times", "load_quenching_events", "ssfr_track", "quench_event", "quench_records",
    "track_start", "first_downward_crossing", "values_at_rows", "ratio_at_rows", "regrid", "log_track",
    "aligned_stack", "binned_event_profile",
]
