"""I/O: simulation handles, output paths, the simulation config, tables and catalogues.

- ``simba``     Simulation / Simba handles (paths, snapshots, redshifts)
- ``paths``     OutputPaths, repo_root / output_dir / data_file (never the working directory)
- ``config``    the visible simulation config (config/simulations.json)
- ``tables``    explicit string decoding, read_table / write_table, row alignment, HDF5 -> FITS
- ``catalogs``  caesar catalogues read with h5py, snapshot / redshift / time helpers
- ``products``  the shared products: path / require / read, provenance stamp / check
- ``registry``  the product list (path template, producer, dependencies, conventions)
- ``qc``        file integrity checks
- ``sysinfo``   memory-usage printout
"""
from .simba import Simulation, Simba
from .paths import OutputPaths, repo_root, output_dir, data_file
from .config import (
    add_simulation,
    remove_simulation,
    list_simulations,
    load_config,
    config_path,
    CONFIG_FILE,
)
from .tables import (convert_hdf5_fits, as_str, read_table, write_table, float_column, to_frame,
                     align_to_rows, galaxy_keys, tagged_path)
from .catalogs import (findsatellites, catalog_path, file_prefix, has_catalog, dataset, sim_attributes,
                       read_galaxy_columns, parent_halo_columns, nearest_snapshot, z_tag,
                       time_redshift_maps, softening_pkpc)
from . import products, registry
from .products import need, read_frame, read_selection
from .qc import hdf5_status

__all__ = [
    "Simulation", "Simba", "OutputPaths", "repo_root", "output_dir", "data_file",
    "add_simulation", "remove_simulation", "list_simulations",
    "load_config", "config_path", "CONFIG_FILE",
    "convert_hdf5_fits", "as_str", "read_table", "write_table", "float_column", "to_frame",
    "align_to_rows", "galaxy_keys", "tagged_path",
    "findsatellites", "catalog_path", "file_prefix", "has_catalog", "dataset", "sim_attributes",
    "read_galaxy_columns", "parent_halo_columns", "nearest_snapshot", "z_tag",
    "time_redshift_maps", "softening_pkpc",
    "products", "registry", "need", "read_frame", "read_selection", "hdf5_status",
]
