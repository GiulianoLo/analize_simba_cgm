"""Simulation path management and catalog access.

Provides:

* :class:`Simulation` – generic, config-driven simulation handle.
* :class:`Simba` – SIMBA-specific convenience (knows box sizes,
  snapshot ranges, and ships the snap-to-redshift mapping).
"""

import os
import numpy as np
from astropy.cosmology import Planck15 as cosmo

from .config import get_simulation_config

# ── package-data path for snap→z mapping files ────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "snap_z_maps")


# ======================================================================
#  Generic Simulation
# ======================================================================

class Simulation:
    """Handle for any simulation's file layout.

    Create from a **config entry** (set up once per machine with
    :func:`simbanator.add_simulation`), or give paths **directly**.

    Parameters
    ----------
    name : str, optional
        Simulation identifier that matches an entry in
        ``~/.simbanator/config.json``.
    data_dir : str, optional
        Directory containing snapshot files.  If given, *name* is
        used only as a label (no config lookup).
    catalog_dir : str, optional
        Directory containing catalog files.  Defaults to
        ``<data_dir>/Groups/``.
    file_format : str, optional
        Python format-string for catalog filenames, e.g.
        ``'m100n1024_{snap:03d}.hdf5'``.
    snap_format : str, optional
        Format-string for raw snapshot filenames.  Defaults to
        ``'snap_' + file_format``.

    Examples
    --------
    >>> # From config (recommended):
    >>> sim = Simulation("simba_m100n1024")
    >>>
    >>> # With explicit paths:
    >>> sim = Simulation("my_sim", data_dir="/data/sims/run1",
    ...                  file_format="snap_{snap:04d}.hdf5")
    """

    def __init__(self, name=None, *, data_dir=None, catalog_dir=None,
                 file_format=None, snap_format=None):

        if data_dir is not None:
            # Explicit paths – no config lookup.
            self.name = name or "default"
            self.data_dir = str(data_dir)
            self.catalog_dir = str(catalog_dir) if catalog_dir else os.path.join(self.data_dir, "Groups")
            self.file_format = file_format or "{snap:03d}.hdf5"
        elif name is not None:
            # Load from ~/.simbanator/config.json
            cfg = get_simulation_config(name)
            self.name = name
            self.data_dir = cfg["data_dir"]
            self.catalog_dir = cfg.get("catalog_dir") or os.path.join(self.data_dir, "Groups")
            self.file_format = cfg.get("file_format", "{snap:03d}.hdf5")
        else:
            raise TypeError(
                "Provide either a config name or explicit data_dir.\n"
                "  Simulation('simba_m100n1024')          # from config\n"
                "  Simulation('my_sim', data_dir='/...')   # explicit"
            )

        self.snap_format = snap_format or ("snap_" + self.file_format)
        self.cosmology = cosmo

        # Will be set by subclasses or by user
        self.snaps = None
        self.zeds = None

    # ── path helpers ──────────────────────────────────────────────────

    def get_catalog_file(self, snap):
        """Return the full path to the catalog file for *snap*."""
        return os.path.join(self.catalog_dir, self.file_format.format(snap=int(snap)))

    # Backward-compatible aliases
    get_caesar_file = get_catalog_file

    def get_snapshot_file(self, snap):
        """Return the full path to the raw snapshot file for *snap*."""
        return os.path.join(self.data_dir, self.snap_format.format(snap=int(snap)))

    # Backward-compatible alias
    get_sim_file = get_snapshot_file

    # ── catalog loading (caesar is optional) ──────────────────────────

    def load_catalog(self, snap, verbose=False):
        """Load and return a Caesar catalog for *snap*.

        Requires the ``caesar`` package.
        """
        try:
            import caesar
        except ImportError:
            raise ImportError(
                "caesar is required to load catalogs.\n"
                "Install with:  pip install caesar  (or conda)"
            )
        path = self.get_catalog_file(snap)
        if verbose:
            print(f"Loading catalog: {path}")
        return caesar.load(path)

    # Backward-compatible alias
    get_caesar = load_catalog

    # ── redshift helpers ──────────────────────────────────────────────

    def get_z_from_snap(self, snap):
        """Return the redshift for a single snapshot number.

        Only works if the subclass or user has populated
        ``self.scale_factors``.
        """
        if not hasattr(self, "scale_factors") or self.scale_factors is None:
            raise ValueError("No snap→z mapping loaded for this simulation.")
        return 1.0 / self.scale_factors[int(snap)] - 1

    def get_redshifts(self):
        """Return the full array of redshifts (one per snapshot in ``self.snaps``)."""
        if self.zeds is None:
            raise ValueError("No redshifts loaded for this simulation.")
        return self.zeds

    def __repr__(self):
        return f"Simulation('{self.name}', data_dir='{self.data_dir}')"


# ======================================================================
#  SIMBA convenience
# ======================================================================

# Map box size (Mpc/h) → config name and filename pattern
_SIMBA_BOXES = {
    25:  ("simba_m25n512",   "m25n512_{snap:03d}.hdf5"),
    50:  ("simba_m50n512",   "m50n512_{snap:03d}.hdf5"),
    100: ("simba_m100n1024", "m100n1024_{snap:03d}.hdf5"),
}


class Simba(Simulation):
    """SIMBA simulation handle.

    Wraps :class:`Simulation` with SIMBA-specific defaults:
    snapshot range 6–151, filename conventions, and a bundled
    snap-to-redshift mapping.

    Parameters
    ----------
    box : int
        Box size in Mpc/h (25, 50, or 100).
    variant : str, optional
        Run variant suffix (e.g. ``'noagn'``, ``'nox'``).
    data_dir : str, optional
        Override the data directory (skips config lookup).
    catalog_dir : str, optional
        Override the catalog directory.

    Examples
    --------
    >>> sb = Simba(box=100)                        # reads config
    >>> sb = Simba(box=25, data_dir="/data/hr/")   # explicit path
    """

    SNAP_MIN = 6
    SNAP_MAX = 151

    def __init__(self, box=100, variant=None, *, data_dir=None, catalog_dir=None):
        self.box = box
        self.variant = variant

        if box not in _SIMBA_BOXES:
            raise ValueError(
                f"Unknown SIMBA box size {box}. "
                f"Supported: {list(_SIMBA_BOXES.keys())}"
            )

        config_name, file_fmt = _SIMBA_BOXES[box]
        if variant:
            config_name = f"{config_name}_{variant}"

        if data_dir is not None:
            super().__init__(
                name=config_name,
                data_dir=data_dir,
                catalog_dir=catalog_dir,
                file_format=file_fmt,
            )
        else:
            try:
                super().__init__(name=config_name)
            except KeyError:
                raise KeyError(
                    f"SIMBA box {box} not configured yet.\n"
                    f"Set it up once with:\n"
                    f"  import simbanator as sb\n"
                    f"  sb.add_simulation('{config_name}',\n"
                    f"      data_dir='/path/to/snapshots',\n"
                    f"      catalog_dir='/path/to/Groups')\n"
                    f"\nOr pass data_dir= directly:\n"
                    f"  sb.Simba(box={box}, data_dir='/path/to/data')"
                ) from None
            # The config might not store file_format, so override
            # with the known SIMBA convention.
            self.file_format = file_fmt
            self.snap_format = f"snap_{file_fmt}"

        # SIMBA snapshot range
        self.snaps = np.arange(self.SNAP_MIN, self.SNAP_MAX + 1)[::-1]
        self._load_redshifts()

        # Photometric filters used for SED modelling
        self.filters = _SIMBA_FILTERS
        self.filters_pretty = _SIMBA_FILTERS_PRETTY

    # ── snap → z mapping (bundled as package data) ────────────────────

    def _load_redshifts(self):
        """Load the scale-factor table shipped with the package."""
        zfile = os.path.join(_DATA_DIR, f"zsnap_map_caesar_box{self.box}.txt")
        if not os.path.isfile(zfile):
            self.scale_factors = None
            self.zeds = None
            return
        self.scale_factors = np.loadtxt(zfile)
        self.zeds = np.array([
            1.0 / self.scale_factors[s] - 1 for s in self.snaps
        ])

    def __repr__(self):
        return f"Simba(box={self.box}, data_dir='{self.data_dir}')"


# ── SIMBA photometric filters ────────────────────────────────────────

_SIMBA_FILTERS = [
    "GALEX_FUV", "GALEX_NUV",
    "SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z",
    "2MASS_H", "2MASS_J", "2MASS_Ks",
    "PS1_y", "PS1_z",
    "WISE_RSR_W1", "WISE_RSR_W2", "WISE_RSR_W3", "WISE_RSR_W4",
    "SPITZER_IRAC_36", "SPITZER_IRAC_45",
    "SPITZER_IRAC_58", "SPITZER_IRAC_80",
    "HERSCHEL_PACS_BLUE", "HERSCHEL_PACS_GREEN", "HERSCHEL_PACS_RED",
    "HERSCHEL_SPIRE_PSW", "HERSCHEL_SPIRE_PMW", "HERSCHEL_SPIRE_PLW",
    "JCMT_450", "JCMT_850",
]

_SIMBA_FILTERS_PRETTY = [
    "GALEX FUV", "GALEX NUV",
    "SDSS u", "SDSS g", "SDSS r", "SDSS i", "SDSS z",
    "2MASS H", "2MASS J", "2MASS Ks",
    "PS1 y", "PS1 z",
    "WISE 1", "WISE 2", "WISE 3", "WISE 4",
    r"IRAC $3.6\,\mu\mathrm{m}$", r"IRAC $4.5\,\mu\mathrm{m}$",
    r"IRAC $5.8\,\mu\mathrm{m}$", r"IRAC $8\,\mu\mathrm{m}$",
    "HERSCHEL PACS BLUE", "HERSCHEL PACS GREEN", "HERSCHEL PACS RED",
    "HERSCHEL SPIRE PSW", "HERSCHEL SPIRE PMW", "HERSCHEL SPIRE PLW",
    r"JCMT $450\,\mu\mathrm{m}$", r"JCMT $850\,\mu\mathrm{m}$",
]

