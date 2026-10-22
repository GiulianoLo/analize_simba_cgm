"""Task-based output directory management.

The :class:`OutputPaths` class creates a structured output tree
organised by **simulation** and **analysis task** (not by file
format).

Typical layout::

    output/
    └── simba_m100n1024/
        ├── progenitors/
        ├── filtered_particles/
        │   └── snap_151/
        ├── histories/
        ├── radial_profiles/
        ├── plots/
        ├── sed/
        └── <custom>/

Repository-relative locations (:func:`repo_root`, :func:`output_dir`, :func:`data_file`) come
from the package location, never from the working directory, so a notebook finds the same
files from any folder. ``config/simulations.json`` may override them in a ``"paths"`` section::

    "paths": {"repo_root": "/mnt/home/glorenzon/analize_simba_cgm",
              "output_root": "/mnt/home/glorenzon/analize_simba_cgm/output"}
"""

import os
from pathlib import Path

from .config import REPO_ROOT, load_config

# where data_file() looks, in order (relative to repo_root); the root itself is the legacy
# place of the filter .res files until step 8 of the restructure moves them
DATA_DIRS = ("obs_data", "obs_data/filters", "simbanator/rt/masters", ".")


def _paths_cfg():
    return load_config().get("paths", {}) or {}


def repo_root():
    """The repository root: the ``paths.repo_root`` config entry, else the package's parent."""
    return Path(_paths_cfg().get("repo_root") or REPO_ROOT)


def output_dir(*parts, create=False):
    """``<output_root>/<parts...>``; output_root = config ``paths.output_root`` or ``<repo>/output``."""
    root = Path(_paths_cfg().get("output_root") or (repo_root() / "output"))
    path = root.joinpath(*map(str, parts))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_file(name):
    """Absolute path of a bundled data file (observations, filter curves, RT masters).

    ``name`` may contain sub-folders (``"almac11/ks_table.csv"``). Searched in :data:`DATA_DIRS`
    under :func:`repo_root`; ``FileNotFoundError`` lists every place looked at.
    """
    root = repo_root()
    tried = []
    for d in DATA_DIRS:
        path = (root / d / name).resolve()
        if path.is_file():
            return path
        tried.append(str(path))
    raise FileNotFoundError(f"data file {name!r} not found; looked in:\n  " + "\n  ".join(tried))


class OutputPaths:
    """Manage output directories for a given simulation.

    Directories are created lazily (only when first accessed).

    Parameters
    ----------
    simulation_name : str
        Label for the simulation (becomes a subdirectory).
    base_dir : str, optional
        Root output directory.  Defaults to :func:`output_dir` (``<repo>/output``).

    Examples
    --------
    >>> out = OutputPaths("simba_m100n1024")
    >>> out.progenitors
    '/home/user/project/output/simba_m100n1024/progenitors'
    >>> out.filtered_snap(151)
    '/home/user/project/output/simba_m100n1024/filtered_particles/snap_151'
    >>> out.subdir("my_custom_analysis")
    '/home/user/project/output/simba_m100n1024/my_custom_analysis'
    """

    def __init__(self, simulation_name, base_dir=None):
        self.simulation_name = simulation_name
        self.base_dir = str(base_dir or output_dir())
        self.root = os.path.join(self.base_dir, simulation_name)

    def _ensure(self, path):
        """Create *path* if it doesn't exist and return it."""
        os.makedirs(path, exist_ok=True)
        return path

    # ── pre-defined task directories (lazy) ───────────────────────────

    @property
    def progenitors(self):
        """``<root>/progenitors/`` – progenitor index tables."""
        return self._ensure(os.path.join(self.root, "progenitors"))

    @property
    def filtered_particles(self):
        """``<root>/filtered_particles/`` – particle subsets."""
        return self._ensure(os.path.join(self.root, "filtered_particles"))

    def filtered_snap(self, snap):
        """``<root>/filtered_particles/snap_<NNN>/``."""
        return self._ensure(
            os.path.join(self.filtered_particles, f"snap_{int(snap):03d}")
        )

    @property
    def histories(self):
        """``<root>/histories/`` – SFH data, target selections."""
        return self._ensure(os.path.join(self.root, "histories"))

    @property
    def radial_profiles(self):
        """``<root>/radial_profiles/``."""
        return self._ensure(os.path.join(self.root, "radial_profiles"))

    @property
    def plots(self):
        """``<root>/plots/`` – all figures."""
        return self._ensure(os.path.join(self.root, "plots"))

    @property
    def sed(self):
        """``<root>/sed/`` – Powderday inputs and outputs."""
        return self._ensure(os.path.join(self.root, "sed"))

    # ── generic ───────────────────────────────────────────────────────

    def subdir(self, *parts):
        """Create and return a custom subdirectory under the sim root.

        Parameters
        ----------
        *parts : str
            One or more path components, e.g.
            ``out.subdir("convergence_tests", "run3")``.

        Returns
        -------
        str
            Absolute path to the created directory.
        """
        return self._ensure(os.path.join(self.root, *parts))

    def __repr__(self):
        return f"OutputPaths('{self.simulation_name}')"

