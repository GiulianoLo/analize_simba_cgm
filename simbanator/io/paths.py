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
"""

import os


class OutputPaths:
    """Manage output directories for a given simulation.

    Directories are created lazily (only when first accessed).

    Parameters
    ----------
    simulation_name : str
        Label for the simulation (becomes a subdirectory).
    base_dir : str, optional
        Root output directory.  Defaults to ``./output`` relative to
        the current working directory.

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
        self.base_dir = base_dir or os.path.join(os.getcwd(), "output")
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

