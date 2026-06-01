"""Galaxy merger detection and classification from Caesar HDF5 catalogs.

Provides:

* :class:`Progenitor` – properties of a potential merger companion at one snapshot.
* :class:`Galaxy` – container for a galaxy's progenitor history across snapshots.
* :func:`process_galaxies_with_tracks` – scan Caesar HDF5 snapshot catalogs for
  neighbours within a search radius and record them as merger candidates.
* :func:`analyze_mergers` – classify candidates into major/minor merger counts
  per galaxy per snapshot.

Typical usage::

    from simbanator.io.simba import Simulation
    from simbanator.analysis.mergers import process_galaxies_with_tracks, analyze_mergers

    sim = Simulation('simba_m100n1024')
    snaplist = list(range(151, 5, -1))   # z=0 first

    galaxies = process_galaxies_with_tracks(
        track_fits_path="output/progenitors/tracks.fits",
        box_size=100.0,                  # Mpc/h
        sb=sim,
        snaplist=snaplist,
    )

    major, minor = analyze_mergers(
        galaxies,
        array_size=(len(snaplist), len(galaxies)),
        mass_threshold_maj=0.25,
        mass_threshold_min=0.10,
    )
"""

import warnings
import h5py
from astropy.io import fits
import numpy as np
import time

# Caesar HDF5 dataset paths for the fields we need.
# Override at module level if your Caesar build uses different internal paths.
_H5_POS   = 'galaxy_data/pos'                          # shape (N, 3), Mpc/h
_H5_SMASS = 'galaxy_data/dicts/masses.stellar'          # M_sun
_H5_RHALF = 'galaxy_data/dicts/radii.stellar_half_mass' # kpc/h (by default)
_H5_H2    = 'galaxy_data/dicts/masses.H2'              # M_sun
_H5_DUST  = 'galaxy_data/dicts/masses.dust'            # M_sun


class Progenitor:
    """Properties of one merger companion galaxy recorded at a single snapshot.

    Parameters
    ----------
    mass : float
        Symmetric stellar mass ratio ``min(m_sec, m_main) / max(m_sec, m_main)`` ∈ (0, 1].
    cat_index : int
        Catalog row index of this companion in its snapshot HDF5 file.
        Only meaningful within the same snapshot — Caesar IDs are not stable
        across snapshots.
    x, y, z : float
        Position of the companion in catalog units.
    merger : int
        1 if this entry represents a merger candidate, 0 otherwise.
    fragmentation : int
        1 if this entry is a tidal fragment of an already-tracked companion
        (detected by positional proximity + lower mass ratio), 0 otherwise.
        Fragments are excluded from merger counts by :func:`analyze_mergers`.
    snapshot : int
        Snapshot index (0-based column) at which this companion was recorded.
    H2 : float
        Molecular hydrogen mass of the companion (M_sun).
    dust : float
        Dust mass of the companion (M_sun).
    distance : float, optional
        Separation from the main galaxy in catalog units (Mpc/h).
    processed : int, optional
        Flag marking whether this progenitor has been processed downstream.
    """

    def __init__(self, mass, x, y, z, merger, snapshot, H2, dust,
                 cat_index=None, fragmentation=0, distance=0, processed=0):
        self.id = None
        self.cat_index = cat_index
        self.mass = mass
        self.x = x
        self.y = y
        self.z = z
        self.merger = merger
        self.fragmentation = fragmentation
        self.snapshot = snapshot
        self.distance = distance
        self.processed = processed
        self.H2 = H2
        self.dust = dust

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Attribute {key} not found in Progenitor")

    def print_info(self):
        print(f"Progenitor ID: {self.id}")
        print(f"Cat index: {self.cat_index}")
        print(f"Mass ratio: {self.mass}")
        print(f"Position: ({self.x}, {self.y}, {self.z})")
        print(f"Merger: {self.merger}")
        print(f"Fragmentation: {self.fragmentation}")
        print(f"Snapshot: {self.snapshot}")
        print(f"Distance: {self.distance}")
        print(f"Processed: {self.processed}")
        print(f"H2: {self.H2}")
        print(f"dust: {self.dust}")


class Galaxy:
    """Container for a galaxy's merger progenitor history.

    Parameters
    ----------
    mass : float, optional
        Stellar mass at the reference snapshot.
    radii_stellar_half_mass : float, optional
        Stellar half-mass radius at the reference snapshot.
    x, y, z : float, optional
        Position at the reference snapshot.
    """

    def __init__(self, mass=0, radii_stellar_half_mass=0, x=0, y=0, z=0):
        self._progenitors = {}
        self._id_counter = 0
        self.mass = mass
        self.r = radii_stellar_half_mass
        self.x = x
        self.y = y
        self.z = z

    def add_progenitor(self, progenitor):
        key = (progenitor.snapshot, progenitor.cat_index)
        for existing in self._progenitors.values():
            if (existing.snapshot, existing.cat_index) == key:
                return
        self._id_counter += 1
        progenitor.id = self._id_counter
        self._progenitors[progenitor.id] = progenitor

    def remove_progenitor(self, id):
        if id in self._progenitors:
            del self._progenitors[id]
        else:
            print(f"No progenitor found with ID {id}.")

    def update_progenitor(self, id, **kwargs):
        if id not in self._progenitors:
            raise ValueError(f"Progenitor with id {id} not found")
        self._progenitors[id].update(**kwargs)

    def update_mass(self, new_mass):
        self.mass = new_mass

    def update_radii_stellar_half_mass(self, new_radii):
        self.r = new_radii

    def update_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_progenitor(self, id):
        return self._progenitors.get(id, None)

    def set_track(self, track_array):
        self.track = track_array

    @property
    def progenitors(self):
        return self._progenitors.copy()

    def get_attribute_values(self, attribute):
        """Return an array of scalar values for *attribute* across all progenitors."""
        values = []
        for progenitor in self._progenitors.values():
            try:
                value = getattr(progenitor, attribute)
                if hasattr(value, "__iter__") and not isinstance(value, str):
                    if len(value) == 1:
                        value = value[0]
                value = float(value)
                values.append(value)
            except AttributeError:
                raise ValueError(f"Attribute {attribute} not found in Progenitor")
            except TypeError:
                raise ValueError(f"Attribute {attribute} cannot be converted to float")
            except ValueError:
                raise ValueError(f"Attribute {attribute} value is not a single numerical value")
        return np.array(values)

    def print_info(self):
        print(f"Galaxy Mass: {self.mass}")
        print(f"Radii Stellar Half Mass: {self.r}")
        print(f"Position: ({self.x}, {self.y}, {self.z})")
        print("Progenitors:")
        for progenitor in self._progenitors.values():
            progenitor.print_info()
            print("-" * 20)


def euclidean_distance(p1, p2):
    """Return the Euclidean separation between two objects with x/y/z attributes."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def plot_sphere(ax, center_x, center_y, center_z, radius, color='b'):
    """Draw a wireframe sphere on *ax* to visualise a search volume."""
    import matplotlib.pyplot as plt  # noqa: F401 – matplotlib is optional

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center_x
    y = radius * np.outer(np.sin(u), np.sin(v)) + center_y
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z
    ax.plot_wireframe(x, y, z, color=color, alpha=0.01)


def process_galaxies_with_tracks(
    track_fits_path,
    box_size,
    *,
    caesar_paths=None,
    sb=None,
    snaplist=None,
    search_radius_factor=5.0,
    mass_threshold=1e9,
    rhalf_unit_factor=1e-3,
    match_radius=0.1,
    frag_radius_factor=2.0,
):
    """Scan Caesar HDF5 snapshot catalogs and record merger companions.

    For every snapshot, each tracked galaxy's neighbours within
    ``search_radius_factor * r_half`` are added as :class:`Progenitor` entries.
    Distances are computed with the minimum-image convention for the periodic box.
    No FITS conversion of the catalogs is required.

    Parameters
    ----------
    track_fits_path : str
        FITS file containing the progenitor track array, shape
        ``(N_galaxies, N_snaps)``.  Produced by
        :func:`~simbanator.analysis.progenitors.caesar_read_progen` or
        :func:`~simbanator.analysis.progenitors.read_progen`.
    box_size : float
        Periodic box side length in the same units as the catalog positions
        (e.g. Mpc/h for SIMBA).
    caesar_paths : list of str, optional
        Ordered list of Caesar HDF5 catalog paths, one per snapshot.
        The order must match the column order in *track_fits_path*.
    sb : :class:`~simbanator.io.simba.Simulation`, optional
        Simulation handle.  Used with *snaplist* to resolve catalog paths
        automatically via ``sb.get_caesar_file(snap)``.
    snaplist : list of int, optional
        Snapshot numbers, ordered to match the track columns.
        Required when *sb* is given instead of *caesar_paths*.
    search_radius_factor : float, optional
        Search sphere radius as a multiple of the stellar half-mass radius.
    mass_threshold : float, optional
        Minimum neighbour stellar mass (M☉) to consider.
    rhalf_unit_factor : float, optional
        Converts the stored half-mass radius to position units.
        Default ``1e-3`` assumes radii in kpc/h, positions in Mpc/h.
    match_radius : float, optional
        Maximum separation (catalog position units, Mpc/h) within which a
        companion at snapshot *t* is considered the same physical object as
        one tracked at snapshot *t-1*.  Should comfortably cover the typical
        comoving displacement between adjacent snapshots (~50–150 kpc/h for
        SIMBA-100).  Default 0.1 Mpc/h (100 kpc/h).
    frag_radius_factor : float, optional
        A new companion (no positional match in the previous snapshot) is
        flagged as a tidal fragment when an already-active companion lies
        within ``frag_radius_factor * match_radius`` **and** carries a higher
        stellar mass ratio.  This suppresses spurious extra merger events
        caused by FoF splitting a single infalling object into pieces.
        Default 2.0, giving a fragment-detection radius of 200 kpc/h with the
        default *match_radius*.

    Returns
    -------
    dict
        ``{galaxy_index: Galaxy}`` with one :class:`Progenitor` per companion
        *approach* (first entry into the search sphere). Tidal fragments are
        included but flagged with ``fragmentation=1``.
    """
    if caesar_paths is not None:
        catalog_paths = list(caesar_paths)
    elif sb is None:
        raise ValueError(
            "Provide either caesar_paths, or sb (+ optionally snaplist)."
        )
    # else: catalog_paths resolved below after reading the track file

    with fits.open(track_fits_path) as hdul:
        track_data = hdul[1].data
        colnames   = list(track_data.names)

    # Drop the 'GroupID' bookkeeping column written by caesar_read_progen.
    snap_col_names = [c for c in colnames if c.upper() != 'GROUPID']

    # If column names are integer snapshot numbers (caesar_read_progen writes
    # them as e.g. '44', '45', …), sort ascending so track_array[:,i] always
    # aligns with the i-th snapshot in ascending order.
    try:
        snap_col_names = sorted(snap_col_names, key=int)
        _file_snaps = [int(c) for c in snap_col_names]
    except ValueError:
        _file_snaps = None  # column names are not integers (e.g. 'snap_000')

    track_array = np.column_stack(
        [np.asarray(track_data[c], dtype=np.int64) for c in snap_col_names]
    )  # (N_gals, N_snaps)

    # When sb is provided without explicit caesar_paths, build catalog_paths
    # from the file's own snapshot order so they always align.
    if caesar_paths is None and sb is not None:
        ref_snaps = _file_snaps if _file_snaps is not None else snaplist
        if ref_snaps is None:
            raise ValueError(
                "Provide either caesar_paths, snaplist, or a track file with "
                "integer-named columns."
            )
        catalog_paths = [sb.get_caesar_file(s) for s in ref_snaps]

    n_track_snaps = track_array.shape[1]
    if len(catalog_paths) != n_track_snaps:
        warnings.warn(
            f"snaplist has {len(catalog_paths)} entries but the track file "
            f"has {n_track_snaps} snapshot columns. "
            f"Processing only the first {min(len(catalog_paths), n_track_snaps)} snapshots.",
            stacklevel=2,
        )
        catalog_paths = catalog_paths[:n_track_snaps]

    galaxies = {}
    for gal_idx, track_row in enumerate(track_array):
        galaxy = Galaxy()
        galaxy.set_track(track_row)
        galaxies[gal_idx] = galaxy

    # Per-galaxy state for cross-snapshot companion matching.
    # Caesar catalog IDs are not stable across snapshots, so companions are
    # matched by minimum-image position within match_radius.
    # active[gal_id] = {local_id: {'pos': ndarray(3,), 'mass_ratio': float}}
    active_companions = {gal_id: {} for gal_id in galaxies}
    active_counters   = {gal_id: 0  for gal_id in galaxies}
    frag_radius = frag_radius_factor * match_radius

    for snap_idx, catalog_path in enumerate(catalog_paths):
        print(f"Processing snapshot {snap_idx}: {catalog_path}")
        t_start = time.time()

        with h5py.File(catalog_path, 'r') as f:
            pos   = f[_H5_POS][:]    # (N, 3)
            smass = f[_H5_SMASS][:]
            rhalf = f[_H5_RHALF][:]
            h2    = f[_H5_H2][:]
            dust  = f[_H5_DUST][:]

        n_cat = len(smass)
        cat_indices = np.arange(n_cat)

        for gal_id, galaxy in galaxies.items():
            main_index = int(galaxy.track[snap_idx])
            if main_index < 0 or main_index >= n_cat:
                continue

            main_mass = smass[main_index]
            if main_mass == 0:
                continue

            main_pos      = pos[main_index]
            search_radius = search_radius_factor * rhalf[main_index] * rhalf_unit_factor

            # Minimum-image convention for periodic boundary conditions
            dx = pos[:, 0] - main_pos[0]
            dy = pos[:, 1] - main_pos[1]
            dz = pos[:, 2] - main_pos[2]
            dx -= box_size * np.round(dx / box_size)
            dy -= box_size * np.round(dy / box_size)
            dz -= box_size * np.round(dz / box_size)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            mask = ((cat_indices != main_index)
                    & (smass > mass_threshold)
                    & (dist < search_radius))

            candidate_indices = np.where(mask)[0]
            active = active_companions[gal_id]

            # ── cross-snapshot positional matching ────────────────────────
            # For each candidate in this snapshot, find the nearest active
            # companion (from the previous snapshot) by minimum-image distance.
            # Matches within match_radius are continuations of the same physical
            # object; everything else is a first entry into the search sphere.
            matched_active_ids = set()
            new_entries = []

            for i in candidate_indices:
                comp_pos   = pos[i]
                mass_ratio = min(smass[i], main_mass) / max(smass[i], main_mass)

                best_id   = None
                best_dist = np.inf
                for ac_id, ac in active.items():
                    adx = comp_pos[0] - ac['pos'][0]
                    ady = comp_pos[1] - ac['pos'][1]
                    adz = comp_pos[2] - ac['pos'][2]
                    adx -= box_size * np.round(adx / box_size)
                    ady -= box_size * np.round(ady / box_size)
                    adz -= box_size * np.round(adz / box_size)
                    d = np.sqrt(adx**2 + ady**2 + adz**2)
                    if d < best_dist:
                        best_dist = d
                        best_id   = ac_id

                if best_id is not None and best_dist < match_radius:
                    # Continuation — update tracked position and mass ratio
                    matched_active_ids.add(best_id)
                    active[best_id]['pos']        = comp_pos.copy()
                    active[best_id]['mass_ratio'] = mass_ratio
                else:
                    new_entries.append((i, comp_pos, mass_ratio))

            # Expire companions that left the sphere or were absorbed
            for ac_id in [k for k in active if k not in matched_active_ids]:
                del active[ac_id]

            # ── first-entry and fragmentation detection ───────────────────
            # A new companion appearing within frag_radius of an already-active
            # companion that has a higher mass ratio is likely a tidal fragment
            # produced when FoF splits the infalling object — not a new merger.
            for i, comp_pos, mass_ratio in new_entries:
                is_fragment = False
                for ac in active.values():
                    adx = comp_pos[0] - ac['pos'][0]
                    ady = comp_pos[1] - ac['pos'][1]
                    adz = comp_pos[2] - ac['pos'][2]
                    adx -= box_size * np.round(adx / box_size)
                    ady -= box_size * np.round(ady / box_size)
                    adz -= box_size * np.round(adz / box_size)
                    if (np.sqrt(adx**2 + ady**2 + adz**2) < frag_radius
                            and ac['mass_ratio'] > mass_ratio):
                        is_fragment = True
                        break

                active_counters[gal_id] += 1
                new_id = active_counters[gal_id]
                active[new_id] = {'pos': comp_pos.copy(), 'mass_ratio': mass_ratio}

                progenitor = Progenitor(
                    mass=mass_ratio,
                    cat_index=int(i),
                    x=comp_pos[0],
                    y=comp_pos[1],
                    z=comp_pos[2],
                    merger=1,
                    fragmentation=int(is_fragment),
                    snapshot=snap_idx,
                    distance=dist[i],
                    processed=1,
                    H2=h2[i],
                    dust=dust[i],
                )
                galaxy.add_progenitor(progenitor)

            # Stored values reflect the last valid (lowest-z) snapshot since
            # catalog_paths is sorted ascending.
            galaxy.update_position(*main_pos)
            galaxy.update_mass(main_mass)
            galaxy.update_radii_stellar_half_mass(rhalf[main_index])

        t_end = time.time()
        print(f"Finished snapshot {snap_idx} in {t_end - t_start:.2f} sec")

    return galaxies


def analyze_mergers(galaxies, array_size, mass_threshold_maj, mass_threshold_min):
    """Classify recorded merger companions into major and minor merger counts.

    Only first-entry companions (``fragmentation == 0``) are counted.
    Tidal fragments flagged by :func:`process_galaxies_with_tracks` are skipped
    so that a single infalling object split by FoF is not counted multiple times.

    Parameters
    ----------
    galaxies : dict
        Output of :func:`process_galaxies_with_tracks`.
    array_size : tuple of int
        Shape ``(n_snaps, n_galaxies)`` for the output arrays.
    mass_threshold_maj : float
        Mass ratio above which a merger is classified as major (e.g. 0.25).
    mass_threshold_min : float
        Minimum mass ratio for a minor merger (e.g. 0.10). Companions below
        this threshold are ignored.

    Returns
    -------
    major_mergers, minor_mergers : ndarray of int, shape *array_size*
        Per-snapshot, per-galaxy counts — one entry per companion approach.
    """
    major_mergers = np.zeros(array_size, dtype=int)
    minor_mergers = np.zeros(array_size, dtype=int)

    for col, (galaxy_id, galaxy) in enumerate(galaxies.items()):
        for progenitor in galaxy._progenitors.values():
            if progenitor.merger != 1 or progenitor.fragmentation == 1:
                continue
            row = int(progenitor.snapshot)
            if row >= array_size[0]:
                continue
            if progenitor.mass >= mass_threshold_maj:
                major_mergers[row, col] += 1
            elif progenitor.mass >= mass_threshold_min:
                minor_mergers[row, col] += 1

    return major_mergers, minor_mergers
