"""Galaxy merger detection and classification from FITS catalogs.

Provides:

* :class:`Progenitor` – properties of a potential merger companion at one snapshot.
* :class:`Galaxy` – container for a galaxy's progenitor history across snapshots.
* :func:`process_galaxies_with_tracks` – scan snapshot catalogs for neighbours
  within a search radius and record them as merger candidates.
* :func:`analyze_mergers` – classify candidates into major/minor merger counts
  per galaxy per snapshot.

Typical usage::

    import simbanator as sb

    galaxies = sb.process_galaxies_with_tracks(
        fits_paths=fits_paths,
        track_fits_path="tracks.fits",
        box_size=100.0,           # Mpc/h
        search_radius_factor=5.0,
        mass_threshold=1e9,
        rhalf_unit_factor=1e-3,   # kpc/h → Mpc/h
    )

    major, minor = sb.analyze_mergers(
        galaxies,
        array_size=(n_snaps, n_galaxies),
        mass_threshold_maj=0.25,
        mass_threshold_min=0.10,
    )
"""

from astropy.io import fits
import numpy as np
import time


class Progenitor:
    """Properties of one merger companion galaxy recorded at a single snapshot.

    Parameters
    ----------
    mass : float
        Stellar mass ratio ``m_neighbour / m_main``.
    x, y, z : float
        Position of the companion in catalog units.
    merger : int
        1 if this entry represents a merger candidate, 0 otherwise.
    fragmentation : int
        1 if this entry represents a fragmentation event, 0 otherwise.
    snapshot : int
        Snapshot index at which this companion was recorded.
    H2 : float
        Molecular hydrogen mass of the companion.
    dust : float
        Dust mass of the companion.
    distance : float, optional
        Separation from the main galaxy in catalog units.
    processed : int, optional
        Flag marking whether this progenitor has been processed downstream.
    """

    def __init__(self, mass, x, y, z, merger, fragmentation, snapshot, H2, dust,
                 distance=0, processed=0):
        self.id = None
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
        print(f"Mass: {self.mass}")
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
        for existing in self._progenitors.values():
            if (progenitor.x, progenitor.y, progenitor.z) == (existing.x, existing.y, existing.z):
                print("A progenitor with the same position already exists. Progenitor not added.")
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


def process_galaxies_with_tracks(fits_paths, track_fits_path, box_size,
                                  search_radius_factor=5.0, mass_threshold=1e9,
                                  rhalf_unit_factor=1e-3, figure=False):
    """Scan FITS snapshot catalogs and record merger companions for each galaxy.

    For every snapshot, each tracked galaxy's neighbours within
    ``search_radius_factor * r_half`` are added as :class:`Progenitor` entries.
    Distances are computed with the minimum-image convention for the periodic box.

    Parameters
    ----------
    fits_paths : list of str
        Ordered list of per-snapshot FITS catalog paths. Index 0 is z=0.
    track_fits_path : str
        FITS file containing the progenitor track array, shape
        ``(N_galaxies, N_snaps)``.
    box_size : float
        Periodic box side length in the same units as ``pos_x/y/z`` (e.g. Mpc/h).
    search_radius_factor : float, optional
        Search radius expressed as a multiple of the stellar half-mass radius.
    mass_threshold : float, optional
        Minimum neighbour stellar mass to consider (in solar masses).
    rhalf_unit_factor : float, optional
        Factor to convert ``radii_stellar_half_mass`` to the position units.
        Default ``1e-3`` assumes radii in kpc/h and positions in Mpc/h.
    figure : bool, optional
        Reserved for future diagnostic plots; currently unused.

    Returns
    -------
    dict
        Mapping ``{galaxy_index: Galaxy}`` with progenitors recorded across
        all snapshots.
    """
    with fits.open(track_fits_path) as hdul:
        track_data = hdul[1].data

    galaxies = {}
    n_snaps = len(fits_paths)

    for gal_idx, track_row in enumerate(track_data):
        galaxy = Galaxy()
        galaxy.set_track(track_row)
        galaxies[gal_idx] = galaxy

    for snap_idx in range(n_snaps):
        catalog_path = fits_paths[snap_idx]
        print(f"Processing snapshot {snap_idx}: {catalog_path}")
        t_start = time.time()

        with fits.open(catalog_path) as hdul:
            cat = hdul[1].data
            cat_indices = np.arange(len(cat))

            for gal_id, galaxy in galaxies.items():
                main_index = galaxy.track[snap_idx]
                if main_index < 0 or main_index >= len(cat):
                    continue

                main = cat[main_index]
                main_mass = main['stellar_mass']
                if main_mass == 0:
                    continue

                main_pos = np.array([main['pos_x'], main['pos_y'], main['pos_z']])
                main_rhalf = main['radii_stellar_half_mass']
                search_radius = search_radius_factor * main_rhalf * rhalf_unit_factor

                # Minimum-image convention for periodic boundary conditions
                dx = cat['pos_x'] - main_pos[0]
                dy = cat['pos_y'] - main_pos[1]
                dz = cat['pos_z'] - main_pos[2]
                dx -= box_size * np.round(dx / box_size)
                dy -= box_size * np.round(dy / box_size)
                dz -= box_size * np.round(dz / box_size)
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                mask = ((cat_indices != main_index)
                        & (cat['stellar_mass'] > mass_threshold)
                        & (dist < search_radius))
                neighbors = cat[mask]
                neighbor_distances = dist[mask]

                for nb, d in zip(neighbors, neighbor_distances):
                    progenitor = Progenitor(
                        mass=nb['stellar_mass'] / main_mass,
                        x=nb['pos_x'],
                        y=nb['pos_y'],
                        z=nb['pos_z'],
                        merger=1,
                        fragmentation=0,
                        snapshot=snap_idx,
                        distance=d,
                        processed=1,
                        H2=nb['H2_mass'],
                        dust=nb['dust_mass'],
                    )
                    galaxy.add_progenitor(progenitor)

                if snap_idx == 0:
                    galaxy.update_position(main_pos[0], main_pos[1], main_pos[2])
                    galaxy.update_mass(main_mass)
                    galaxy.update_radii_stellar_half_mass(main_rhalf)

        t_end = time.time()
        print(f"Finished snapshot {snap_idx} in {t_end - t_start:.2f} sec")

    return galaxies


def analyze_mergers(galaxies, array_size, mass_threshold_maj, mass_threshold_min):
    """Classify recorded merger candidates into major and minor merger counts.

    Parameters
    ----------
    galaxies : dict
        Output of :func:`process_galaxies_with_tracks`.
    array_size : tuple of int
        Shape ``(n_snaps, n_galaxies)`` for the output arrays.
    mass_threshold_maj : float
        Mass ratio above which a merger is classified as major (e.g. 0.25).
    mass_threshold_min : float
        Minimum mass ratio for a minor merger (e.g. 0.10). Candidates below
        this threshold are ignored.

    Returns
    -------
    major_mergers, minor_mergers : ndarray of int, shape *array_size*
        Per-snapshot, per-galaxy merger counts.
    """
    major_mergers = np.zeros(array_size, dtype=int)
    minor_mergers = np.zeros(array_size, dtype=int)

    for col, (galaxy_id, galaxy) in enumerate(galaxies.items()):
        progs = galaxy._progenitors
        for progenitor in progs.values():
            if progenitor.merger != 1:
                continue
            row = int(progenitor.snapshot)
            if progenitor.fragmentation == 0:
                if progenitor.mass >= mass_threshold_maj:
                    major_mergers[row, col] += 1
                elif progenitor.mass >= mass_threshold_min:
                    minor_mergers[row, col] += 1
            elif progenitor.fragmentation == 1:
                same_snap = [
                    p for p in progs.values()
                    if p.snapshot == progenitor.snapshot
                    and p.merger == 1
                    and p.fragmentation == 1
                ]
                if len(same_snap) > 1:
                    major_mergers[row, col] += 1
                else:
                    minor_mergers[row, col] += 1

    return major_mergers, minor_mergers
