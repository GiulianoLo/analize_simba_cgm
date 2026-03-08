import os
import numpy as np
import h5py


def indices_to_ranges(idx):
    """
    Convert sorted integer indices into contiguous ranges.

    Example
    -------
    [10,11,12,20,21] → [(10,13), (20,22)]
    """

    if len(idx) == 0:
        return []

    ranges = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i != prev + 1:
            ranges.append((start, prev + 1))
            start = i
        prev = i

    ranges.append((start, prev + 1))

    return ranges


def read_ranges(dataset, idx):
    """
    Efficiently read arbitrary particle indices from an HDF5 dataset
    using contiguous range reads.
    """

    idx = np.asarray(idx)

    if len(idx) == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)

    idx = np.sort(idx)

    ranges = indices_to_ranges(idx)

    pieces = [dataset[a:b] for a, b in ranges]

    return np.concatenate(pieces, axis=0)




def extract_particles(
        cs,
        simfile,
        snap,
        galaxy_id=None,
        halo_id=None,
        center=None,
        radius=None,
        orientation="none",
        ptypes=("PartType0", "PartType4"),
        ignore_fields=None,
        output=None,
        verbose=1
):
    """
    Unified particle extraction from a simulation snapshot.

    Particles can be selected either from a CAESAR catalog object
    (galaxy or halo) or from a spatial aperture.

    Parameters
    ----------
    cs : caesar object
        Loaded CAESAR catalog.

    simfile : str
        Path to snapshot HDF5 file.

    snap : int
        Snapshot number.

    galaxy_id : int, optional
        Galaxy GroupID to extract.

    halo_id : int, optional
        Halo GroupID to extract.

    center : array-like (3), optional
        Spatial center [x,y,z].

    radius : float, optional
        Aperture radius (same units as snapshot coordinates).

    orientation : str
        'face-on' (default), 'edge-on', or 'none'.

    ptypes : tuple
        Particle types to include.

    ignore_fields : list
        Dataset names to skip.

    output : str
        Output HDF5 file path.

    verbose : int
        Verbosity level.
    """

    from ..utils.geometry import shrink_center, principal_axes, rotate_to_frame

    if ignore_fields is None:
        ignore_fields = []

    if output is None:
        output = f"filtered_snap{snap}.h5"

    if verbose:
        print("Reading snapshot:", simfile)

    with h5py.File(simfile, "r") as inp:

        particle_lists = {}

        # ---------------------------------------------------------
        # 1. Determine particle selection
        # ---------------------------------------------------------

        if galaxy_id is not None:

            gal = cs.galaxies[galaxy_id]

            plist_map = {
                "PartType0": gal.glist,
                "PartType1": getattr(gal, "dmlist", []),
                "PartType4": gal.slist,
                "PartType5": gal.bhlist,
            }

            for pt in ptypes:
                idx = np.array(plist_map.get(pt, []))
                if len(idx) > 0:
                    particle_lists[pt] = np.sort(idx)

        elif halo_id is not None:

            halo = cs.halos[halo_id]

            plist_map = {
                "PartType0": halo.glist,
                "PartType1": getattr(halo, "dmlist", []),
                "PartType4": halo.slist,
                "PartType5": halo.bhlist,
            }

            for pt in ptypes:
                idx = np.array(plist_map.get(pt, []))
                if len(idx) > 0:
                    particle_lists[pt] = np.sort(idx)

        elif center is not None and radius is not None:

            center = np.asarray(center)

            for pt in ptypes:

                if pt not in inp:
                    continue

                pos = inp[pt]["Coordinates"][:]

                r = np.sqrt(np.sum((pos - center) ** 2, axis=1))
                mask = r <= radius

                idx = np.where(mask)[0]

                if len(idx) > 0:
                    particle_lists[pt] = idx

        else:
            raise ValueError(
                "Must specify galaxy_id, halo_id, or (center + radius)"
            )

        if verbose:
            for pt in particle_lists:
                print(pt, "particles:", len(particle_lists[pt]))

        # ---------------------------------------------------------
        # 2. Determine orientation
        # ---------------------------------------------------------

        center_vec = None
        evecs = None

        if orientation != "none":

            if "PartType4" in particle_lists:

                idx = particle_lists["PartType4"]

                pos = read_ranges(inp["PartType4"]["Coordinates"], idx)
                mass = read_ranges(inp["PartType4"]["Masses"], idx)

                center_vec = shrink_center(pos, masses=mass)

                _, _, evecs, _ = principal_axes(pos - center_vec, masses=mass)

            else:

                pt0 = next(iter(particle_lists))
                pos = inp[pt0]["Coordinates"][particle_lists[pt0]]

                center_vec = pos.mean(axis=0)
                _, _, evecs, _ = principal_axes(pos - center_vec)

        # ---------------------------------------------------------
        # 3. Write output
        # ---------------------------------------------------------

        with h5py.File(output, "w") as out:

            inp.copy(inp["Header"], out, "Header")

            for pt, idx in particle_lists.items():

                out.create_group(pt)

                for k in inp[pt]:

                    if k in ignore_fields:
                        continue

                    dataset = inp[pt][k]

                    data = read_ranges(dataset, idx)

                    if k in ("Coordinates", "Velocities") and orientation != "none":

                        if k == "Coordinates":
                            data = data - center_vec

                        data = rotate_to_frame(data, 0, evecs)

                        if orientation == "edge-on":
                            data = data[:, [0, 2, 1]]

                    out[pt].create_dataset(k, data=data)

                n = len(idx)
                pindex = int(pt[-1])

                for attr in ("NumPart_ThisFile", "NumPart_Total"):

                    arr = out["Header"].attrs[attr].copy()
                    arr[pindex] = n
                    out["Header"].attrs[attr] = arr

    if verbose:
        print("Finished extraction →", output)