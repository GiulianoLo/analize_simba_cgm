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
    galaxy_ids=None,
        halo_id=None,
        center=None,
        radius=None,
        orientation="none",
        ptypes=("PartType0", "PartType4"),
        ignore_fields=None,
        output=None,
        sim_name=None,
    prefix="",
    verbose=1,
    overwrite=False,
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

    galaxy_ids : iterable of int, optional
        List of galaxy GroupIDs to extract in one pass from a single open
        snapshot file. When provided, one output file is created per galaxy.

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

    output : str, optional
        Full output HDF5 file path.  If omitted, the file is saved as
        ``output/filtered/<sim_name>_snap<snap>_gal<id>.h5`` (or
        ``halo<id>`` / ``aperture`` for the other selection modes).

    sim_name : str, optional
        Simulation label used for the default output filename.  Derived
        from *simfile* when not provided.

    prefix : str
        Prefix prepended to the default output filename.

    verbose : int
        Verbosity level.

    overwrite : bool, optional
        If True, overwrite existing output files. If False (default),
        existing files are kept unchanged and skipped.
    """

    from ..utils.geometry import shrink_center, principal_axes, rotate_to_frame

    def _build_particle_lists_from_obj(obj):
        plist_map = {
            "PartType0": obj.glist,
            "PartType1": getattr(obj, "dmlist", []),
            "PartType4": obj.slist,
            "PartType5": obj.bhlist,
        }

        particle_lists_local = {}
        for pt in ptypes:
            idx_local = np.array(plist_map.get(pt, []))
            if len(idx_local) > 0:
                particle_lists_local[pt] = np.sort(idx_local)

        return particle_lists_local

    def _write_output(inp, particle_lists_local, outpath):
        center_vec_local = None
        evecs_local = None

        if orientation != "none":

            if "PartType4" in particle_lists_local:

                idx_local = particle_lists_local["PartType4"]

                pos_local = read_ranges(inp["PartType4"]["Coordinates"], idx_local)
                mass_local = read_ranges(inp["PartType4"]["Masses"], idx_local)

                center_vec_local = shrink_center(pos_local, masses=mass_local)

                _, _, evecs_local, _ = principal_axes(pos_local - center_vec_local, masses=mass_local)

            else:

                pt0_local = next(iter(particle_lists_local))
                pos_local = inp[pt0_local]["Coordinates"][particle_lists_local[pt0_local]]

                center_vec_local = pos_local.mean(axis=0)
                _, _, evecs_local, _ = principal_axes(pos_local - center_vec_local)

        write_mode = "w" if overwrite else "x"
        with h5py.File(outpath, write_mode) as out:

            inp.copy(inp["Header"], out, "Header")

            for pt, idx_local in particle_lists_local.items():

                out.create_group(pt)

                for k in inp[pt]:

                    if k in ignore_fields:
                        continue

                    dataset = inp[pt][k]

                    data = read_ranges(dataset, idx_local)

                    if k in ("Coordinates", "Velocities") and orientation != "none":

                        if k == "Coordinates":
                            data = data - center_vec_local

                        data = rotate_to_frame(data, 0, evecs_local)

                        if orientation == "edge-on":
                            data = data[:, [0, 2, 1]]

                    out[pt].create_dataset(k, data=data)

                n = len(idx_local)
                pindex = int(pt[-1])

                for attr in ("NumPart_ThisFile", "NumPart_Total"):

                    arr = out["Header"].attrs[attr].copy()
                    arr[pindex] = n
                    out["Header"].attrs[attr] = arr

        if verbose:
            print("Finished extraction ->", outpath)

        return outpath

    if ignore_fields is None:
        ignore_fields = []

    if galaxy_ids is not None and galaxy_id is not None:
        raise ValueError("Provide either galaxy_id or galaxy_ids, not both")

    if galaxy_ids is not None and (halo_id is not None or center is not None or radius is not None):
        raise ValueError("galaxy_ids mode cannot be combined with halo/aperture selection")

    if sim_name is None:
        # Derive from the simfile basename, stripping snap-number suffix
        sim_name = os.path.splitext(os.path.basename(simfile))[0]

    out_dir = os.path.join(os.getcwd(), 'output', sim_name, 'filtered', f'snap_{snap:03}')
    os.makedirs(out_dir, exist_ok=True)

    if galaxy_ids is not None:
        if output is not None:
            raise ValueError("output is not supported with galaxy_ids; one file is written per galaxy")

        galaxy_ids = list(galaxy_ids)
        outputs = []

        if verbose:
            print("Reading snapshot once for batch extraction:", simfile)

        with h5py.File(simfile, "r") as inp:
            for gid in galaxy_ids:
                gid = int(gid)
                outpath = os.path.join(out_dir, f'{prefix}_snap{snap:03}_gal{gid:06}.h5')

                if os.path.exists(outpath) and not overwrite:
                    if verbose:
                        print("Output exists, skipping (overwrite=False):", outpath)
                    outputs.append(outpath)
                    continue

                gal = cs.galaxies[gid]
                particle_lists = _build_particle_lists_from_obj(gal)

                if verbose:
                    for pt in particle_lists:
                        print(f"gal {gid} {pt} particles:", len(particle_lists[pt]))

                outputs.append(_write_output(inp, particle_lists, outpath))

        return outputs

    if output is None:
        if galaxy_id is not None:
            sel_tag = f'gal{galaxy_id}'
        elif halo_id is not None:
            sel_tag = f'halo{halo_id}'
        else:
            sel_tag = 'aperture'
        output = os.path.join(out_dir, f'{prefix}_snap{snap:03}_{sel_tag}.h5')

    if os.path.exists(output) and not overwrite:
        if verbose:
            print("Output exists, skipping (overwrite=False):", output)
        return output

    if verbose:
        print("Reading snapshot:", simfile)

    with h5py.File(simfile, "r") as inp:

        particle_lists = {}

        # ---------------------------------------------------------
        # 1. Determine particle selection
        # ---------------------------------------------------------

        if galaxy_id is not None:
            gal = cs.galaxies[galaxy_id]
            particle_lists = _build_particle_lists_from_obj(gal)

        elif halo_id is not None:

            halo = cs.halos[halo_id]
            particle_lists = _build_particle_lists_from_obj(halo)

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

        return _write_output(inp, particle_lists, output)