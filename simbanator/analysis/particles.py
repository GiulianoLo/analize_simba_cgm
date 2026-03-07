"""Filter simulation particles by Caesar objects or spatial apertures.

Reduces full snapshot HDF5 files to subsets containing only
the particles belonging to specific galaxies/halos or within
a spherical region.
"""

import os
import numpy as np
import h5py
import unyt  # noqa: F401 (available for downstream use)
from unyt import unyt_array, kpc  # noqa: F401

from ..io.paths import OutputPaths


def copy_skeleton(src_filename, dst_filename):
    """Copy the group/dataset structure of an HDF5 file without data.

    This ensures the reduced file can still be interpreted by yt.
    """
    with h5py.File(src_filename, 'r') as src_file:
        with h5py.File(dst_filename, 'w') as dst_file:
            def copy_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    dst_file.create_group(name)
                elif isinstance(obj, h5py.Dataset):
                    dst_file.create_dataset(name, shape=obj.shape, dtype=obj.dtype)
                for key, value in obj.attrs.items():
                    dst_file[name].attrs[key] = value
            src_file.visititems(copy_structure)


def filter_particles_by_obj(cs, simfile, snap, selection, oidx,
                            verbose=0, overwrite=True, ignore_fields=None,
                            keyword=None, output_dir=None):
    """Save a subset of particles belonging to specific galaxies or halos.

    Parameters
    ----------
    cs : caesar object
        Loaded Caesar catalog.
    simfile : str
        Path to the full snapshot HDF5 file.
    snap : int
        Snapshot number.
    selection : str
        ``'galaxy'`` or ``'halo'``.
    oidx : list of int
        Object indices to extract.
    verbose : int
        Verbosity level (0, 1, 2).
    overwrite : bool
        If *False*, a *keyword* must be supplied for the output name.
    ignore_fields : list of str, optional
        Dataset names to skip.
    keyword : str, optional
        Extra tag appended to output filenames.
    output_dir : str, optional
        Directory for output files.  Defaults to
        ``./output/filtered_particles/``.
    """
    if ignore_fields is None:
        ignore_fields = []

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output', 'filtered_particles')
    out_folder = os.path.join(output_dir, f'snap_{int(snap):03d}')
    os.makedirs(out_folder, exist_ok=True)

    if selection not in ('galaxy', 'halo'):
        raise ValueError("selection must be 'galaxy' or 'halo'")
    if not overwrite and keyword is None:
        raise ValueError('keyword must be provided if overwrite is False')

    with h5py.File(simfile, 'r') as input_file:
        for obj in oidx:
            # Build output filename
            tag = f'{selection[0:3]}{obj:06.0f}'
            if keyword is None:
                output = os.path.join(out_folder, f'subset_snap{snap}_{tag}.h5')
            else:
                output = os.path.join(out_folder, f'subset_snap{snap}_{tag}_{keyword}.h5')

            copy_skeleton(simfile, output)

            with h5py.File(output, 'a') as output_file:
                for ptype in ('PartType0', 'PartType1', 'PartType4', 'PartType5'):
                    if ptype not in input_file:
                        continue
                    if verbose > 0:
                        print('Processing:', ptype)

                    _o = (cs.galaxies[obj] if selection == 'galaxy'
                          else cs.halos[obj])

                    plist_map = {
                        'PartType0': _o.glist,
                        'PartType1': _o.dmlist if hasattr(_o, 'dmlist') else _o.glist,
                        'PartType4': _o.slist,
                        'PartType5': _o.bhlist,
                    }
                    plist = plist_map.get(ptype)
                    if plist is None:
                        continue

                    sorted_plist = np.sort(plist)

                    for k in input_file[ptype]:
                        if k in ignore_fields:
                            if verbose > 1:
                                print(k, 'skipped...')
                            continue
                        if verbose > 0:
                            print(ptype, k)

                        # Read only the needed rows directly from HDF5
                        filtered_dset = input_file[ptype][k][sorted_plist]

                        if k in output_file[ptype]:
                            del output_file[ptype][k]
                        output_file[ptype][k] = filtered_dset

                    # Update header particle counts
                    for attr in ('NumPart_ThisFile', 'NumPart_Total'):
                        temp = output_file['Header'].attrs[attr]
                        temp[int(ptype[8:])] = len(plist)
                        output_file['Header'].attrs[attr] = temp

                    if verbose > 0:
                        print("Updated Header:", dict(output_file['Header'].attrs))

    print('Finished with particle filters')


def filter_by_aperture(cs, simfile, snap, center, radius,
                       selection=None, verbose=0, overwrite=True,
                       ignore_fields=None, keyword=None, closegal=None,
                       output_dir=None):
    """Filter particles within a spherical aperture.

    Parameters
    ----------
    cs : caesar object
        Loaded Caesar catalog.
    simfile : str
        Path to the full snapshot HDF5 file.
    snap : int
        Snapshot number.
    center : array-like or int
        ``[x, y, z]`` coordinates or an object ID (if *selection* is set).
    radius : float or list
        Aperture radius in kpc, or ``[r_inner, r_outer]`` for an annulus.
    selection : str, optional
        ``'galaxy'`` or ``'halo'`` when *center* is an object ID.
    verbose : int
        Verbosity level.
    overwrite : bool
        If *False*, a *keyword* must be supplied.
    ignore_fields : list of str, optional
        Dataset names to skip.
    keyword : str, optional
        Extra tag for the output filename.
    closegal : int, optional
        ID used in the output filename.
    output_dir : str, optional
        Directory for output files.  Defaults to
        ``./output/filtered_particles/``.
    """
    if ignore_fields is None:
        ignore_fields = []

    if verbose > 0:
        print(f'Analyzing particle file {simfile}')

    a = cs.simulation.scale_factor

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output', 'filtered_particles')
    out_folder = os.path.join(output_dir, f'snap_{int(snap):03d}')
    os.makedirs(out_folder, exist_ok=True)

    if not overwrite and keyword is None:
        raise ValueError('keyword must be provided if overwrite is False')

    # Build output filename
    r_tag = radius if not isinstance(radius, list) else f'{radius[0]}-{radius[1]}'
    sel_tag = selection or 'region'
    gal_tag = f'{closegal:06.0f}' if closegal is not None else 'all'
    base = f'region_snap{snap:03d}_r{r_tag}_{sel_tag}{gal_tag}'
    if keyword is not None:
        base += f'_{keyword}'
    output = os.path.join(out_folder, f'{base}.h5')

    # Resolve center coordinates
    if isinstance(center, (list, np.ndarray)) and len(center) == 3:
        center_coord = np.array(center)
    elif isinstance(center, int) and selection in ('galaxy', 'halo'):
        if selection == 'galaxy':
            center_coord = cs.galaxies[center].pos
        else:
            center_coord = cs.halos[center].pos
    else:
        raise ValueError(
            'center must be [x,y,z] or an int with a valid selection'
        )

    with h5py.File(simfile, 'r') as input_file:
        if verbose > 0:
            print('Copying file structure')
        copy_skeleton(simfile, output)
        if verbose > 0:
            print('Finished copying file structure')

        with h5py.File(output, 'a') as output_file:
            for ptype in ('PartType0', 'PartType1', 'PartType4', 'PartType5'):
                if ptype not in input_file:
                    continue
                if verbose > 0:
                    print('Selecting particles for:', ptype)

                if 'Coordinates' not in input_file[ptype]:
                    if verbose > 0:
                        print(f'Coordinates not found in {ptype}')
                    continue

                pos = input_file[ptype]['Coordinates'][:]
                if verbose > 0:
                    print(f'Aperture: {radius} kpc → {radius / a if not isinstance(radius, list) else [r / a for r in radius]} ckpc  centered at {center_coord}')

                distances = np.sqrt(np.sum((pos - center_coord) ** 2, axis=1))

                if isinstance(radius, list):
                    if verbose > 0:
                        print('Annulus mode')
                    mask = (distances >= radius[0] / a) & (distances <= radius[1] / a)
                else:
                    mask = distances <= radius / a

                filtered_positions = pos[mask]
                if len(filtered_positions) < 1:
                    print('SelectionError: No particles found within the aperture')
                    continue

                for k in input_file[ptype]:
                    if k in ignore_fields:
                        continue
                    if verbose > 0:
                        print(ptype, k)

                    # Read only the needed rows directly from HDF5
                    filtered_dset = input_file[ptype][k][mask]

                    if k in output_file[ptype]:
                        del output_file[ptype][k]
                    output_file[ptype][k] = filtered_dset

                for attr in ('NumPart_ThisFile', 'NumPart_Total'):
                    temp = output_file['Header'].attrs[attr]
                    temp[int(ptype[8:])] = len(filtered_positions)
                    output_file['Header'].attrs[attr] = temp

                if verbose > 0:
                    print("Updated Header:", dict(output_file['Header'].attrs))

    print('Finished with aperture filter')


# ---------------------------------------------------------------------------
# Oriented extraction (face-on / edge-on)
# ---------------------------------------------------------------------------

def extract_galaxy_oriented(cs, simfile, snap, galaxy_ids,
                            output_dir=None, verbose=0,
                            ignore_fields=None,
                            ptypes=('PartType0', 'PartType4')):
    """Extract galaxy particles and rotate into face-on / edge-on frames.

    For each galaxy two HDF5 files are produced:

    * ``galaxy_<ID>_snap_<N>_face.h5`` — face-on (xy = disk plane)
    * ``galaxy_<ID>_snap_<N>_edge.h5`` — edge-on (xz swap of face-on)

    The galaxy centre is found with an iterative shrinking-sphere
    algorithm on the stellar component.  Positions and velocities
    of **all** requested particle types are rotated into the
    principal-axis frame of the stellar mass distribution.

    Parameters
    ----------
    cs : caesar object
        Loaded Caesar catalog.
    simfile : str
        Path to the full snapshot HDF5 file.
    snap : int
        Snapshot number.
    galaxy_ids : int or list of int
        Galaxy GroupID(s) to extract.
    output_dir : str, optional
        Directory for output files.  Defaults to
        ``./output/filtered_particles/oriented/``.
    verbose : int
        Verbosity level (0, 1, 2).
    ignore_fields : list of str, optional
        Dataset names to skip entirely.
    ptypes : tuple of str
        Particle types to include (default gas + stars).

    Notes
    -----
    Coordinates and Velocities are centred and rotated; all other
    datasets are copied verbatim.  The face-on frame has the disk
    in the *xy*-plane; the edge-on frame swaps *y* ↔ *z*.
    """
    from ..utils.geometry import shrink_center, principal_axes, rotate_to_frame

    if isinstance(galaxy_ids, (int, np.integer)):
        galaxy_ids = [int(galaxy_ids)]
    if ignore_fields is None:
        ignore_fields = []

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output',
                                  'filtered_particles', 'oriented')
    out_folder = os.path.join(output_dir, f'snap_{int(snap):03d}')
    os.makedirs(out_folder, exist_ok=True)

    plist_key = {
        'PartType0': 'glist',
        'PartType1': 'dmlist',
        'PartType4': 'slist',
        'PartType5': 'bhlist',
    }

    with h5py.File(simfile, 'r') as inp:
        for gal_id in galaxy_ids:
            gal = cs.galaxies[gal_id]

            face_path = os.path.join(
                out_folder, f'galaxy_{gal_id}_snap_{snap}_face.h5')
            edge_path = os.path.join(
                out_folder, f'galaxy_{gal_id}_snap_{snap}_edge.h5')

            # --- resolve particle index lists per type -------------------
            plists = {}
            for pt in ptypes:
                attr = plist_key.get(pt)
                if attr is None or not hasattr(gal, attr):
                    continue
                idx = np.sort(getattr(gal, attr))
                if len(idx) == 0:
                    continue
                plists[pt] = idx

            if not plists:
                if verbose > 0:
                    print(f'Galaxy {gal_id}: no particles found, skipping')
                continue

            # --- find centre & principal axes from stars -----------------
            star_pt = 'PartType4'
            if star_pt in plists and 'Coordinates' in inp[star_pt]:
                slist = plists[star_pt]
                star_pos = inp[star_pt]['Coordinates'][slist]
                star_mass = inp[star_pt]['Masses'][slist]
                center = shrink_center(star_pos, masses=star_mass)
                _, _, evecs, _ = principal_axes(
                    star_pos - center, masses=star_mass)
            else:
                # fallback: use first available ptype
                pt0 = next(iter(plists))
                pos0 = inp[pt0]['Coordinates'][plists[pt0]]
                center = pos0.mean(axis=0)
                _, _, evecs, _ = principal_axes(pos0 - center)

            if verbose > 0:
                print(f'Galaxy {gal_id}: centre = {center}')

            # --- create output files with Header ------------------------
            with (h5py.File(face_path, 'w') as f_face,
                  h5py.File(edge_path, 'w') as f_edge):

                for fout in (f_face, f_edge):
                    inp.copy(inp['Header'], fout, 'Header')
                    for pt in plists:
                        fout.create_group(pt)

                # --- fill each particle type ----------------------------
                for pt, idx in plists.items():
                    if verbose > 0:
                        print(f'  {pt}: {len(idx)} particles')

                    for k in inp[pt]:
                        if k in ignore_fields:
                            continue
                        data = inp[pt][k][idx]

                        if k in ('Coordinates', 'Velocities'):
                            if k == 'Coordinates':
                                data_centered = data - center
                            else:
                                data_centered = data  # velocities: no shift
                            rotated = rotate_to_frame(
                                data_centered, 0, evecs)

                            face_data = np.asarray(rotated, dtype=float)
                            edge_data = face_data[:, [0, 2, 1]]  # swap y ↔ z

                            f_face[pt].create_dataset(k, data=face_data)
                            f_edge[pt].create_dataset(k, data=edge_data)
                        else:
                            f_face[pt].create_dataset(k, data=data)
                            f_edge[pt].create_dataset(k, data=data)

                    # update header counts
                    n = len(idx)
                    ptype_idx = int(pt[-1])
                    for fout in (f_face, f_edge):
                        for attr in ('NumPart_ThisFile', 'NumPart_Total'):
                            arr = fout['Header'].attrs[attr].copy()
                            arr[ptype_idx] = n
                            fout['Header'].attrs[attr] = arr

                    if verbose > 0:
                        print(f'    → wrote {face_path}')
                        print(f'    → wrote {edge_path}')

    print(f'Finished oriented extraction for {len(galaxy_ids)} galaxy(ies)')

