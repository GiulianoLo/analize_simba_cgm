"""Compute radial profiles of galaxy properties from simulation snapshots."""

import numpy as np
import yt
import caesar


def radial_profile(snapfile, catfile, galaxy_id, properties_dict,
                   radii=None, region=False, dens=False, dim=None, norm=True):
    """Compute radial profiles of specified particle properties.

    Parameters
    ----------
    snapfile : str
        Path to the simulation snapshot file.
    catfile : str
        Path to the Caesar catalog file.
    galaxy_id : int
        Galaxy GroupID in the Caesar catalog.
    properties_dict : dict
        ``{particle_type: [property_names]}``
        e.g. ``{'PartType0': ['Masses', 'Metallicity']}``.
    radii : array-like, optional
        Radial bin edges in kpc (default ``np.arange(0, 100, 1)``).
    region : bool
        If *True*, use all particles; otherwise use only those
        belonging to the galaxy.
    dens : bool
        If *True*, compute surface density; otherwise compute mean.
    dim : dict, optional
        Unit conversion mapping,
        e.g. ``{'PartType0': ['code_mass', 'Msun']}``.
    norm : bool
        Normalise profiles to [0, 1].

    Returns
    -------
    tuple
        ``(radii, profiles)`` where *profiles* is a nested dict
        ``{ptype: {property: list}}``.
    """
    if radii is None:
        radii = np.arange(0, 100, 1)

    ds = yt.load(snapfile)
    obj = caesar.load(catfile)
    ad = ds.all_data()
    gal = [i for i in obj.galaxies if i.GroupID == galaxy_id][0]
    center = gal.pos.in_units('kpc').value

    def get_data(particle_type, props, dim_info, indices=None):
        pos = ad[particle_type, 'Coordinates'].in_units('kpc').value
        if indices is not None:
            pos = pos[indices]
        data_list = []
        for prop in props:
            data = ad[particle_type, prop[:-2] if '_s' in prop else prop]
            if indices is not None:
                data = data[indices]
            data = ds.arr(data, dim_info[particle_type][0]).in_units(dim_info[particle_type][1]).value
            data_list.append(data)
        return pos, data_list

    profiles = {
        ptype: {prop: [] for prop in properties_dict[ptype]}
        for ptype in properties_dict
    }

    for ptype, props in properties_dict.items():
        indices = None
        if not region:
            if ptype == 'PartType0':
                indices = gal.glist
            elif ptype == 'PartType4':
                indices = gal.slist
        pos, data = get_data(ptype, props, dim, indices=indices)

        radial_distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

        for r in radii:
            mask = radial_distances < r
            A = np.pi * r ** 2

            for i, prop in enumerate(props):
                if dens:
                    profiles[ptype][prop].append(np.sum(data[i][mask]) / A)
                else:
                    profiles[ptype][prop].append(np.mean(data[i][mask]))

    if norm:
        for ptype in profiles:
            for prop in profiles[ptype]:
                if len(profiles[ptype][prop]) > 0:
                    temp = np.array(profiles[ptype][prop])
                    ptp = np.nanmax(temp) - np.nanmin(temp)
                    if ptp > 0:
                        profiles[ptype][prop] = (temp - np.nanmin(temp)) / ptp
                    else:
                        profiles[ptype][prop] = temp
    else:
        for ptype in profiles:
            for prop in profiles[ptype]:
                profiles[ptype][prop] = np.array(profiles[ptype][prop])

    return radii, profiles
