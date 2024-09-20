import yt
import numpy as np
import caesar

def radial_profile(snapfile, catfile, galaxy_id, properties_dict,\
                   radii=np.arange(0, 100, 1), region=False, dens=False, dim=None, norm=True):
    """
    Load snapshot data and return radial profiles of specified properties, always loading coordinates.
    
    Parameters:
    - snapfile: path to the simulation snapshot file.
    - catfile: path to the Caesar catalog file.
    - galaxy_id: the index of the galaxy in the Caesar catalog.
    - properties_dict: dictionary where keys are particle types (e.g., 'PartType0') and values are lists of properties (e.g., ['Masses', 'Metallicity']).
    - radii: range of radii for which to compute profiles (default: np.arange(0, 100, 1)).
    - region: Boolean to compute properties for a larger region or just the galaxy itself.
    - dim: dictionary where keys are particle types and values are lists of unit conversions for each property
      (e.g., {'PartType0': {'Masses': ['code_mass', 'Msun'], 'Metallicity': ['dimensionless', 'dimensionless']}}).
    
    Returns:
    - radii: array of radial distances.
    - profiles: dictionary containing radial profiles for each particle type and property.
    """
    ds = yt.load(snapfile)
    obj = caesar.load(catfile)
    ad = ds.all_data()
    gal = [i for i in obj.galaxies if i.GroupID == galaxy_id][0]
    center = gal.pos.in_units('kpc').value

    def get_data(particle_type, props, dim, indices=None):
        pos = ad[particle_type, 'Coordinates'].in_units('kpc').value
        if indices is not None:
            pos = pos[indices]
        data_list = []
        for prop in props:
            data = ad[particle_type, prop[:-2] if '_s' in prop else prop]
            if indices is not None:
                data = data[indices]
           
            # Convert property data using the dimensional units before and after conversion
            #print(particle_type)
            #print(dim[particle_type][0], dim[particle_type][1])
            data = ds.arr(data, dim[particle_type][0]).in_units(dim[particle_type][1]).value
            
            data_list.append(data)
    
        return pos, data_list

    # Prepare to store profiles
    profiles = {ptype: {prop: [] for prop in properties_dict[ptype]} for ptype in properties_dict}

    # Load coordinates and relevant properties for each particle type
    if region:
        for ptype, props in properties_dict.items():
            if ptype == 'PartType0':  # Gas particles
                pos, data = get_data(ptype, props, dim, indices=None)
            elif ptype == 'PartType4':  # Star particles
                pos, data = get_data(ptype, props, dim, indices=None)

            # Calculate radial distances from galaxy center
            radial_distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))

            # Calculate radial profiles for the current particle type and properties
            for r in radii:
                mask = radial_distances < r
                A = np.pi * r ** 2
                
                for i, prop in enumerate(props):
                    if dens:
                        profiles[ptype][prop].append(sum(data[i][mask]) / A)  # Surface density
                    else:
                        profiles[ptype][prop].append(np.mean(data[i][mask]))  # Mean property like metallicity
        
    
    else:
        for ptype, props in properties_dict.items():
            if ptype == 'PartType0':  # Gas particles
                pos, data = get_data(ptype, props, dim, indices=gal.glist)
            elif ptype == 'PartType4':  # Star particles
                pos, data = get_data(ptype, props, dim, indices=gal.slist)

            # print(ptype, pos)
            # print(ptype, data)
            # plt.figure()
            # plt.scatter(pos[:,0], pos[:,1])
            # plt.figure()
            # plt.hist(data)
            
            # Calculate radial distances from galaxy center
            radial_distances = np.sqrt(np.sum((pos - center) ** 2, axis=1))
            
            # Calculate radial profiles for the current particle type and properties
            for r in radii:
                mask = radial_distances < r
                A = np.pi * r ** 2
                
                for i, prop in enumerate(props):
                    if dens:
                        profiles[ptype][prop].append(sum(data[i][mask]) / A)  # Surface density
                    else:
                        profiles[ptype][prop].append(np.mean(data[i][mask]))  # Mean property like metallicity

    #Normalize the profiles
    if norm:
        for ptype in profiles:
            for prop in profiles[ptype]:
                if len(profiles[ptype][prop]) > 0:
                    temp = np.array(profiles[ptype][prop])
                    profiles[ptype][prop] = (temp - np.nanmin(temp)) / (np.nanmax(temp) - np.nanmin(temp))
    else:
        for prop in profiles[ptype]:
            if len(profiles[ptype][prop]) > 0:
                profiles[ptype][prop] = np.array(profiles[ptype][prop])

    return radii, profiles

