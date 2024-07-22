def findsatellites(pos_c, sb, snap, r=30):
    """find indexes of galaxies at a distance d<r from a given center position
       using caesar files

       Args:
           pos_c (dictionary): central position for the search (key is the id of the galaxy or any ID)
           sb (simba class): deals with the paths and caesar files
           snap (int): snapshot of the simulation
           r (float): search radius in Kpc

       Return:
           dictionary of indexes for satellites (one key for each of the original dictionary)
    """
    ids_out = {}
    cs = sb.get_caesar(snap)
    h = cs.simulation.scale_factor
    ids = np.asarray([i.GroupID for i in cs.galaxies])
    pos_l = np.asarray([i.pos for i in cs.galaxies])
    pos_cent = pos_c.copy()*h
    pos_sim  = pos_l.copy()*h
    for key, p in enumerate(pos_c):
        dist = np.sqrt((pos_l[:,0]-p[0])**2+(pos_l[:,1]-p[1])**2+(pos_l[:,2]-p[2])**2)
        ids_out[key] = ids[np.where(dist<r)[0]]
    return ids_out