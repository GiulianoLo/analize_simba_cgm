import numpy as np

def Z_to_OH12(Z):
    """ Convert metallicities in oxygen abundance
    """
    logOH12 = np.log10(Z/0.0127)+8.69
    return logOH12

    
def Dust_to_Metal(M_dust, M_h2, abundance):
    """calculate dust-to-metal (DTM) ratio as in De Vis (2019): a systematic metallicity study of DustPedia
    """
    #metal fraction
    f_z = 27.36*(10**(abundance-12))
    #metal mass
    M_z = f_z*M_h2 + M_dust
    
    DTM = M_dust/M_z
    
    return DTM
