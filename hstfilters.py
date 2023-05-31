"""
Functions needed to extract wavelength and throughput information for a desired HST WFC3/UVIS1 filter.

https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs
"""

# __all__ = [point_val,
        #    point_val_g,
        #    points_int,
        #    jori,
        #    lin2dinterp]
__author__ = 'Charlotte M. Wood'
__version__ = '0.1'


import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('pg.mplstyle')
import scipy.interpolate as interp
import astropy.units as u

import dustconst as c
import grainsizedist as gsd

# extract wavelength and throughput values for a given HST WFC3 UVIS filter
def throughput(filter):
    """
    Extracts wavelengths and throughputs for a given HST WFC3/UVIS1 filter. Data files can be found at https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs.

    Inputs:
    -----
    filter: str
    The name of the desired filter. Example format: 'f350lp', 'f555w', etc.


    Outputs:
    -----
    hstwave: array-like
    Array of wavelengths from HST filter data files, converted from angstroms to cm.

    throughput: array-like
    Array of throughput values corresponding to each wavelength.
    """

    # define the filename which holds the desired data
    filename = 'UVIS/wfc3_uvis1_{filter:s}.txt'

    # put wavelengths, throughputs into arrays
    wavelength = np.loadtxt(filename, usecols=(0), unpack=True) * u.AA
    hstwave = wavelength.to(u.cm)
    throughput = np.loadtxt(filename, usecols=(1), unpack=True)

    return hstwave, throughput