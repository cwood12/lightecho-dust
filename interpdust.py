"""
Functions needed to interpolate the g and Qsc value tables from Weingartner & Draine (2001); ApJ 548,296 and Draine & Hensley (2021); ApJ 909,94.
"""

# __all__ = [point_val,
        #    point_val_g,
        #    points_int,
        #    jori,
        #    lin2dinterp]
__author__ = 'Charlotte M. Wood'
__version__ = '0.2'


import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('pg.mplstyle')
import scipy.interpolate as interp

import dustconst as c
import grainsizedist as gsd


# parse & split Draine & Hensley (2021) data files based on grain orientation
def jori(model):
    """
    Splits data files from Draine & Hensley (2021); ApJ 909,94 into separate arrays of Qsc for each orientation using line values given in DH21_readme.txt. All dust property files, list of wavelengths, list of sizes, and readme can be found at http://arks.princeton.edu/ark:/88435/dsp01qb98mj541.

    Inputs:
    -----
    model: str
    Filename of desired dust model from Draine & Hensley (2021); ApJ 909,94


    Outputs:
    -----
    jori1: array-like
    2D array of Qsc values for orientation 1

    jori2: array-like
    2D array of Qsc values for orientation 2

    jori3: array-like
    2D array of Qsc values for orientation 3
    """

    # jori=1: k  ||  a, E perp a
    jori1 = model[6774:7903]
    # jori=2: k perp a, E  ||  a
    jori2 = model[7903:9032]
    # jori=3: k perp a, E perp a
    jori3 = model[9032:]

    return jori1, jori2, jori3   


# assign Qsc values to specific (wavelength, size) points
def point_val_Qsc(wave, size, jori):
    """
    Create arrays of points (wavelength, grain size) and values (Qsc) for interpolation with scipy.LinearNDInterpolator.


    Inputs:
    -----
    wave: array-like of floats
    1D array of wavelengths from DH21_wave.dat with units of cm.

    size: array-like of floats
    1D array of sizes from DH21_aeff.dat with units of cm.

    jori: array-like of floats
    2D array of Qsc values from the provided model table corresponding to a specific grain orientation. See Draine & Hensley (2021), ApJ, 909, 94.


    Outputs:
    -----
    points: Numpy array
    An array of (wavelength, size) points. Both wavelength and size are float values.

    values: Numpy array
    An array of Qsc values (float) corresponding to each (wavelength, size) point.
    """

    # combine wavelength and size into a meshgrid
    xx, yy = np.meshgrid(wave, size)
    # create an array of points from the meshgrid
    points = np.asarray([[xx[idx2,idx1], yy[idx2,idx1]] for idx1 in range(0,len(wave)) for idx2 in range(0,len(size))])
    # create an array of corresponding Qsc values
    values = np.asarray([jori[idx1][idx2] for idx1 in range(0,len(wave)) for idx2 in range(0,len(size))])

    return points, values


# assign g values to specific (wavelength, size) points
def point_val_g(wave, size, g):
    """
    Create arrays of points (wavelength, grain size) and values (g) for interpolation with scipy.LinearNDInterpolator.


    Inputs:
    -----
    wave: array-like of floats
    1D array of wavelengths from LD93_wave.dat with units of cm.

    size: array-like of floats
    1D array of sizes from LD93_aeff.dat with units of cm.

    g: array-like of floats
    1D array of g values from either Gra_81.dat or suvSil_81.dat. See Weingartner & Draine (2001), ApJ, 548, 296.


    Outputs:
    -----
    points: Numpy array
    An array of (wavelength, size) points. Both wavelength and size are float values.

    values: Numpy array
    An array of g values (float) corresponding to each (wavelength, size) point.
    """

    # combine wavelength and size into a meshgrid
    xx, yy = np.meshgrid(wave, size)
    # create an array of points from the meshgrid
    points = np.asarray([[xx[idx2,idx1], yy[idx2,idx1]] for idx1 in range(0,len(wave)) for idx2 in range(0,len(size))])
    # create an array of the corresponding g values
    values = np.asarray([g[idx1+(len(wave)*idx2)] for idx1 in range(0,len(wave)) for idx2 in range(0,len(size))])

    return points, values


# create a new array of points after performing the interpolation
def points_int(x, y):
    """
    Create a new array of points to calculate Qsc or g values across with interpolation. The new array contains all original points plus the intermediate, interpolated steps.

    
    Inputs:
    -----
    x: array-like
    Expanded array of wavelength (x) values.

    y: array-like
    Expanded array of size (y) values.


    Outputs:
    -----
    points: Numpy array
    An array of (x,y) points.
    """

    # combine x and y into a meshgrid
    xx, yy = np.meshgrid(x, y)
    # create an array of (x,y) points from the meshgrid
    points = np.asarray([[xx[idx2,idx1], yy[idx2,idx1]] for idx1 in range(0,len(x)) for idx2 in range(0,len(y))])

    return points