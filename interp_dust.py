"""
Functions needed to interpolate the g and Qsc value tables from Weingartner & Draine (2001); ApJ 548,296 and Draine & Hensley (2021); ApJ 909,94.
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

import dustconst as c
import grainsizedist as gsd


# assign Qsc values to specific (wavelength, size) points
def point_val(jori, wave, size):
    """
    Create arrays of points (wavelength, grain size) and values (Qsc) for interpolation.


    Inputs:
    -----
    jori: array-like
    2D array of Qsc values read in from table corresponding to a specific grain orientation. See Draine & Hensley (2021); ApJ 909,94.

    wave: array-like
    Array of wavelengths used for Qsc values, in cm

    size: array-like
    Array of sizes used for Qsc values, in cm


    Outputs:
    -----
    points: array-like
    Array of (wavelength, size) points

    values: array-like
    Array of Qsc values corresponding to each (wavelength, size) point
    """

    # set blank arrays for points and values to be appended to later
    points = []
    values = []

    # iterate over each wavelength
    for idx1 in range(0, len(wave)):
        # iterate over each size
        for idx2 in range(0, len(size)):
            # create the (wavelength, size) point and append to points array
            points.append([wave[idx1], size[idx2]])
            # find the corresponding Qsc value and append to values array
            values.append(jori[idx1][idx2])

    return points, values


# assign g values to specific (wavelength, size) points
def point_val_g(wave, size, g):
    """
    Create arrays of points (wavelength, grain size) and values (g) for interpolation.


    Inputs:
    -----
    wave: array-like
    Array of wavelengths used for g values, in cm

    size: array-like
    Array of sizes used for g values, in cm

    g: array-like
    Array of g values read in from table. See Weingartner & Draine (2001); ApJ 548,296


    Outputs:
    -----
    points: array-like
    Array of (wavelength, size) points

    values: array-like
    Array of g values corresponding to each (wavelength, size) point
    """

    # set blank arrays for points and values to be appended to later
    points = []
    values = []

    # iterate over each size
    for idx2 in range(0, len(size)):
        # iterate over each wavelength
        for idx1 in range(0, len(wave)):

            # create (wavelength, size) point
            point = [wave[idx1], size[idx2]]
            # find the corresponding g value index
            val_idx = idx1 + (len(wave)*idx2)
            # pull out the g value for the correct index
            value = g[val_idx]
            # append point and value to points and values arrays
            points.append(point)
            values.append(value)

    return points, values


# create new points array after interpolation
def points_int(x, y):
    """
    Create new array of points after interpolation. Includes all original points plus intermediate interoplated steps.

    
    Inputs:
    -----
    x: array-like
    Array of interpolated x (wavelength) values

    y: array-like
    Array of interpolated y (size) values


    Outputs:
    -----
    pointsint: array-like
    Array of (x, y) points
    """

    # set blank array to append points to later
    pointsint = []

    # iterate over x values
    for val1 in x:
        # iterate over y values
        for val2 in y:
            # create (x, y) point and append to array
            pointsint.append([val1, val2])

    return pointsint


# split data files based on orientation of grain
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

# run 2d linear interpolation
def lin2dinterp(points, values, pointsint):
    """
    Run linear 2D interpolation for the given points and values

    Inputs:
    -----
    points: array-like
    Array of (wavelength, size) points for Qsc or g as given in the tables

    values: array-like
    Array of corresponding Qsc or g values for points

    pointsint: array-like
    Array of points after interpolation is performed. Output of pointsint function.


    Output:
    -----
    valuesint: array-like
    Array of values after interpolation
    """

    # set up interpolation function given the provided points and corresponding values
    # fill_value is set to 0.00 to handle NaNs
    lininterp = interp.LinearNDInterpolator(points, values, fill_value=0.00)
    # create array of values using interpolation function
    valuesint = [lininterp(point) for point in pointsint]

    return valuesint
