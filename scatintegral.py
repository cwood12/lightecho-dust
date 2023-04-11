"""
Functions needed to calculate the integrand of the scattering function (Eq. 8) from Sugerman (2003); ApJ 126,1939.
"""

# __all__ = [extract_g,
#            extract_Qsc,
#            scatangle,
#            Phi,
#            dS]
__author__ = 'Charlotte M. Wood'
__version__ = '0.1'


import numpy as np
import astropy.units as u

import dustconst as c
import grainsizedist as gsd
import interp_dust as id


# extract g(lambda, a) value from table 
def extract_g(wave, a, pointsg, valuesg):
    """
    Extracts and returns the value of g from interpolated data table from Weingartner & Draine (2001); ApJ 548,296. g describes the degree of forward scattering.

    
    Inputs:
    -----
    wave: float
    A specified value for the wavelength in cm, on which g depends

    a: float
    A specified value for the size of the grain in cm, on which g depends

    pointsg: array-like
    An array of (wavelength, size) points

    valuesg: array-like
    An array of g values corresponding to each (wavelength, size) point

    
    Outputs:
    -----
    valuesg[idx]: float
    The corresponding value of g for the specified (wave, a) point
    """

    # iterate over the array of points to search for matching (wave, a) point
    for idx in range(0,len(pointsg)):
        # if the first value of the current point matches wave and if the second value matches a
        # return the corresponding g value
        if pointsg[idx][0] == wave and pointsg[idx][1] == a:
            return valuesg[idx]
        # if one or both do not match, move to the next iteration
        else:
            continue


# extract Qsc(lambda, a) value from table
def extract_Qsc(wave, a, pointsq, valuesq):
    """
    Extracts and returns the value of Qsc from interpolated data table from Draine et al. (2021); ApJ 917,3. Qsc describes the grain scattering efficiency.

    
    Inputs:
    -----
    wave: float
    A specified value for the wavelength in cm, on which Qsc depends

    a: float
    A specified value for the size of the grain in cm, on which Qsc depends

    pointsq: array-like
    An array of (wavelength, size) points

    valuesq: array-like
    An array of Qsc values corresponding to each (wavelength, size) point

    
    Outputs:
    -----
    valuesq[idx]: float
    The corresponding value of Qsc for the specified (wave, a) point
    """

    # iterate over the array of points to search for matching (wave, a) point
    for idx in range(0,len(pointsq)):
        # if the first value of the current point matches wave and if the second value matches a
        # return the corresponding Qsc value
        if pointsq[idx][0] == wave and pointsq[idx][1] == a:
            return valuesq[idx]
        # if one or both do not match, move to the next iteration
        else:
            continue


# calculate the scattering angle for a given epoch of observation
def scatangle(radcm):
    """
    Calculates the scattering angle for the light echo in a given epoch of observation.

    
    Inputs:
    -----
    radcm: float
    r, value of the strait-line distance between the SN and the scattering dust in units of cm. See Fig. 1 of Sugerman (2003); ApJ 126,1939.

    NOTE: the speed of light in cm/s (c), the time of the observation in days since peak (t), and the time of peak in days since peak (tpeak) are set in separate file, dustconst.py.

    
    Outputs:
    -----
    ang: float
    The value of the scattering angle in radians
    """

    # angle = 1 - [c * (t - tpeak) / radcm]
    ang = 1 - (c.c*(c.t.to(u.s) - c.tpeak.to(u.s))/radcm)
    return ang*u.rad


# calculate the phase function Phi(mu, lambda, a)
def Phi(mu, g):
    """
    Calculates the scattering phase function, Phi(mu, lambda, a). See Eq. 3 of Sugerman (2003); ApJ 126,1939.

    
    Inputs:
    -----
    mu: float
    Value of cos(scattering angle) for a given epoch of observation

    g: float
    Value of the degree of forward scattering for a given size and wavelength (from extract_g)


    Outputs:
    -----
    phi: float
    Value of the scattering phase function for the given scattering angle, wavelength, and size
    """

    # Phi = (1 - g^2) / [(1 + g^2 - 2 g mu)^(3/2)]
    phi = (1 - g**2) / ((1 + g**2 - 2*g*mu)**(3/2))
    return phi


# calculate the integrand of the integrated scattering function
def dS(a, Qsc, phi, f):
    """
    Calculates the integrand of the integrated scattering function for a given wavelength, size, and scattering angle. Integrate this over all grain sizes to get the value of the integrated scattering function for a given wavelength. See Eq. (8) of Sugerman (2003); ApJ 126,1939.


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    Qsc: float
    Value of the grain scattering efficiency for given wavelength and grain size (from extract_Qsc)

    phi: float
    Value of the scattering phase function for the given scattering angle, wavelength, and grain size (from Phi)

    f: float
    Value of the grain size distribution function (see grainsizedist.py) for the specified a


    Outputs:
    -----
    dS: float
    Value of the integrand of the integrated scattering function for the specified size in cm^2
    """

    # S(lambda, mu) = integral[Qsc(lambda, a) * pi * a^2 * Phi(mu, lambda, a) * f(a) da]
    # dS = Qsc(lambda, a) * pi * a^2 * Phi(mu, lambda, a) * f(a)
    dS = Qsc * np.pi*a**2 * phi * f
    return dS
