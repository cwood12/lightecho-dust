"""
Functions needed to recreate grain size distribution functions from Weingartner & Draine (2001)[ApJ 548,296], Draine et al. (2021)[ApJ 917,3] and Hensley & Draine (2023)[ApJ 948,55]
"""

# __all__ = []
__author__ = 'Charlotte M. Wood'
__version__ = '0.1'

import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('pg.mplstyle')
from astropy.table import Table
from numpy.polynomial import Polynomial as poly
import scipy.integrate as integrate
from scipy.special import erf
import dustconst as c
import astropy.units as u

# NOTE: all variables in functions not part of the call are constants defined in "dustconst.py"

# curvature function
def F(a, beta, at):
    """
    Calculates the curvature term (Eq. 6) from Weingartner & Draine (2001)


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    beta: float
    Constant, taken from Table 1 of Weingartner & Draine (2001)

    at: float
    Value of the upper limit for size, in cm


    Outputs:
    -----
    1 + (beta * a / at): float
    Value of F if beta is greater than or equal to zero

    (1 - (beta * a / at))^-1: float
    Value of F if beta is less than 0
    """
    
    # determine if value of beta is > or = 0
    if beta >= 0:
        # return value of first form
        return 1 + (beta * a / at)
    # if beta < 0
    else:
        # return value of second form
        return (1 - (beta * a / at))**(-1)


# including very small grains for carbonaceous dust
# Bi for B1 and B2 to go into D(a)
def Bi_carb(a0i,bci):
    """
    Calculates the value of Bi (Eq. 3) from Weingartner & Draine (2001)


    Inputs:
    -----
    a0i: float
    Value of the minimum grain size in cm, a01 or a02

    bci: float
    Value of the total C abundance per H nucleus, bc1 or bc2

    NOTE: a normalization factor (sig), the density of graphite (rho), and the mass of a carbon atom (mc) are set in a separate file, dustconst.py.


    Outputs:
    -----
    Bi1*Bi2: float
    Product of the two terms for Bi, unitless
    """

    # calculate the first term of Bi
    Bi1 = ((3 / (2*np.pi)**(3/2)) * np.exp(-4.5*c.sig**2) / (c.rho*a0i**3*c.sig))
    # calculate the second term of Bi
    Bi2 = bci*c.mc / (1 + erf((3*c.sig/np.sqrt(2)) + (np.log(a0i/c.a01)/(c.sig*np.sqrt(2)))))
    # multiply the two terms and return
    return Bi1*Bi2

# D(a), size distribution for the smallest grains
def Da_carb(a, B1, B2):
    """
    Calculates the value of D(a) (Eq. 2) from Weingartner & Draine (2001)


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    B1: float
    Result from Bi_carb(a01, bc1)

    B2: float
    Result from Bi_carb(a02, bc2)


    Outputs:
    -----
    da1+da2: float
    Addition of the two terms from the summation
    """

    # expression for i=1
    da1 = (B1/a)*np.exp(-0.5*(np.log(a/c.a01)/c.sig)**2)
    # expression for i=2
    da2 = (B2/a)*np.exp(-0.5*(np.log(a/c.a02)/c.sig)**2)
    # sum da1+da2 and return
    return da1+da2

# distribution for carbonaceous dust given grain size
def Dist_carb(a, B1, B2):
    """
    Calculates the grain size distribution for carbonaceous "graphite" grains (Eq. 4) from Weingartner & Draine (2001)


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    B1: float
    Result from Bi_carb(a01, bc1)

    B2: float
    Result from Bi_carb(a02, bc2)

    NOTE: constants Cg, alphag, betag, cutoff grain size (atg), and control size (acg) are set in separate file dustconst.py.


    Outputs:
    -----
    Number of grains for a particular size (units 1/cm)
    """

    # (4) = D(a) + dist1 * [1 or dist2]
    # calculate the first term, dist1
    dist1 = ((c.Cg/a)*(a/c.atg)**c.alphag * F(a, c.betag, c.atg))

    # determine which additional term to use based on grain size
    # if a < 3.5e-8, the function is undefined so return 0
    if a.value < 3.5e-8:
        return 0
    # if 3.5e-8 <= a < atg, return the result of Da_carb(a, B1, B2) + dist1 * 1
    elif a.value >= 3.5e-8 and a < c.atg:
        return Da_carb(a, B1, B2) + dist1
    # if a > atg, return the result of Da_carb(a, B1, B2) + (dist1*dist2)
    else:
        dist2 = np.exp(-((a - c.atg)/c.acg)**3)
        return Da_carb(a, B1, B2) + (dist1 * dist2)



# distribution for silicate dust given grain size
def Dist_sil(a):
    """
    Calculates the grain size distribution for silicate grains (Eq. 5) from Weingartner & Draine (2001)


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    NOTE: constants Cs, alphas, betas, cutoff grain size (ats), and control size (acs) are set in separate file dustconst.py.


    Outputs:
    -----
    Number of grains for a particular size (units 1/cm)
    """

    # (5) = dist1 * [1 or dist2]
    # calculate the first term, dist1
    dist1 = (c.Cs/a)*(a/c.ats)**c.alphas * F(a, c.betas, c.ats)

    # determine which additional term to use based on grain size
    # if a < 3.5e-8, the function is undefined so return 0
    if a.value < 3.5e-8:
        return 0
    # if 3.5e-8 <= a < atg, return dist1 * 1
    elif a.value >= 3.5e-8 and a < c.ats:
        return dist1
    # if a > atg, return dist1*dist2
    else:
        dist2 = np.exp(-((a - c.ats)/c.acs)**3)
        return dist1*dist2


# astrodust size distribution
def Dist_astro(a):
    """
    Calculates the grain size distribution for "astrodust" (Eq. 14) from Draine et al. (2021)


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    NOTE: constant p, the astrodust volume per H (Vad) in cm^3, the minimum grain size for astrodust (aminAd) in cm, and the maximum grain size for astrodust (amaxAd) in cm are set in a separate file, dustconst.py.


    Outputs:
    dist: float
    Value of the number of grains for the given grain size in 1/cm
    """

    # (14) = [(4+p)*(3Vad/4pi)*a^p]/[amaxAd^(4+p) - aminAd^(4+p)]
    dist = (4 + c.p) * (3 / (4*np.pi))*c.Vad * a**c.p / (c.amaxAd**(4+c.p) - c.aminAd**(4+c.p))
    return dist


# PAH distribution
def Dist_pah(a):
    """
    Calculates the grain size distribution for polycyclic aromatic hydrocarbon (PAH) grains (Eq. 15) from Draine et al. (2021)


    Inputs:
    -----
    a: float
    Value of the grain size in cm

    NOTE: constants B1pah, B2pah, a01pah, a02pah, and sig are set in a separate file, dustconst.py.


    Outputs:
    -----
    da1+da2: float
    Addition of the two summation terms, value of the number of grains for the given size (1/cm)
    """

    # calculate the first summation term with Bj=B1pah and a0j=a01pah
    da1 = (c.B1pah/a)*np.exp(-0.5*(np.log(a/c.a01pah)/c.sig)**2)
    # calculate the second summation term with Bj=B2pah and a0j=a02pah
    da2 = (c.B2pah/a)*np.exp(-0.5*(np.log(a/c.a02pah)/c.sig)**2)
    # add terms and return
    return da1+da2


# PAH distribution from Hensley & Draine (2023)
def PAH(a):
    """
    Calculate the PAH distribution from Hensley & Draine (2023), Eq. 17.

    
    Inputs:
    -----
    a: float
    Value of the grain size, in cm. 
    
    NOTE: Constants aminPAH, B1, B2, a01pah, a02pah, and sig are set in separate file, dustconst.py


    Outputs:
    -----
    dist1+dist2: float
    Value of the summation from the size distribution equation, value of the number of grains for the given size (1/cm).
    """

    # equation is only valid for sizes > aminPAH
    if a < c.aminPAH:
        return 0/u.cm
    # for valid sizes, calculate the value of Eq. 17 for j=1 and j=2
    else:
        dist1 = (c.B1/a)*np.exp(-0.5*(np.log(a/c.a01pah)/c.sig)**2)
        dist2 = (c.B2/a)*np.exp(-0.5*(np.log(a/c.a02pah)/c.sig)**2)
        
        # combine for value of summation and return
        return dist1+dist2


# ionization fraction of PAHs from Hensley & Draine (2023)
def fion(a):
    """
    Calculate the ionized fraction of PAHs (Eq. 19, Hensley & Draine 2023).

    
    Inputs:
    -----
    a: float
    Value of the grain size, in cm.


    Outputs:
    -----
    f: float
    Fraction of ionized PAHs for the given grain size.
    """
    
    # Eq. 19
    f = 1 - (1 / (1 + (a/(10e-8*u.cm))))
    return f


# astrodust distribution from Hensley & Draine (2023)
def astrodust(a):
    """
    Calculate the astrodust size distribution from Hensley & Draine (2023), Eq. 24.

    
    Inputs:
    -----
    a: float
    Value of the grain size, in cm.

    NOTE: Constants A0-A5, BAd, a0Ad, and sigAd are set in separate file, dustconst.py.
    Ai = [A1, A2, A3, A4, A5]


    Outputs:
    -----
    dist1+dist2: float
    Addition of the two terms of Eq. 24, value of the number of grains for the given size.
    """

    if a < c.aminAd23 or a > c.amaxAd23:
        return 0/u.cm
    
    else:
        # initialize for summation in exp
        expsum = 0
        # sum over i = 1 to 5
        for idx in range(0,5):
            x = c.Ai[idx]*(np.log(a/(1e-8*u.cm))**(idx+1))
            expsum += x

        # calculate each term
        dist1 = (c.A0/a) * np.exp(expsum)
        dist2 = (c.BAd/a) * np.exp(-0.5*(np.log(a/c.a0Ad)/c.sigAd)**2)
    
        # add the terms and return
        return dist1+dist2
