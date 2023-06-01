"""
Implement routine from Sugerman (2003); ApJ 126,1939
"Observability of Scattered Light Echoes Around Variable Stars and Cataclysmic Events"

Includes dust distribution routines from Weingartner & Draine (2001)[ApJ 548,296], Draine et al. (2021)[ApJ 917,3], and Hensley & Draine (2023)[ApJ 948,55]. 

Optical properties of dust (Qsc, g, Csc) come from Draine & Hensley (2021)[ApJ 909,94] (http://arks.princeton.edu/ark:/88435/dsp01qb98mj541), Weingartner & Draine (2001)[ApJ 548,296] (https://www.astro.princeton.edu/~draine/dust/dust.diel.html), and Draine & Hensley (2023)[ApJ 948,55] (https://dataverse.harvard.edu/dataverse/astrodust). 
"""

# __all__ = []
__author__ = 'Charlotte M. Wood'
__version__ = '0.1'

import numpy as np
np.set_printoptions(precision=4, floatmode='fixed')
import matplotlib.pyplot as plt
plt.style.use('pg.mplstyle')
import scipy.interpolate as interp
import scipy.integrate as integrate
from scipy.special import erf
import astropy.units as u

import grainsizedist as gsd
import dustconst as c
import interpdust as id
import scatintegral as scaint
import hstfilters as hst


#----------
# Open data files
#----------

# All files downloaded from the above websites have been placed into folders named 'dustmodels_WD01', 'dustmodels_DH21', and 'dustmodels_HD23'.
# Constants set in dustconst.py assume preferred Milky Way dust model (Case A, Rv = 3.1, bc = 6e-5) from WD01


# pull out available wavelengths for Qsc values, convert to cm from um, and take the log
logwaveq = np.log10(1e-4*np.loadtxt('dustmodels/DH21_wave.dat')) #log(cm)
# pull out available sizes for Qsc values, convert to cm from um, and take the log
logsizeq = np.log10(1e-4*np.loadtxt('dustmodels/DH21_aeff.dat')) #log(cm)


# "For astrodust we assume a porosity P = 0.20" - Draine et al. (2021)
# Astrodust with Fe = 0.10 only provided for b/a = 0.5 and 2.0
# Model 1: P0.20_Fe0.00_0.500
# Model 2: P0.20_Fe0.00_2.000
# Model 3: P0.20_Fe0.10_0.500
# Model 4: P0.20_Fe0.10_2.000

# model #1 - poro = 0.2, Fe = 0.0, b/a = 0.5
# Qsc values split by orientation
file1 = 'dustmodels/q_DH21Ad_P0.20_Fe0.00_0.500.dat'
model1 = np.loadtxt(file1)
model1_jori1, model1_jori2, model1_jori3 = id.jori(model1)


# older models used for g (degree of forward scattering) values
# carbonaceous dust
carbong = 'dustmodels/Gra_81.dat'
gcarb = np.loadtxt(carbong, usecols=(3), unpack=True)
# silicate dust
silicong = 'dustmodels/suvSil_81.dat'
gsil = np.loadtxt(silicong, usecols=(3), unpack=True)

# pull out available wavelengths for g values, convert to cm from um, and take the log
logwaveg = np.log10(1e-4*np.loadtxt('dustmodels/LD93_wave.dat', unpack=True)) #log(cm)
# pull out available sizes for the g values, convert to cm from um, and take the log
logsizeg = np.log10(1e-4*np.loadtxt('dustmodels/LD93_aeff.dat', unpack=True)) #log(cm)


#----------
# Perform interpolation
#----------

# rearrange data into grids for interpolation
# (wavelength, size) points and Qsc values for Model 1 and Orientation 1
pointsq11, valuesq11 = id.point_val_Qsc(logwaveq, logsizeq, model1_jori1)


# (wavelength, size) points and g values for carbonaceous/PAH dust
pointsgc, valuesgc = id.point_val_g(logwaveg, logsizeg, gcarb)
# (wavelength, size) points and g values silicate/astro dust
pointsgs, valuesgs = id.point_val_g(logwaveg, logsizeg, gsil)


# set min, max wavelength and size for interpolation grid (in log space)
minwave, maxwave = min(logwaveq), max(logwaveq)
minsize, maxsize = min(logsizeq), max(logsizeq)
print(10**minsize, 10**maxsize)

# define size of grid in log space
# step for wave already 0.005 in log space
# step for size already 0.025 in log space
# start with step sizes = 0.5 * current step
stepx = 0.0025
stepy = 0.0125
# make x and y grid
x = np.arange(minwave, maxwave+stepx, stepx)
y = np.arange(minsize, maxsize+stepy, stepy)
# turn into points and remove log
pointsinterp = id.points_int(x, y)
pointsinterp_nolog = id.points_int(10**x, 10**y)


# use N-dimensional linear interpolator with log grid
print('Interpolating Qscat...')
lininterp_Qsc = interp.LinearNDInterpolator(pointsq11, valuesq11, fill_value=0.00)
# valuesint_Qsc = [lininterp_Qsc(point) for point in pointsinterp]
valuesint_Qsc = lininterp_Qsc(pointsinterp)

print('Interpolating g...')
# gcarb_interp = id.lin2dinterp(pointsgc, valuesgc, pointsinterp)
# gsil_interp = id.lin2dinterp(pointsgs, valuesgs, pointsinterp)
lininterp_gcarb = interp.LinearNDInterpolator(pointsgc, valuesgc, fill_value=0.00)
# valuesint_gcarb = [lininterp_gcarb(point) for point in pointsinterp]
valuesint_gcarb = lininterp_gcarb(pointsinterp)

lininterp_gsil = interp.LinearNDInterpolator(pointsgs, valuesgs, fill_value=0.00)
# valuesint_gsil = [lininterp_gsil(point) for point in pointsinterp]
valuesint_gsil = lininterp_gsil(pointsinterp)


#----------
# Determine grain size distributions for each dust type
#----------

# carbonaceous/graphite and silicate from WD01
# astrodust and PAHs from Draine et al. (2021)

# create array of sizes used only for plotting distributions
a = np.linspace(3.5e-8, 1.0e-4, 10000)*u.cm

# grain size distribution for carbonaceous dust
# calculate B1
B1 = gsd.Bi_carb(c.a01, c.bc1)
# calculate B2
B2 = gsd.Bi_carb(c.a02, c.bc2)
carbon_distribution = [gsd.Dist_carb(idx, B1, B2).value for idx in a]/u.cm

# grain size distribution for silicate dust
silicate_distribution = [gsd.Dist_sil(idx).value for idx in a]/u.cm

# grain size distribution for "astrodust"
astrodust_distribution = [gsd.Dist_astro(idx).value for idx in a]/u.cm

# grain size distribution for PAHs
pah_distribution = [gsd.Dist_pah(idx).value for idx in a]/u.cm

# plot all the distributions
# multiplied by 1e29 a^4 to replicate Fig. 2 (WD01) and Fig. 9a (Drain et al. 2021)
plt.figure()

plt.loglog(a, 1e29*a**4*carbon_distribution, label='Carbonaceous Dust')
plt.loglog(a, 1e29*a**4*silicate_distribution, label='Silicate Dust')
plt.loglog(a, 1e29*a**4*astrodust_distribution, label='"Astrodust"')
plt.loglog(a, 1e29*a**4*pah_distribution, label='PAHs')

plt.ylim(0.1, 100)
plt.legend(ncol=1, loc='best')
plt.show()


#----------
# Calculate integrated scattering function
#----------

# Eq. (8) from Sugerman (2003)

# calculate radius (r from Fig. 1, Sugerman 2003) for echo in given observation
rad = np.sqrt(c.pecho**2 + c.z**2) #ly
radcm = rad.to(u.cm)

# set wavelength (if only evaluating at one wavelength)
wave = 10**x[300]*u.cm #cm
print('Wavelength = ' + str(wave))

# determine scattering angle and calculate mu for given observation
scatangle = scaint.scatangle(radcm)
mu = np.cos(scatangle)
print('Mu = ' + str(mu))

# remove log from x/wavelength and y/size arrays and add units
x_nolog = 10**x * u.cm
y_nolog = 10**y * u.cm

# integrate across
print('Performing scattering integration...')

# note with new data from Hensley & Draine (2023), values given as Csca, which is Qsca * pi * a^2. In integrated scattering function, replace Qsc*sigma with Csca.