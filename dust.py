"""
Implement routine from Sugerman (2003); ApJ 126,1939
"Observability of Scattered Light Echoes Around Variable Stars and Cataclysmic Events"

Includes dust distribution routines from Weingartner & Draine (2001); ApJ 548,296 and Draine et al. (2021); ApJ 917,3. 

Optical properties of dust (Qsc, g) come from Draine & Hensley (2021); ApJ 909,94 (http://arks.princeton.edu/ark:/88435/dsp01qb98mj541) and Weingartner & Draine (2001); ApJ 548,296 (https://www.astro.princeton.edu/~draine/dust/dust.diel.html). 
"""

# __all__ = []
__author__ = 'Charlotte M. Wood'
__version__ = '0.1'

import numpy as np
np.set_printoptions(precision=4, floatmode='fixed')
import matplotlib.pyplot as plt
plt.style.use('pg.mplstyle')
import scipy.integrate as integrate
from scipy.special import erf
import astropy.units as u

import grainsizedist as gsd
import dustconst as c
import interp_dust as id
import scatintegral as scaint

#----------

# All files downloaded from the above websites have been placed into a folder named "dustmodels" and file names are kept as-is
# Constants set in dustconst.py assume preferred Milky Way dust model (Case A, Rv = 3.1, bc = 6e-5) from WD01

# pull out available wavelengths for Qsc values, convert to cm from um, and take the log
logwaveq = np.log10(1e-4*np.loadtxt('dustmodels/DH21_wave.dat')) #log(cm)
# pull out available sizes for Qsc values, convert to cm from um, and take the log
logsizeq = np.log10(1e-4*np.loadtxt('dustmodels/DH21_aeff.dat')) #log(cm)

# set min, max wavelength and size for interpolation grid (in log space)
minwave, maxwave = min(logwaveq), max(logwaveq)
minsize, maxsize = min(logsizeq), max(logsizeq)

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

# rearrange data into grids for interpolation
# (wavelength, size) points and Qsc values for Model 1 and each orientation
pointsq11, valuesq11 = id.point_val(model1_jori1, logwaveq, logsizeq)
pointsq12, valuesq12 = id.point_val(model1_jori2, logwaveq, logsizeq)
pointsq13, valuesq13 = id.point_val(model1_jori3, logwaveq, logsizeq)

# (wavelength, size) points and g values for carbonaceous/PAH dust
pointsgc, valuesgc = id.point_val_g(logwaveg, logsizeg, gcarb)
# (wavelength, size) points and g values silicate/astro dust
pointsgs, valuesgs = id.point_val_g(logwaveg, logsizeg, gsil)

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
model11_interp = id.lin2dinterp(pointsq11, valuesq11, pointsinterp)
model12_interp = id.lin2dinterp(pointsq12, valuesq12, pointsinterp)
model13_interp = id.lin2dinterp(pointsq13, valuesq13, pointsinterp)
print('Interpolating g...')
gcarb_interp = id.lin2dinterp(pointsgc, valuesgc, pointsinterp)
gsil_interp = id.lin2dinterp(pointsgs, valuesgs, pointsinterp)

#----------
# calculate the grain size distribution for each dust type
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

# calculate integrated scattering function across all sizes for given wavelength and scattering angle (Eq. 8 from Sugerman 2003)

# calculate rad (r from Fig. 1, Sugerman 2003) for echo in given observation and convert to cm
rad = np.sqrt(c.pecho**2 + c.z**2) #ly
radcm = rad.to(u.cm)

# set wavelength (remove when integrating over all wavelengths)
wave = 10**x[300]*u.cm #cm
print('Wavelength = ' + str(wave))

# determine scattering angle and calculate mu for given observation
scatangle = scaint.scatangle(radcm)
mu = np.cos(scatangle)
print('Mu = ' + str(mu))

# remove log from x (wave) & y (size) arrays, add units
x_nolog = 10**x * u.cm
y_nolog = 10**y * u.cm

print('Performing scattering integration...')
# Model 1, orientation 1
ds_array = []
int_sizes = []
for size in y_nolog:
    if size.value >= 3.5e-8 and size.value <= 1.0e-4:
        g = scaint.extract_g(wave.value, size.value, pointsinterp_nolog, gcarb_interp)[0]
        Qscat = scaint.extract_Qsc(wave.value, size.value, pointsinterp_nolog, model11_interp)[0]

        phi = scaint.Phi(mu, g)

        f = gsd.Dist_carb(size, B1, B2)
        dS = scaint.dS(size, Qscat, phi, f)
        int_sizes.append(size.value)
        ds_array.append(dS.value)

int_result = integrate.trapezoid(ds_array, int_sizes)*u.cm**2
print(int_result)

#----------






# # Model 1, orientation 2
# ds_array = []
# int_sizes = []
# for size in y_nolog:
#     if size.value >= 3.5e-8 and size.value <= 1.0e-4:
#         g = scaint.extract_g(wave.value, size.value, pointsinterp_nolog, gcarb_interp)[0]
#         Qscat = scaint.extract_Qsc(wave.value, size.value, pointsinterp_nolog, model12_interp)[0]

#         phi = scaint.Phi(mu, g)

#         f = gsd.Dist_carb(size, B1, B2)
#         dS = scaint.dS(size, Qscat, phi, f)
#         int_sizes.append(size.value)
#         ds_array.append(dS.value)

# int_result = integrate.trapezoid(ds_array, int_sizes)*u.cm**2
# print(int_result)

# # Model 1, orientation 3
# ds_array = []
# int_sizes = []
# for size in y_nolog:
#     if size.value >= 3.5e-8 and size.value <= 1.0e-4:
#         g = scaint.extract_g(wave.value, size.value, pointsinterp_nolog, gcarb_interp)[0]
#         Qscat = scaint.extract_Qsc(wave.value, size.value, pointsinterp_nolog, model13_interp)[0]

#         phi = scaint.Phi(mu, g)

#         f = gsd.Dist_carb(size, B1, B2)
#         dS = scaint.dS(size, Qscat, phi, f)
#         int_sizes.append(size.value)
#         ds_array.append(dS.value)

# int_result = integrate.trapezoid(ds_array, int_sizes)*u.cm**2
# print(int_result)
