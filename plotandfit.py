import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def saturation(p, I0, P0):
    I = I0 / (1 + P0/ p)
    return I


data_single = np.loadtxt('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/data/leistungskurve_single.dat')
data_background = np.loadtxt('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/data/leistungskurve_background.dat')

power, counts_single, counts_background = data_single[:, 0], data_single[:, 1], data_background[:, 1]

parameters_single, covariance_single = curve_fit(saturation, power, counts_single, p0=[30000, 10])
parameters_background, covariance_background = curve_fit(saturation, power, counts_background, p0=[30000, 10])
fit_I0_single = parameters_single[0]
fit_P0_single = parameters_single[1]
fit_I0_bg = parameters_background[0]
fit_P0_bg = parameters_background[1]

print(f'fit parameters single: I0 = {fit_I0_single}, P0 = {fit_P0_single}.'
      f'fit parameters background: I0 = {fit_I0_bg}, P0 = {fit_P0_bg}.')
fitted_single = saturation(power, fit_I0_single, fit_P0_single)
fitted_background = saturation(power, fit_I0_bg, fit_P0_bg)

fig, ax = plt.subplots()
ax.plot(power, counts_single, 'r+', label='single NV')
ax.plot(power, fitted_single, 'r--', label='fitted single NV')
ax.plot(power, counts_background, 'b+', label='background')
ax.plot(power, fitted_background, 'b--', label='fitted background')

plt.xlabel('power (mW)')
plt.ylabel('c/s')
plt.legend()
plt.savefig('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/data/leistungskurven.png')



