import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def saturation(p, I0, P0):
    I = I0 / (1 + P0/ p)
    return I


data = np.loadtxt('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/Au microantenna 20um/20221019/20221019_leistungskurven_tief.txt')
#  data_background = np.loadtxt('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/data/20221110_bg.dat')

power, counts_single1, counts_single2, counts_single3, counts_bg = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
single1_corrected= counts_single1 - counts_bg
single2_corrected= counts_single2 - counts_bg
single3_corrected= counts_single3 - counts_bg

parameters_single1, covariance_single1 = curve_fit(saturation, power, single1_corrected, p0=[30000, 10])
parameters_single2, covariance_single2 = curve_fit(saturation, power, single2_corrected, p0=[30000, 10])
parameters_single3, covariance_single3 = curve_fit(saturation, power, single3_corrected, p0=[30000, 10])
fit_I0_single1 = parameters_single1[0]
fit_P0_single1 = parameters_single1[1]
fit_I0_single2 = parameters_single2[0]
fit_P0_single2 = parameters_single2[1]
fit_I0_single3 = parameters_single3[0]
fit_P0_single3 = parameters_single3[1]

print(f'fit parameters single 1: I0 = {fit_I0_single1}, P0 = {fit_P0_single1}.')
print(f'fit parameters single 2: I0 = {fit_I0_single2}, P0 = {fit_P0_single2}.')
print(f'fit parameters single 3: I0 = {fit_I0_single3}, P0 = {fit_P0_single3}.')

fitted_single1 = saturation(power, fit_I0_single1, fit_P0_single1)
fitted_single2 = saturation(power, fit_I0_single2, fit_P0_single2)
fitted_single3 = saturation(power, fit_I0_single3, fit_P0_single3)


fig, ax = plt.subplots()
ax.plot(power, single1_corrected, 'r+', label='single NV 1')
ax.plot(power, single2_corrected, 'g+', label='single NV 2')
ax.plot(power, single3_corrected, 'b+', label='single NV 3')
ax.plot(power, fitted_single1, 'r--', label='fitted single NV 1')
ax.plot(power, fitted_single2, 'g--', label='fitted single NV 2')
ax.plot(power, fitted_single3, 'b--', label='fitted single NV 3')
ax.plot(power, counts_bg, 'yo', label='background')

plt.xlabel('power (mW)')
plt.ylabel('c/s')
plt.legend()
plt.savefig('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/Au microantenna 20um/20221019/20221019_Leistungskurven_tief_fit.png')
plt.show()



