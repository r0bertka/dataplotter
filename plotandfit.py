import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def saturation(p, I0, P0):
    return I0 / (1 + P0/ p)


data_double = np.loadtxt('C:/Users/rober/Documents/Aufbau_QC/double_NV_power_curve.txt')
data_single = np.loadtxt('C:/Users/rober/Documents/Aufbau_QC/Tests_Scanning/Au_microantenna_20um/20221017/20221017_Saettigung_Single_NV.txt')

# frequency_0G, intensity_0G = data_0G[:, 0], data_0G[:, 1]
# relative_frequency_0G = (frequency_0G[:] - 2.87e9) / 1e6
# normalized_intensity_0G = intensity_0G[:] / np.max(intensity_0G)
# frequency_500G, intensity_500G = data_500G[:, 0], data_500G[:, 1]
# relative_frequency_500G = (frequency_500G[:] - 1.42e9) / 1e6
# normalized_intensity_500G = intensity_500G[:] / np.max(intensity_500G)

power, counts_single1, counts_single2, counts_single3, counts_bgsingle = data_single[:, 0], data_single[:, 1], data_single[:, 2], data_single[:, 3], data_single[:, 4]
single1_corrected= counts_single1 - counts_bgsingle
single2_corrected= counts_single2 - counts_bgsingle
single3_corrected= counts_single3 - counts_bgsingle

counts_double, counts_bgdouble = data_double[:, 1], data_double[:, 2]
double_corrected = counts_double - counts_bgdouble

parameters_single1, covariance_single1 = curve_fit(saturation, power, single1_corrected, p0=[30000, 10])
parameters_single2, covariance_single2 = curve_fit(saturation, power, single2_corrected, p0=[30000, 10])
parameters_single3, covariance_single3 = curve_fit(saturation, power, single3_corrected, p0=[30000, 10])

parameters_double, covariance_double = curve_fit(saturation, power, double_corrected, p0=[40000, 10])

fit_I0_single1 = parameters_single1[0]
fit_P0_single1 = parameters_single1[1]
fit_I0_single2 = parameters_single2[0]
fit_P0_single2 = parameters_single2[1]
fit_I0_single3 = parameters_single3[0]
fit_P0_single3 = parameters_single3[1]
fit_I0_double = parameters_double[0]
fit_P0_double = parameters_double[1]
#
print(f'fit parameters single 1: I0 = {fit_I0_single1}, P0 = {fit_P0_single1}.')
print(f'fit parameters single 2: I0 = {fit_I0_single2}, P0 = {fit_P0_single2}.')
print(f'fit parameters single 3: I0 = {fit_I0_single3}, P0 = {fit_P0_single3}.')
print(f'fit parameters double: I0 = {fit_I0_double}, P0 = {fit_P0_double}.')
#
fitted_single1 = saturation(power, fit_I0_single1, fit_P0_single1)
fitted_single2 = saturation(power, fit_I0_single2, fit_P0_single2)
fitted_single3 = saturation(power, fit_I0_single3, fit_P0_single3)
fitted_double = saturation(power, fit_I0_double, fit_P0_double)


fig, ax = plt.subplots()
# ax.plot(relative_frequency_0G, normalized_intensity_0G, 'r--', label='B=0mT')
# ax.plot(relative_frequency_500G, normalized_intensity_500G, 'b--', label='B=50mT')
ax.plot(power, single1_corrected, 'r+', label='single NV 1')
ax.plot(power, single2_corrected, 'g+', label='single NV 2')
ax.plot(power, single3_corrected, 'b+', label='single NV 3')
ax.plot(power, double_corrected, 'm+', label = 'double NV')
ax.plot(power, fitted_single1, 'r--', label='fitted single NV 1')
ax.plot(power, fitted_single2, 'g--', label='fitted single NV 2')
ax.plot(power, fitted_single3, 'b--', label='fitted single NV 3')
ax.plot(power, fitted_double, 'm--', label='fitted double NV')
#  ax.plot(power, counts_bgsingle, 'yo', label='background1')
ax.plot(power, counts_bgdouble, 'ko', label='background')

plt.xlabel('relative frequency (MHz)')
plt.ylabel('normalized signal')
plt.legend()
#  plt.savefig('C:/Users/rober/Documents/Aufbau_QC/Pulsed tests/Vergleich_Leistungskurven_single_und_double.png')
plt.show()



