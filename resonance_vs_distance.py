import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

def frac_polynomial(z, a, b, c):
    return a / (b * (z + c) ** 3) + 2.87

distance_array = np.linspace(0, 8, 101)

data = np.loadtxt('C:\\Users\\rober\\Documents\\resonance_vs_distance_QC1.txt')
distance = data[:, 0]  # in mm
f_res = data[:, 1] #  in GHz

params, covar = curve_fit(frac_polynomial, distance, f_res, p0=[1, 0.1, -9])

a_fitted = params[0]
b_fitted = params[1]
c_fitted = params[2]
# f0_fitted = params[3]

f_res_fitted = frac_polynomial(distance_array, a_fitted, b_fitted, c_fitted)

f_res_exp = frac_polynomial(7.62, a_fitted, b_fitted, c_fitted)
print(f_res_exp)

fig, ax = plt.subplots()
ax.plot(distance, f_res, 'r*', label='measured')
ax.plot(distance_array, f_res_fitted, 'r-', label=f'fitted: f(z) = {round(a_fitted, 2)}/({round(b_fitted, 2)}*({round(c_fitted, 2)}+z)) + 2.87')  #
ax.plot(7.62, f_res_exp, 'bx', label='expected')

plt.xlabel('magnet z position (mm)')
plt.ylabel('resonance frequency (GHz)')
plt.legend()

#  plt.savefig('C:\\Users\\rober\\Documents\\t_rabi_vs_mw_amplitude_trunc.png')

plt.show()