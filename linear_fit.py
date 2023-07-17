import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

def linear_fit(x, prefactor):  #, offset):
    return prefactor * x  # + offset

data = np.loadtxt('C:\\Users\\rober\\Documents\\t_rabi_vs_mw_amplitude.dat')
amplitude = data[5:, 0] / 1000  # in V
t_rabi = data[5:, 1] * 1e-9  # in s

f_rabi = 1 / t_rabi * 1e-3  # in kHz

params, covar = curve_fit(linear_fit, amplitude, f_rabi, p0=[1])

A = params[0]
#  B = params[1]

f_rabi_fitted = linear_fit(amplitude, A)  # , B)

fig, ax = plt.subplots()
ax.plot(amplitude, f_rabi, 'r*', label='measured')
ax.plot(amplitude, f_rabi_fitted, 'r-', label=f'fitted: f(x) = {round(params[0], 2)}kHz/V* mw_ampl')  #  + {round(params[1], 2)}kHz')

plt.xlabel('mw amplitude (V)')
plt.ylabel('Rabi frequency (kHz)')
plt.legend()

plt.savefig('C:\\Users\\rober\\Documents\\t_rabi_vs_mw_amplitude_trunc.png')

plt.show()
