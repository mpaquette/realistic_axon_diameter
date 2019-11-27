import numpy as np
import vangelderen as vg
import pylab as pl
from scipy.stats import norm

def nilsson_diameter(sigma_bar, D_0, delta, G):
	# d_min =  ((768 sigma_bar D_0) / (7 gamma^2 delta G^2))^(1/4)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return ((768 * sigma_bar * D_0) / (7. * gamma**2 * delta * G**2))**(1/4.)

def nilsson_dS(d, D_0, delta, G):
	# dS/S_0 =  (7 gamma^2 delta G^2 d^4) / (768 D_0)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return ((7. * gamma**2 * delta * G**2) / (768 * D_0)) * d**4


# invivo diffusivity
# D0 = 2e-9
D0 = 0.66e-9
# most sensitive Connectom-like scheme
# scheme_connectom = np.array([[0.3, 40e-3, 40e-3]])
scheme_connectom = np.array([[1.5, 40e-3, 40e-3]])
# alpha significance level
alpha = 0.05
# data SNR at B0
# SNR = 30.0
SNR = 300.0

#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi


# compute sigma_bar for the diameter limit formula
sigmab = norm().ppf(1-alpha) / SNR


# diameters in m
diams = np.arange(0.1, 5.1, 0.1)*1e-6
# number of noise trial
Ntrial = 100


fit_data = np.zeros((len(diams), Ntrial))

for idiam, diam in enumerate(diams):
	# noiseless_signal = vg.vangelderen_cylinder_perp(D0, 0.5*diam, scheme_connectom, m_max=10)
	noiseless_signal = nilsson_dS(diam, D0, scheme_connectom[0,2], scheme_connectom[0,0])
	noiseless_fit = nilsson_diameter(noiseless_signal, D0, scheme_connectom[0,2], scheme_connectom[0,0])
	print('D = {:.2f}   Fit = {:.2f}'.format(1e6*diam, 1e6*noiseless_fit))
	for itrial in range(Ntrial):
		noise = (1/float(SNR))*np.random.randn()
		noisy_fit = nilsson_diameter(noiseless_signal+noise, D0, scheme_connectom[0,2], scheme_connectom[0,0])
		fit_data[idiam, itrial] = noisy_fit

fit_data[np.isnan(fit_data)] = 0



d_min = nilsson_diameter(sigmab, D0, scheme_connectom[0,2], scheme_connectom[0,0])



pl.figure()
pl.scatter(np.repeat(diams, Ntrial)*1e6, fit_data.ravel()*1e6, color='blue')
pl.plot(diams*1e6, diams*1e6, color='green')
pl.axvline(d_min*1e6, color='r')

pl.xlim([0, 1.05*np.max(diams)*1e6])
pl.ylim([0, 1.05*np.max(diams)*1e6])

pl.xlabel('True diameters (um)')
pl.ylabel('Fitted diameters (um)')

pl.gca().set_aspect('equal')

pl.show()








