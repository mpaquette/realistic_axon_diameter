import numpy as np
import vangelderen as vg
import pylab as pl
from scipy.stats import norm


def nilsson_diameter(sigma_bar, D_0, delta, G):
	# d_min =  ((768 sigma_bar D_0) / (7 gamma^2 delta G^2))^(1/4)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return ((768 * sigma_bar * D_0) / (7. * gamma**2 * delta * G**2))**(1/4.)

def nilsson_diameter_WRONG(sigma_bar, D_0, delta, G):
	# d_min =  ((768 sigma_bar D_0) / (7 gamma^2 delta G^2))^(1/4)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return (768 * sigma_bar * D_0/7.) / ((gamma**2 * delta * G**2))**(1/4.)

def nilsson_dS(d, D_0, delta, G):
	# dS/S_0 =  (7 gamma^2 delta G^2 d^4) / (768 D_0)
	# assume delta = DELTA
	# valid (expanded) around low diffusivity (aka low d_min)
	return ((7. * gamma**2 * delta * G**2) / (768 * D_0)) * d**4

def nilsson_dS_diff(d_small, d_big, D_0, delta, G):
	dS_small = nilsson_dS(d_small, D_0, delta, G)
	dS_big = nilsson_dS(d_big, D_0, delta, G)
	# big minus small because this is the difference from d=0 (aka S=1)
	# dS grows from 0 as diameter increases
	return dS_big - dS_small

def vangelderen_diff(d_small, d_big, D_0, delta, G):
	S_small = vg.vangelderen_cylinder_perp(D_0, 0.5*d_small, np.array([G, delta, delta]), m_max=10)
	S_big = vg.vangelderen_cylinder_perp(D_0, 0.5*d_big, np.array([G, delta, delta]), m_max=10)
	# small minus big because this the signals
	# Signal falls from 1 as diameters increases
	return S_small - S_big


# invivo diffusivity
# D0 = 2e-9
D0 = 0.66e-9
# most sensitive Connectom-like scheme
scheme_connectom = np.array([[0.3, 40e-3, 40e-3]])
# alpha significance level
alpha = 0.05
# data SNR at B0
# SNR = 30.0
SNR = 300.0

#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi


# compute sigma_bar for the diameter limit formula
sigmab = norm().ppf(1-alpha) / SNR


# nilsson_diameter(sigmab, D0, scheme_connectom[0,2] , scheme_connectom[0,0])*1e6
# nilsson_dS(2e-6, D0, scheme_connectom[0,2] , scheme_connectom[0,0])


d_dict = np.arange(0.01,5.01,0.01)*1e-6
S_nilsson = np.array([1-nilsson_dS(d, D0, scheme_connectom[0,2] , scheme_connectom[0,0]) for d in d_dict])
S_vg = np.array([vg.vangelderen_cylinder_perp(D0, 0.5*d, scheme_connectom, m_max=50) for d in d_dict])[:,0]



nilsson_errbar = np.zeros((2, len(d_dict)))
for i,d in enumerate(d_dict):
	# Nilsson
	# lower diameter
	target = S_nilsson[i] + sigmab
	it = np.argmin(np.abs(S_nilsson - target))
	nilsson_errbar[0,i] = d - d_dict[it]
	# higer diameter
	target = S_nilsson[i] - sigmab
	it = np.argmin(np.abs(S_nilsson - target))
	nilsson_errbar[1,i] = d_dict[it] - d

vg_errbar = np.zeros((2, len(d_dict)))
for i,d in enumerate(d_dict):
	# Vangelderen
	# lower diameter
	target = S_vg[i] + sigmab
	it = np.argmin(np.abs(S_vg - target))
	vg_errbar[0,i] = d - d_dict[it]
	# higer diameter
	target = S_vg[i] - sigmab
	it = np.argmin(np.abs(S_vg - target))
	vg_errbar[1,i] = d_dict[it] - d




pl.figure()
pl.errorbar(np.array(d_dict)*1e6, np.array(d_dict)*1e6, yerr=nilsson_errbar*1e6)
pl.title('Nilsson   sigma_bar = {:.3f}'.format(sigmab))
pl.gca().set_aspect('equal')

pl.xlim([0, 1.05*np.max(d_dict)*1e6])
pl.ylim([0, 1.05*np.max(d_dict)*1e6])


pl.figure()
pl.errorbar(np.array(d_dict)*1e6, np.array(d_dict)*1e6, yerr=vg_errbar*1e6)
pl.title('Vangelderen   sigma_bar = {:.3f}'.format(sigmab))
pl.gca().set_aspect('equal')

pl.xlim([0, 1.05*np.max(d_dict)*1e6])
pl.ylim([0, 1.05*np.max(d_dict)*1e6])

pl.show()







