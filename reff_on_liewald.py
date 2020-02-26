import numpy as np

import vangelderen as vg

import pylab as pl


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'


# param
D0 = 2.0e-9

# acq param
delta = 40 # ms
Delta = 40 # ms
GMAX = 0.3 # T/m


# define dico
d_dict = 1e-6*np.arange(0.01,5.01,0.01)[::-1]
S_vg = np.array([vg.vangelderen_cylinder_perp(D0, 0.5*d, np.array([[GMAX, Delta*1e-3, delta*1e-3]]), m_max=50) for d in d_dict])[:,0]



def estimate_diameter_from_dict(value, dico_signal=S_vg, dico_radius=d_dict):
	# assume dico_signal is sorted ascending
	# return linear interpolated radius

	# dico_signal[idx-1] < value <= dico_signal[idx]
	idx = np.searchsorted(S_vg, value, 'left')
	s_low = dico_signal[idx-1]
	s_high = dico_signal[idx]
	r_low = dico_radius[idx-1]
	r_high = dico_radius[idx]

	k = (value - s_low) / (s_high - s_low)
	return r_low + (r_high - r_low)*k


def r_eff(counts, diams):
	# normalize counts
	normCounts = counts / counts.sum()
	# # compute areas
	# areas = np.pi*(diams/2.)**2
	# compute <r^6>
	r_mom6 = (normCounts*(diams/2.)**6).sum()
	# compute <r^2>
	r_mom2 = (normCounts*(diams/2.)**2).sum()
	# return r_eff = (<r^6>/<r^2>)^1/4
	return (r_mom6/r_mom2)**0.25


def d_mean(counts, diams):
	return (counts*diams).sum()

def d_mean_V(counts, diams):
	areas =  np.pi*(diams/2.)**2
	return (counts*areas*diams).sum() / (normCounts*areas).sum()



# Liewald diameter data
# data recovered from plots
data = np.genfromtxt('/home/raid2/paquette/Downloads/Liewald_2014_fig9_Human_brain_1_left.csv')

# the bin center are known so we can correct
diams = np.round(data[:,0], 1)
# re-normalize to sum to 1
normCounts = data[:,1] / data[:,1].sum()

# gen signal for distribution
# raw data
data_unmerged = np.array([vg.vangelderen_cylinder_perp(D0, 0.5e-6*d, np.array([[GMAX, Delta*1e-3, delta*1e-3]]), m_max=50) for d in diams])[:,0]
# areas
areas =  np.pi*(diams/2.)**2
# signal
signal = (normCounts*areas*data_unmerged).sum() / (normCounts*areas).sum()

reff = r_eff(normCounts, diams*1e-6)

meanD = d_mean(normCounts, diams)
meanD_vol = d_mean_V(normCounts, diams)

d_fitted = estimate_diameter_from_dict(signal)




pl.figure(figsize=(10, 5))
mycolormap = pl.cm.hsv
n = 3 + 1
_colors = (mycolormap(i) for i in np.linspace(0, 1, n))
# pl.title('Liewald 2014, Fig9, Human Brain 1 - Left', fontsize=20, )
for i in range(diams.shape[0]):
    pl.bar(diams[i], normCounts[i], 0.1, color='gray', edgecolor='black', linewidth=4)
pl.axvline(d_fitted*1e6, label=r'$d_{{\text{{fit}}}}$ = {:.2f} $\mu$m'.format(d_fitted*1e6), color=next(_colors), linewidth=4)
pl.axvline(2*reff*1e6, label=r'$d_{{\text{{eff}}}}$ = {:.2f} $\mu$m'.format(2*reff*1e6), color=next(_colors), linewidth=4, linestyle='--')
# pl.axvline(meanD_vol, label=r'vol W mean D = {:.2f} $\mu$m'.format(meanD_vol), color=next(_colors), linewidth=4)
pl.axvline(meanD, label=r'$d_{{\text{{mean}}}}$ = {:.2f} $\mu$m'.format(meanD), color=next(_colors), linewidth=4)
pl.legend(fontsize=18)
pl.xlabel(r'Diameters ($\mu$m)', fontsize=16)
pl.ylabel(r'Normalized Axon Counts', fontsize=16)
pl.xticks(fontsize=14)
pl.yticks(fontsize=14)
pl.show()





# Liewald diameter data
# data recovered from plots
data = np.genfromtxt('/home/raid2/paquette/Downloads/Liewald_2014_fig9_Human_brain_1_right.csv')

# the bin center are known so we can correct
diams = np.round(data[:,0], 1)
# re-normalize to sum to 1
normCounts = data[:,1] / data[:,1].sum()

# gen signal for distribution
# raw data
data_unmerged = np.array([vg.vangelderen_cylinder_perp(D0, 0.5e-6*d, np.array([[GMAX, Delta*1e-3, delta*1e-3]]), m_max=50) for d in diams])[:,0]
# areas
areas =  np.pi*(diams/2.)**2
# signal
signal = (normCounts*areas*data_unmerged).sum() / (normCounts*areas).sum()

reff = r_eff(normCounts, diams*1e-6)

meanD = d_mean(normCounts, diams)
meanD_vol = d_mean_V(normCounts, diams)

d_fitted = estimate_diameter_from_dict(signal)




pl.figure(figsize=(10, 5))
mycolormap = pl.cm.hsv
n = 3 + 1
_colors = (mycolormap(i) for i in np.linspace(0, 1, n))
# pl.title('Liewald 2014, Fig9, Human Brain 1 - Right', fontsize=20)
for i in range(diams.shape[0]):
    pl.bar(diams[i], normCounts[i], 0.1, color='gray', edgecolor='black', linewidth=4)
pl.axvline(d_fitted*1e6, label=r'$d_{{\text{{fit}}}}$ = {:.2f} $\mu$m'.format(d_fitted*1e6), color=next(_colors), linewidth=4)
pl.axvline(2*reff*1e6, label=r'$d_{{\text{{eff}}}}$ = {:.2f} $\mu$m'.format(2*reff*1e6), color=next(_colors), linewidth=4, linestyle='--')
# pl.axvline(meanD_vol, label=r'vol W mean D = {:.2f} $\mu$m'.format(meanD_vol), color=next(_colors), linewidth=4)
pl.axvline(meanD, label=r'$d_{{\text{{mean}}}}$ = {:.2f} $\mu$m'.format(meanD), color=next(_colors), linewidth=4)
pl.legend(fontsize=18)
pl.xlabel(r'Diameters ($\mu$m)', fontsize=16)
pl.ylabel(r'Normalized Axon Counts', fontsize=16)
pl.xticks(fontsize=14)
pl.yticks(fontsize=14)
pl.show()






