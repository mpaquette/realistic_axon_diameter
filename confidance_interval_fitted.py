import numpy as np
import vangelderen as vg
import pylab as pl
from scipy.stats import norm, bayes_mvs, gaussian_kde

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'

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
D0 = 2e-9
# D0 = 0.66e-9
# most sensitive Connectom-like scheme
scheme_connectom = np.array([[0.3, 40e-3, 40e-3]])
# scheme_connectom = np.array([[1.5, 40e-3, 40e-3]])
# alpha significance level
# alpha = 0.01
# alphas = np.array([0.001, 0.01, 0.05])
alphas = np.array([0.01])
# data SNR at B0
# SNR = 30.0
SNR = 300

#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi


# compute sigma_bar for the diameter limit formula
# sigmab = norm().ppf(1-alpha) / SNR
sigmabs = norm().ppf(1-alphas) / SNR


# d_min = nilsson_diameter(sigmab, D0, scheme_connectom[0,2], scheme_connectom[0,0])
d_mins = nilsson_diameter(sigmabs, D0, scheme_connectom[0,2], scheme_connectom[0,0])



# diameters in m
# diams = np.arange(0.1, 5.1, 0.1)*1e-6
diams = np.arange(0.1, 5.1, 0.05)*1e-6
# number of noise trial
# Ntrial = 100
# Ntrial = 1000
Ntrial = 10000


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





### computing the mean/std for each radii 
alpha_B = 0.95
bcv = []
for i in range(fit_data.shape[0]):
	bcv.append(bayes_mvs(fit_data[i], alpha_B))


bayes_mean_stat = np.array([s[0].statistic for s in bcv])
bayes_mean_stat_min = np.array([s[0].minmax[0] for s in bcv])
bayes_mean_stat_max = np.array([s[0].minmax[1] for s in bcv])

bayes_std_stat = np.array([s[2].statistic for s in bcv])
bayes_std_stat_min = np.array([s[2].minmax[0] for s in bcv])
bayes_std_stat_max = np.array([s[2].minmax[1] for s in bcv])



# pl.figure()

# mycolormap = pl.cm.hsv
# n = 3 + len(alphas) + 1
# _colors = (mycolormap(i) for i in np.linspace(0, 1, n))

# pl.plot(diams*1e6, diams*1e6, color=next(_colors), alpha=0.75, linestyle='--')


# for d_i, d_min in enumerate(d_mins):
# 	pl.axvline(d_min*1e6, color=next(_colors), label=r'$d_{{\min}}$ = {:.2f} $\mu$m ({:.1f} \% decay with $\alpha$ = {})'.format(d_min*1e6, 100*sigmabs[d_i], alphas[d_i]))

# color1 = next(_colors)
# pl.plot(diams*1e6, bayes_mean_stat*1e6, color=color1, label='Mean')
# pl.fill_between(diams*1e6, bayes_mean_stat_min*1e6, bayes_mean_stat_max*1e6, color=color1, alpha=0.2)

# color2 = next(_colors)
# pl.plot(diams*1e6, bayes_std_stat*1e6, color=color2, label='Std')
# pl.fill_between(diams*1e6, bayes_std_stat_min*1e6, bayes_std_stat_max*1e6, color=color2, alpha=0.2)



# pl.title('Mean with {:.0f}\% confidence interval over {} samples'.format(100*alpha_B, Ntrial), fontsize=16)
# pl.gca().set_aspect('equal')
# pl.legend()
# pl.show()


## validating that bayes mean/std == gaussian moment one for slightly nonsmall N
# pl.figure()

# pl.plot(diams*1e6, bayes_mean_stat*1e6, color='blue', label='Mean-B')
# pl.plot(diams*1e6, fit_data.mean(axis=1)*1e6, color='blue', label='Mean', linestyle='--')

# pl.plot(diams*1e6, bayes_std_stat*1e6, color='green', label='Std-B')
# pl.plot(diams*1e6, fit_data.std(axis=1)*1e6, color='green', label='Std', linestyle='--')

# pl.show()



interval = 0.8

# ## estimating data peak to get a left and right side X% interval
# smooth_diam = np.linspace(0, fit_data.max()*1.01, 10000)


# peak_diams = np.zeros(fit_data.shape[0])
# lower_diams = np.zeros(fit_data.shape[0])
# upper_diams = np.zeros(fit_data.shape[0])


# for i in range(fit_data.shape[0]):

# 	gkde = gaussian_kde(fit_data[i])
# 	smoothed = gkde.pdf(smooth_diam)

# 	# sanity check of KDE fit
# 	# pl.figure()
# 	# pl.hist(fit_data[i], 100, density=True)
# 	# pl.plot(smooth_diam, smoothed)
# 	# pl.show()


# 	peak_diams[i] = smooth_diam[smoothed.argmax()]
# 	lower_diams[i] = np.quantile(fit_data[i][fit_data[i]<=peak_diams[i]], 1-interval/2)
# 	upper_diams[i] = np.quantile(fit_data[i][fit_data[i]>=peak_diams[i]], interval/2)

# 	print('D = {:.2f}   Range = {:.2f}'.format(diams[i]*1e6, (upper_diams[i]-lower_diams[i])*1e6))



# pl.figure()
# color1 = 'blue'
# pl.plot(diams*1e6, peak_diams*1e6, color=color1, label='Peak')
# pl.fill_between(diams*1e6, lower_diams*1e6, upper_diams*1e6, color=color1, alpha=0.2)

# # # sanity check for skewness
# # pl.plot(diams*1e6, ((upper_diams-peak_diams)-(peak_diams-lower_diams))*1e6)

# pl.show()




# ## estimating data median to get a left and right side X% interval
# interval = 0.8
# peak_diams_med = np.zeros(fit_data.shape[0])
# lower_diams_med = np.zeros(fit_data.shape[0])
# upper_diams_med = np.zeros(fit_data.shape[0])

# for i in range(fit_data.shape[0]):

# 	peak_diams_med[i] = np.quantile(fit_data[i], 0.5)
# 	lower_diams_med[i] = np.quantile(fit_data[i][fit_data[i]<=peak_diams_med[i]], 1-interval)
# 	upper_diams_med[i] = np.quantile(fit_data[i][fit_data[i]>=peak_diams_med[i]], interval)

# 	print('D = {:.2f}   Range = {:.2f}'.format(diams[i]*1e6, (upper_diams_med[i]-lower_diams_med[i])*1e6))



# pl.figure()
# color1 = 'blue'
# pl.plot(diams*1e6, peak_diams_med*1e6, color=color1, label='Peak')
# pl.fill_between(diams*1e6, lower_diams_med*1e6, upper_diams_med*1e6, color=color1, alpha=0.2)

# # # sanity check for skewness
# # pl.plot(diams*1e6, ((upper_diams_med-peak_diams_med)-(peak_diams_med-lower_diams_med))*1e6)

# pl.show()




## estimating data median to get a left and right side X% interval
interval = 0.8
peak_diams_mean = np.zeros(fit_data.shape[0])
lower_diams_mean = np.zeros(fit_data.shape[0])
upper_diams_mean = np.zeros(fit_data.shape[0])

for i in range(fit_data.shape[0]):

	peak_diams_mean[i] = bayes_mean_stat[i]
	lower_diams_mean[i] = np.quantile(fit_data[i][fit_data[i]<=bayes_mean_stat[i]], 1-interval)
	upper_diams_mean[i] = np.quantile(fit_data[i][fit_data[i]>=bayes_mean_stat[i]], interval)

	# print('D = {:.2f}   Range = {:.2f}'.format(diams[i]*1e6, (upper_diams_med[i]-lower_diams_med[i])*1e6))







pl.figure(figsize=(10,10))
mycolormap = pl.cm.hsv
n = 4 + len(d_mins) + 1
_colors = (mycolormap(i) for i in np.linspace(0, 1, n))



# pl.scatter(np.repeat(diams, Ntrial)*1e6, fit_data.ravel()*1e6, color=next(_colors), alpha=0.01, edgecolors="none")
# pl.scatter(np.repeat(diams, Ntrial)*1e6, fit_data.ravel()*1e6, color='red', alpha=0.01, edgecolors="none")
# adding left right jitter
jitter_intensity = 0.25
step = (diams[1:] - diams[:-1]).mean()
jitter = (0.5-np.random.rand(Ntrial*diams.shape[0]))*step*jitter_intensity
pl.scatter((np.repeat(diams, Ntrial)+jitter)*1e6, fit_data.ravel()*1e6, color='red', alpha=0.01, edgecolors="none")


# pl.plot(diams*1e6, diams*1e6, color=next(_colors))
pl.plot(diams*1e6, diams*1e6, color='black', linestyle='--')

# pl.fill_between(diams*1e6, lower_diams_mean*1e6, upper_diams_mean*1e6, color=color1, alpha=0.2)
# color1=next(_colors)
color1='blue'
# pl.plot(diams*1e6, lower_diams_mean*1e6, color=color1, linewidth=3, label=r'{:.0f}\% Confidence Interval'.format(100*interval))
idx_trunc = np.where(lower_diams_mean > 0)[0][0]
pl.plot(diams[idx_trunc-1:]*1e6, lower_diams_mean[idx_trunc-1:]*1e6, color=color1, linewidth=3, label=r'{:.0f}\% Confidence Interval (mean)'.format(100*interval))
pl.plot(diams*1e6, upper_diams_mean*1e6, color=color1, linewidth=3)

# pl.plot(diams*1e6, bayes_mean_stat*1e6, color=next(_colors), linewidth=2, label=r'Mean $d_{{\text{{fit}}}}$')
# pl.plot(diams*1e6, bayes_mean_stat*1e6, color='lime', linewidth=2, label=r'Mean $d_{{\text{{fit}}}}$')
pl.plot(diams*1e6, bayes_mean_stat*1e6, color='lime', linewidth=2)


for d_i, d_min in enumerate(d_mins):
	# pl.axvline(d_min*1e6, color=next(_colors), label=r'$d_{{\min}}$ = {:.2f} $\mu$m ({:.1f} \% decay with $\alpha$ = {})'.format(d_min*1e6, 100*sigmabs[d_i], alphas[d_i]))
	pl.axvline(d_min*1e6, color='fuchsia', label=r'$d_{{\min}}$ = {:.2f} $\mu$m ({:.1f} \% decay with $\alpha$ = {})'.format(d_min*1e6, 100*sigmabs[d_i], alphas[d_i]), linewidth=2)


pl.xlim([0, 1.05*np.max(diams)*1e6])
pl.ylim([0, 1.05*np.max(diams)*1e6])

pl.xlabel(r'True diameters ($\mu$m)', fontsize=20)
pl.ylabel(r'Fitted diameters ($\mu$m)', fontsize=20)

pl.xticks(fontsize=16)
pl.yticks(fontsize=16)

pl.gca().set_aspect('equal')

pl.legend(loc=2, fontsize=18)

pl.title('SNR = {:.0f}'.format(SNR), fontsize=20)

pl.show()









fit_mean = fit_data.mean(axis=1)
fit_std = fit_data.std(axis=1)

fit_bias = fit_mean - diams



# pl.figure()
# pl.plot(diams*1e6, fit_bias*1e6)
# pl.xlabel('diam')
# pl.ylabel('bias')


# pl.figure()
# pl.plot(diams*1e6, fit_std*1e6)
# pl.xlabel('diam')
# pl.ylabel('std')


# pl.figure()
# pl.plot(diams*1e6, fit_mean*1e6)
# pl.xlabel('diam')
# pl.ylabel('mean')

# pl.show()




fit_err = fit_data - diams[:, None]
greatest_err_magn = np.max(np.abs(fit_err), axis=1)


th_greatest = 0.1
idx_greatest = np.where(greatest_err_magn/diams <= th_greatest)
lim_greatest = np.min(diams[idx_greatest])

th_greatest_abs = 0.25e-6
idx_greatest_abs = np.where(greatest_err_magn <= th_greatest_abs)
lim_greatest_abs = np.min(diams[idx_greatest_abs])




outlier_alpha = 0.01
outlier_z = norm().ppf(1-outlier_alpha/2.)
# test to see if we need to filter outlier (only keeping [-Z_{alpha/2}*std, +Z_{alpha/2}*std])

min_values = fit_err.mean(axis=1) - outlier_z*fit_err.std(axis=1)
max_values = fit_err.mean(axis=1) + outlier_z*fit_err.std(axis=1)


greatest_err_magn_fil = np.array([np.max(np.abs(fit_err[i][np.logical_and(fit_err[i]<=max_values[i], fit_err[i]>=min_values[i])])) for i in range(fit_err.shape[0])])

idx_greatest_fil = np.where(greatest_err_magn_fil/diams <= th_greatest)
lim_greatest_fil = np.min(diams[idx_greatest_fil])

idx_greatest_abs_fil = np.where(greatest_err_magn_fil <= th_greatest_abs)
lim_greatest_abs_fil = np.min(diams[idx_greatest_abs_fil])




# pl.figure()
# mycolormap = pl.cm.hsv
# n = 2 + 3 + 1
# _colors = (mycolormap(i) for i in np.linspace(0, 1, n))
# pl.scatter(np.repeat(diams, Ntrial)*1e6, fit_data.ravel()*1e6, color=next(_colors), alpha = 0.1, edgecolors="none")
# pl.plot(diams*1e6, diams*1e6, color=next(_colors))


# pl.axvline(d_min*1e6, color=next(_colors), label=r'$d_{{\min}}$ = {:.2f} $\mu$m ({:.1f} \% decay with $\alpha$ = {})'.format(d_min*1e6, 100*sigmabs[1], alphas[1]))

# # pl.axvline(lim_greatest*1e6, color=next(_colors), label=r'$d$ = {:.2f} $\mu$m with max error {:.1f} \%'.format(lim_greatest*1e6, 100*th_greatest))
# # pl.axvline(lim_greatest_fil*1e6, color=next(_colors), label=r'$d$ = {:.2f} $\mu$m with max error {:.1f} \% fil={:.1f}\%'.format(lim_greatest_fil*1e6, 100*th_greatest, 100*outlier_alpha))
# pl.axvline(lim_greatest_fil*1e6, color=next(_colors), label=r'$d$ = {:.2f} $\mu$m with max error {:.1f} \%'.format(lim_greatest_fil*1e6, 100*th_greatest))
# # pl.axvline(lim_greatest_abs*1e6, color=next(_colors), label=r'$d$ = {:.2f} $\mu$m with max error {:.2f} $\mu$m'.format(lim_greatest_abs*1e6, th_greatest_abs*1e6))
# # pl.axvline(lim_greatest_abs_fil*1e6, color=next(_colors), label=r'$d$ = {:.2f} $\mu$m with max error {:.2f} $\mu$m fil={:.1f}\%'.format(lim_greatest_abs_fil*1e6, th_greatest_abs*1e6, 100*outlier_alpha))
# pl.axvline(lim_greatest_abs_fil*1e6, color=next(_colors), label=r'$d$ = {:.2f} $\mu$m with max error {:.2f} $\mu$m'.format(lim_greatest_abs_fil*1e6, th_greatest_abs*1e6))

# pl.xlim([0, 1.05*np.max(diams)*1e6])
# pl.ylim([0, 1.05*np.max(diams)*1e6])

# pl.xlabel(r'True diameters ($\mu$m)', fontsize=20)
# pl.ylabel(r'Fitted diameters ($\mu$m)', fontsize=20)

# pl.gca().set_aspect('equal')

# pl.legend(fontsize=18)

# pl.title('SNR = {:.0f}'.format(SNR), fontsize=20)

# pl.show()










