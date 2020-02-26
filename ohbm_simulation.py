import numpy as np
import vangelderen as vg
import pylab as pl
from scipy.stats import norm


def diameter_intersection(point, values, dico):
	# find the diameter in dico corresponding to the intersection of point and values

	# locate the approximate location in dico
	smaller_idx = np.where(values >= point)[0]
	# test diameter smaller then dico minimum
	if len(smaller_idx) == 0:
		return dico[0]
	smaller_max = smaller_idx[-1]
	bigger_idx = np.where(values < point)[0]
	# test diameter bigger then dico maximum
	if len(bigger_idx) == 0:
		return dico[-1]
	bigger_min = bigger_idx[0]

	# linear interpolation
	# a >= P > b ==> a: values[smaller_max]  b: values[bigger_min]  p: point
	# p = a - R (a-b)
	# R = a-p / a-b
	R = (values[smaller_max] - point) / (values[smaller_max] - values[bigger_min])
	diameter = dico[smaller_max] +  R*(dico[bigger_min]-dico[smaller_max])

	return diameter


# Setup

# Both scheme use delta = DELTA because extra DELTA bring no measurable contrast in restricted diffusion
# very strong and long Connectom scheme, following Nilsson's 
scheme_1 = np.array([[0.3, 40e-3, 40e-3]])
# strong ish, normal preclincal scheme, ~Dyrby
scheme_2 = np.array([[0.15, 10e-3, 10e-3]])
scheme_names = ['Nilsson', 'Dyrby']
schemes = [scheme_1, scheme_2]


# invivo diffusivity
D0 = 2e-9


# stat test level
alpha = 0.05
# data SNR at B0
SNRs = [30.0, 164.0]


# compute sigma_bar for the diameter limit formula
sigmabs = norm().ppf(1-alpha) / SNRs



# build signal dictionary
d_dict = np.arange(0.01,6.51,0.01)*1e-6
S_vg = [np.array([vg.vangelderen_cylinder_perp(D0, 0.5*d, scheme, m_max=50) for d in d_dict])[:,0] for scheme in schemes]




# crappy logging of value
log = {}

logy = []
vin = 1.0

pl.figure()
fs = 18
# color cycling hack
# colmap = pl.cm.jet
colmap = pl.cm.gist_rainbow
colors = colmap(np.linspace(0, 1, len(schemes)+len(SNRs)))

for i in range(len(schemes)):
	pl.plot(d_dict*1e6, vin*S_vg[i], label=scheme_names[i], color=colors[i]) # colors from 0 to len(schemes)
for i in range(len(SNRs)):
	pl.axhline(vin-sigmabs[i], label='detection Threshold for SNR = {:.0f}'.format(SNRs[i]), color=colors[i+len(schemes)]) # colors len(schemes) to len(schemes)+len(SNRs)

for i in range(len(schemes)):
	for j in range(len(SNRs)):
		diam_limit = diameter_intersection(vin-sigmabs[j], vin*S_vg[i], d_dict)
		# pl.axvline(diam_limit*1e6, color='red', alpha=0.3)
		pl.annotate('{:.2f} um'.format(diam_limit*1e6), xy=(diam_limit*1e6, vin-sigmabs[j]), xytext=(diam_limit*1e6, (vin-sigmabs[j])-0.015), fontsize=fs-4)
		logy.append(diam_limit)


pl.xlabel('Axon Diameter (um)', fontsize=fs)
pl.ylabel('Normalized signal (S/S_0)', fontsize=fs)
pl.title('Signal curve (for {:.0f}% f_in) and Threshold (alpha = {:.0f}%)'.format(100*vin, 100*alpha), fontsize=fs+4)
pl.legend(fontsize=fs)
pl.xticks(fontsize=fs-2)
pl.yticks(fontsize=fs-2)
pl.show()

log[vin] = np.array(logy)




logy = []
vin = 0.7

pl.figure()
fs = 18
# color cycling hack
# colmap = pl.cm.jet
colmap = pl.cm.gist_rainbow
colors = colmap(np.linspace(0, 1, len(schemes)+len(SNRs)))

for i in range(len(schemes)):
	pl.plot(d_dict*1e6, vin*S_vg[i], label=scheme_names[i], color=colors[i]) # colors from 0 to len(schemes)
for i in range(len(SNRs)):
	pl.axhline(vin-sigmabs[i], label='detection Threshold for SNR = {:.0f}'.format(SNRs[i]), color=colors[i+len(schemes)]) # colors len(schemes) to len(schemes)+len(SNRs)

for i in range(len(schemes)):
	for j in range(len(SNRs)):
		diam_limit = diameter_intersection(vin-sigmabs[j], vin*S_vg[i], d_dict)
		# pl.axvline(diam_limit*1e6, color='red', alpha=0.3)
		pl.annotate('{:.2f} um'.format(diam_limit*1e6), xy=(diam_limit*1e6, vin-sigmabs[j]), xytext=(diam_limit*1e6, (vin-sigmabs[j])-0.015), fontsize=fs-4)
		logy.append(diam_limit)


pl.xlabel('Axon Diameter (um)', fontsize=fs)
pl.ylabel('Normalized signal (S/S_0)', fontsize=fs)
pl.title('Signal curve (for {:.0f}% f_in) and Threshold (alpha = {:.0f}%)'.format(100*vin, 100*alpha), fontsize=fs+4)
pl.legend(fontsize=fs)
pl.xticks(fontsize=fs-2)
pl.yticks(fontsize=fs-2)
pl.show()

log[vin] = np.array(logy)




fins = np.linspace(0.1,1,100)
pl.figure()
pl.plot(fins, fins**(-1/4.), linewidth=3)
pl.xlabel('Intra Axonal Signal Fraction', fontsize=18)
pl.ylabel('d_min multiplier', fontsize=18)
pl.title('d_min multiplier as a function of f_in', fontsize=20)
pl.show()



































