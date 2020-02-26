import numpy as np
import pylab as pl






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













# data recovered from plots
data = np.genfromtxt('/home/raid2/paquette/Downloads/Liewald_2014_fig9_Human_brain_1_left.csv')

# the bin center are known so we can correct
diams = np.round(data[:,0], 1)
# re-normalize to sum to 1
normCounts = data[:,1] / data[:,1].sum()

# volume weight
W = ((normCounts*((diams/2.)**2)) / (normCounts*((diams/2.)**2)).sum())
# so-called "axon diameter index" a'
a_prime = (W * diams).sum()


pl.figure()
pl.title('Liewald 2014, Fig9, Human Brain 1 - Left   a\' = {:.3f}'.format(a_prime))
for i in range(diams.shape[0]):
    pl.bar(diams[i], normCounts[i], 0.05, color='k')
pl.show()

# data recovered from plots
data = np.genfromtxt('/home/raid2/paquette/Downloads/Liewald_2014_fig9_Human_brain_1_right.csv')

# the bin center are known so we can correct
diams = np.round(data[:,0], 1)
# re-normalize to sum to 1
normCounts = data[:,1] / data[:,1].sum()

# volume weight
W = ((normCounts*((diams/2.)**2)) / (normCounts*((diams/2.)**2)).sum())
# so-called "axon diameter index" a'
a_prime = (W * diams).sum()


pl.figure()
pl.title('Liewald 2014, Fig9, Human Brain 1 - right   a\' = {:.3f}'.format(a_prime))
for i in range(diams.shape[0]):
    pl.bar(diams[i], normCounts[i], 0.05, color='k')
pl.show()

















