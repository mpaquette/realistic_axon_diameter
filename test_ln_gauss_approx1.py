# approximating distribution of ln(normal)

import numpy as np
import scipy.stats as ss
import pylab as pl


mu = 0.5
sigma = 0.01

Ntrial = 10000

repeat = 10


for ii in range(repeat):

	# exponant uniform in [-3, 3]
	expo = -3 + 6*np.random.rand()
	# mantissa uniform in [0, 10]
	mant = 10*np.random.rand()

	# mean
	mu = mant*10**expo

	# exponant uniform in [expo-3.5, expo-1.5]
	expo = (expo-3.5) + 2*np.random.rand()
	# mantissa uniform in [0, 10]
	mant = 10*np.random.rand()

	# std
	sigma = mant*10**expo

	if mu - 4*sigma < 0:
		sigma = mu/4.1
		print('shit happens')


	# N(mu, sigma^2)
	samples = mu + sigma*np.random.randn(Ntrial)

	# pl.figure()
	# pl.hist(samples, 100)
	# pl.title('pdf of N({:.2f}, {:.2f}^2)  approximation with {} samples'.format(mu, sigma, Ntrial))
	# pl.show()


	ln_samples = np.log(samples)

	pl.figure()
	pl.hist(ln_samples, 100, alpha=0.5, label='ln(N)', density=True)
	pl.title('pdf of ln( N({:.2f}, {:.2f}^2) ) approximation with {} samples'.format(mu, sigma, Ntrial))
	# # pl.show()

	# N(ln(mu), (sigma/mu)^2)
	samples_from_ln = np.log(mu) + (sigma/mu)*np.random.randn(Ntrial)


	# pl.figure()
	pl.hist(samples_from_ln, 100, alpha=0.5, label='fake N', density=True)
	# pl.title('pdf of N(ln(mu), (sigma/mu)^2) approximation with {} samples'.format(Ntrial))
	pl.legend()
	pl.show()

	# print('\n')
	# print(ln_samples.mean())
	# print(samples_from_ln.mean())
	# print(ln_samples.std())
	# print(samples_from_ln.std())

	# sval, pval = ss.ks_2samp(ln_samples, samples_from_ln)
	# print('v = {:.3f}    p = {:.3f}'.format(sval, pval))


	# pl.figure()
	# pl.hist(ln_samples, 100, density=True, histtype='step', cumulative=True, label='ln( N(mu, sigma^2) )')
	# pl.hist(samples_from_ln, 100, density=True, histtype='step', cumulative=True, label='N(ln(mu), (sigma/mu)^2)')
	# pl.legend(loc=2)
	# pl.show()


