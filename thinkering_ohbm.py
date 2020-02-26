import numpy as np

import pylab as pl





# assume gaussian 1D diffusion
# assume gaussian noise 
# S_0 == 1
# S = exp(-b D) + noise
# noise ~ N(mu=0, sigma^2)
# S ~ N(exp(-b D), sigma^2)


# D = - ln(S) / b

# # um^2 / ms
# D = 1.0 
# # ms / um^2
# b = 2.0



# bD in [0.1, 10]
bD = np.linspace(0.1, 10, 100)
# 
sigma = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
# for each pair
Ntrial = 10000
# clip negatives to
eps = 1e-16


def S_gauss_1D(b, D, S0=1.0):
	return S0*np.exp(-b*D)


def fit_gauss_1D(b, S, S0=1.0):
	return -np.log(S/S0)/b



signals = np.zeros((bD.shape[0], sigma.shape[0], Ntrial))
bDs = np.zeros((bD.shape[0], sigma.shape[0], Ntrial))


for i_bD, x_bD in enumerate(bD):
	print('{} / {}'.format(i_bD, bD.shape[0]))
	noiseless_data = S_gauss_1D(1.0, x_bD, S0=1.0)
	for i_sig, x_sig in enumerate(sigma):
		for ii in range(Ntrial):
			noisydata = noiseless_data + x_sig*np.random.randn()
			signals[i_bD, i_sig, ii] = noisydata
			fit_D = fit_gauss_1D(1, max(noisydata, eps), S0=1.0)
			bDs[i_bD, i_sig, ii] = fit_D







pl.figure()
pl.imshow(np.log(signals.mean(2).T), interpolation='Nearest')
pl.title('ln(mean signals)')
pl.colorbar()


pl.figure()
pl.imshow((signals<eps).sum(2).T / sigma[:,None], interpolation='Nearest')
pl.title('# invalid div sigma')
pl.colorbar()



pl.figure()
pl.imshow(bDs.mean(2).T / bD, interpolation='Nearest')
pl.title('Average bD div true bD')
pl.colorbar()


pl.show()





# um^2 / ms
D = 0.7 
# ms / um^2
bs = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
D_errors = np.array([0.001, 0.005, 0.01, 0.05, 0.1])


data = np.zeros((bs.shape[0], D_errors.shape[0], 2))


for ib,b in enumerate(bs):
	noiseless_data = S_gauss_1D(b, D, S0=1.0)
	for ierr,err in enumerate(D_errors):
		wrong_data_plus = S_gauss_1D(b, D*(1+err), S0=1.0)
		data[ib, ierr, 0] = noiseless_data - wrong_data_plus
		wrong_data_minus = S_gauss_1D(b, D*(1-err), S0=1.0)
		data[ib, ierr, 1] = noiseless_data - wrong_data_minus






















# errors = D - fit_data


# pl.figure()
# # pl.hist(fit_data, 100, alpha=0.5, density=True)
# pl.hist(errors, 100, alpha=0.5, density=True)


# # samples_from_ln = -np.log(noiseless_data) + (sigma/noiseless_data)*np.random.randn(Ntrial) / b
# samples_from_ln = -(sigma/noiseless_data)*np.random.randn(Ntrial) / b

# # pl.figure()
# pl.hist(samples_from_ln, 100, alpha=0.5, density=True)

# pl.show()



# # The approximation make sense!
# # Maclaurin serie
# # ln(a - x) = ln(a) - sum_i=1^inf x^k / k*a^k
# # therefore
# # ln(a - x) ~ ln(a) - x/a
# # if x ~ N(0, sigma^2)
# # then (a - x) ~ N(a, sigma^2)
# # and ln(a - x) approx ~ N(ln(a), (sigma/a)^2)











