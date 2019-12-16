import numpy as np

import pylab as pl





# assume gaussian 1D diffusion
# assume gaussian noise 
# S_0 == 1
# S = exp(-b D) + noise
# noise ~ N(mu=0, sigma^2)
# S ~ N(exp(-b D), sigma^2)


# D = - ln(S) / b

# um^2 / ms
D = 0.7 

# ms / um^2
b = 2.0


def S_gauss_1D(b, D, S0=1.0):
	return S0*np.exp(-b*D)


def fit_gauss_1D(b, S, S0=1.0):
	return -np.log(S/S0)/b



sigma = 0.05

Ntrial = 10000


data = []
fit_data = np.zeros(Ntrial)


noiseless_data = S_gauss_1D(b, D, S0=1.0)
for ii in range(Ntrial):
	noisydata = noiseless_data + sigma*np.random.randn()
	data.append(noisydata)
	fit_D = fit_gauss_1D(b, noisydata, S0=1.0)
	fit_data[ii] = fit_D


errors = D - fit_data


pl.figure()
# pl.hist(fit_data, 100, alpha=0.5, density=True)
pl.hist(errors, 100, alpha=0.5, density=True)


# samples_from_ln = -np.log(noiseless_data) + (sigma/noiseless_data)*np.random.randn(Ntrial) / b
samples_from_ln = -(sigma/noiseless_data)*np.random.randn(Ntrial) / b

# pl.figure()
pl.hist(samples_from_ln, 100, alpha=0.5, density=True)

pl.show()



# The approximation make sense!
# Maclaurin serie
# ln(a - x) = ln(a) - sum_i=1^inf x^k / k*a^k
# therefore
# ln(a - x) ~ ln(a) - x/a
# if x ~ N(0, sigma^2)
# then (a - x) ~ N(a, sigma^2)
# and ln(a - x) approx ~ N(ln(a), (sigma/a)^2)











