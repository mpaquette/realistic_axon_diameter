import numpy as np
from scipy.special import ellipe as E2

def r2(a,b,t):
    return a**2*np.cos(t)**2 + b**2*np.sin(t)**2

# t = np.linspace(0, 2*np.pi, 16, endpoint=False) # full circle
t = np.linspace(0, np.pi, 16, endpoint=False) # half-circle, should be the same by symmetry
# t = np.linspace(0, np.pi/2, 16) # quarter circle, gives error because of the endpoint ambiguity

# mean_r2 = {}
mean_r = {}

def anal_r2(a,b):
    # return 0.5 * (a**2 + (1-b**2))
    return 0.5 * (a**2 + (1-b**2))

def anal_r_doc(a,b):
    c = a**2 - b**2
    return (a/2.) + (b**2 / (4.*c)) * np.log((a+c)/(a-c))


# mean_anal_r2 = {}
mean_anal_r = {}


for i in range(100):
    a = 10*np.random.rand()
    b = np.random.rand()*a
    # anal = anal_r2(a,b)
    anal = (2*a/np.pi)*E2(1-(b**2/a**2)) # this is the true solution
    num = r2(a,b,t)
    # mean_r2[(a,b)] = num.mean()
    mean_r[(a,b)] = np.sqrt(num).mean()
    # mean_anal_r2[(a,b)] = anal
    mean_anal_r[(a,b)] = anal


errors = []
ratios = []
As = []
Bs = []

# for k in mean_r2.keys():
for k in mean_r.keys():
	# num = mean_r2[k]
	num = mean_r[k]
	# anal = mean_anal_r2[k]
	anal = mean_anal_r[k]
	# print(np.abs(num-anal))
	# print(np.ceil(np.log10(np.abs(num-anal))), k[0], k[1])
	print('{}  {:.2f}'.format(np.ceil(np.log10(np.abs(num-anal))), k[0]/k[1]))
	errors.append(np.abs(num-anal))
	ratios.append(k[0]/k[1])
	As.append(k[0])
	Bs.append(k[1])

errors = np.array(errors)
ratios = np.array(ratios)
As = np.array(As)
Bs = np.array(Bs)

pl.figure()
pl.scatter(ratios, np.log(errors))
pl.figure()
pl.scatter(As, np.log(errors))
pl.figure()
pl.scatter(Bs, np.log(errors))
pl.show()
pl.scatter(As-Bs, np.log(errors))
pl.show()
pl.scatter((As-Bs)/As, np.log(errors))
pl.show()

y = np.log(errors)
y[np.isinf(y)] = y[np.logical_not(np.isinf(y))].min() - 1

pl.figure()
pl.scatter(ratios, np.log(errors))

pl.figure()
pl.scatter(ratios, np.log(y - y.min()))

pl.figure()
pl.scatter(ratios, np.exp(y))

pl.show()


import pylab as pl

# y1 = np.array(sorted(mean_r2.values()))
# y2 = np.array(sorted(mean_anal_r2.values()))
y1 = np.array(sorted(mean_r.values()))
y2 = np.array(sorted(mean_anal_r.values()))

pl.figure()
pl.plot(y1, label='num')
pl.plot(y2, label='anal')
pl.legend()
pl.show()





