# Van Gelderen formula for diffusion in cylinders
import numpy as np
from scipy.special import jnp_zeros

import pylab as pl

from vangelderen import vangelderen_cylinder_perp
from scheme import expand_scheme, remove_unphysical


D_invivo = 2e-9

list_G = [0.3]
list_DELTA = np.linspace(10e-3, 50e-3, 9)
list_delta = np.linspace(5e-3, 40e-3, 15)
scheme = expand_scheme(list_G, list_DELTA, list_delta)
# scheme = remove_unphysical(scheme)



pl.figure()
Rs = np.linspace(0.5e-6, 2.5e-6, 5)
n = np.sqrt(len(Rs))
ny = int(np.ceil(n))
nx = int(np.floor(n))
if nx*ny < len(Rs):
	nx += 1

for iR,R in enumerate(Rs):
	S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

	pl.subplot(nx, ny, iR+1)
	pl.imshow(S.reshape(len(list_DELTA), len(list_delta)), interpolation='nearest', extent=[1e3*list_delta.min(),1e3*list_delta.max(),1e3*list_DELTA.max(),1e3*list_DELTA.min()])
	pl.title('R = {:.2f} um'.format(R*1e6))
	pl.xlabel('delta (ms)')
	pl.ylabel('DELTA (ms)')

pl.show()








D_invivo = 2e-9

list_G = [0.3]
list_DELTA = np.linspace(5e-3, 100e-3, 96)
list_delta = [5e-3]
scheme = expand_scheme(list_G, list_DELTA, list_delta)


R = 1.5e-6

S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

pl.figure()
pl.plot(1e3*list_DELTA, S)
pl.xlabel('DELTA (ms)')
pl.ylabel('Signal')
pl.title('R = {:.2f} um     delta = {} ms'.format(R*1e6, list_delta[0]*1e3))

pl.show()






