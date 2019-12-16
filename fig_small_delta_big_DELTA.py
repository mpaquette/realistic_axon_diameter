# Van Gelderen formula for diffusion in cylinders
import numpy as np
from scipy.special import jnp_zeros

import pylab as pl

from vangelderen import vangelderen_cylinder_perp
from scheme import expand_scheme, remove_unphysical

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'




D_invivo = 2e-9

list_G = [0.3]
list_DELTA = np.linspace(10e-3, 50e-3, 9)
list_delta = np.linspace(5e-3, 40e-3, 15)
scheme = expand_scheme(list_G, list_DELTA, list_delta)
# scheme = remove_unphysical(scheme)


textfs = 20


pl.figure()

pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)

Rs = np.linspace(0.5e-6, 3.0e-6, 6)
n = np.sqrt(len(Rs))
ny = int(np.ceil(n))
nx = int(np.floor(n))
if nx*ny < len(Rs):
	nx += 1

pl.suptitle(r'VG for multiple ($\Delta$, $\delta$) and Radius, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)

for iR,R in enumerate(Rs):
	S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

	pl.subplot(nx, ny, iR+1)
	pl.imshow(S.reshape(len(list_DELTA), len(list_delta)), interpolation='nearest', extent=[1e3*list_delta.min(),1e3*list_delta.max(),1e3*list_DELTA.max(),1e3*list_DELTA.min()])
	cbar = pl.colorbar()
	cbar.ax.tick_params(labelsize=textfs-4)
	pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
	pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
	pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
	pl.xticks(fontsize=textfs-4)
	pl.yticks(fontsize=textfs-4)

pl.show()











D_invivo = 2e-9

list_G = [0.3]
list_DELTA = np.linspace(5e-3, 15e-3, 96)
list_delta = [5e-3]
scheme = expand_scheme(list_G, list_DELTA, list_delta)


Rs = np.linspace(0.5e-6, 3.0e-6, 6)


pl.figure()

for R in Rs:
	S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)
	pl.plot(1e3*list_DELTA, S, label='R = {:.1f} um'.format(R*1e6))
pl.xlabel('DELTA (ms)')
pl.ylabel('Signal')
pl.title('delta = {} ms'.format(list_delta[0]*1e3))

pl.show()










D_invivo = 2e-9

list_G = [0.3]
list_DELTA = np.linspace(5e-3, 15e-3, 96)
list_delta = [5e-3]
scheme = expand_scheme(list_G, list_DELTA, list_delta)


Rs = np.linspace(0.5e-6, 3.0e-6, 6)


pl.figure()

for R in Rs:
	S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)
	pl.plot(1e3*list_DELTA, 1 - (S / S[0]), label='R = {:.1f} um'.format(R*1e6))

pl.axhline(0.01)
pl.xlabel('DELTA (ms)')
pl.ylabel('Signal')
pl.title('delta = {} ms'.format(list_delta[0]*1e3))
pl.legend()

pl.show()







