import numpy as np
from scipy.special import jnp_zeros
# import pylab as pl
import matplotlib.pyplot as pl
import matplotlib.colors as colors


#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi



def vangelderen_cylinder_perp_ln_list(D, R, DELTA, delta, G, m_max=10):
    # returns the scaling factor and the list of each summation component for ln(M(DELTA,delta,G)/M(0))
    # D:= free diffusivity in m^2 s^-1
    # R:= cylinder radius in m
    # DELTA:= gradient separation in s
    # delta:= gradient width s
    # G:= gradient magnitude in T m^-1
    am_R = jnp_zeros(1,m_max)
    am = am_R / R
    am2 = am**2
    fac = -2*gamma**2*G**2/D**2
    comp = (1/(am**6*(am_R**2-1))) * (2*D*am2*delta - 2 + 2*np.exp(-D*am2*delta) + 2*np.exp(-D*am2*DELTA) - np.exp(-D*am2*(DELTA-delta)) - np.exp(-D*am2*(DELTA+delta)))
    return fac, comp



def vangelderen_cylinder_perp_ln(D, R, DELTA, delta, G, m_max=5):
    fac, comp = vangelderen_cylinder_perp_ln_list(D, R, DELTA, delta, G, m_max)
    return fac*np.sum(comp)


def vangelderen_cylinder_perp_acq(D, R, acq, m_max=5):
    S = []
    for acqpar in acq:
        G, delta, DELTA = acqpar
        lnS = vangelderen_cylinder_perp_ln(D, R, DELTA, delta, G, m_max)
        S.append(np.exp(lnS))
    return np.array(S)


# acquisitions parameters
# [G, delta, DELTA] in [T/m, s, s]
acq = np.array([[300e-3, 05e-5, 40e-3],
#                 [300e-3, 10e-3, 40e-3],
                [300e-3, 20e-3, 40e-3],
#                 [300e-3, 30e-3, 40e-3],
                [300e-3, 40e-3, 40e-3]])


D = 2.0e-9


min_R = 0.4e-6/2.
max_R = 6.0e-6/2.
step_R = 0.025e-6
# step_R = 0.25e-6
Rs = np.arange(min_R, max_R+step_R, step_R)

# generate Radius dictionary
signals = []
for R in Rs:
    S = vangelderen_cylinder_perp_acq(D, R, acq)
    signals.append(S)


def compute_dico_diff(S, dico, errorfunc=lambda S,gt:np.mean((np.abs(S-gt)/gt))):
    err = []
    for dico_S in dico:
        err.append(errorfunc(S,dico_S))
    return np.array(err)


# Test signal
RR = 0.5*3.001e-6
S = vangelderen_cylinder_perp_acq(D, RR, acq)

err_S = compute_dico_diff(S, signals)



tt = 0.01
ttt = np.min(err_S) + tt

fitted = Rs[err_S <= ttt]



# pl.figure()
# pl.semilogy(Rs*1e6, err_S, c='b')
# # pl.plot(Rs, err_S, c='b')
# pl.axhline(ttt, label='{:.1f}% difference with min'.format(100*tt), c='r')
# pl.title('{:.4f} um :: fitted = [{:.2f}  {:.2f}] um'.format(RR*1e6, fitted.min()*1e6, fitted.max()*1e6))
# pl.legend()
# pl.xlabel('Radius (um)')
# pl.ylabel('RMAE')
# pl.show()



# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# the default error function is too generous because it scale down with multiple times (because of mean and sensitivity skewness)
def compute_2d_slice_dico_diff(S, dico, f1, errorfunc=lambda S,gt:np.mean((np.abs(S-gt)/gt))):
    err1 = []
    for dico_S1 in dico:
        err2 = []
        for dico_S2 in dico:
            dico_S = f1*dico_S1 + (1-f1)*dico_S2
            err2.append(errorfunc(S,dico_S))
        err1.append(err2)
    return np.array(err1)


def plot_err(err, minV, maxV, divV):
    elev_min = minV
    elev_max = maxV
    mid_val = divV
    cmap=pl.cm.RdBu_r # set the colormap to something diverging
    fig = pl.figure()
    pl.imshow(err, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max),extent=[(Rs*1e6).min(),(Rs*1e6).max(),(Rs*1e6).max(),(Rs*1e6).min()])
    pl.colorbar()
#     pl.show()
    return fig

# setting up fractions for 2 cylinders experiment
min_f = 0.
max_f = 1.
step_f = 0.01
# step_f = 0.1
fs = np.arange(min_f, max_f+step_f, step_f)



# Test signal
# shits slow and double computes because symmetry
RR1 = 0.5*3.501e-6
RR2 = 0.5*4.501e-6
ff1 = 0.35

S1 = vangelderen_cylinder_perp_acq(D, RR1, acq)
S2 = vangelderen_cylinder_perp_acq(D, RR2, acq)
S = ff1*S1 + (1-ff1)*S2

err_S = []
for f in fs:
    err_S_f = compute_2d_slice_dico_diff(S, signals, f)
    err_S.append(err_S_f)



err_S_array = np.array(err_S)
tt = 0.01
ttt = np.min(err_S_array) + tt
minV = err_S_array.min()
maxV = err_S_array.max()

# for i,f in enumerate(fs):
# #     fig = plot_err(err_S[i], minV, maxV, tt)
#     fig = plot_err(err_S[i], minV, maxV, ttt)
#     # pl.scatter([RR*1e6], [RR*1e6], c='r')
#     pl.title('{:.1f} % HOR   radius: {:.1f} % {:.2f} um +AND+ {:.1f} % {:.2f} um'.format(f*100, 100*ff1, RR1*1e6, 100*(1-ff1), RR2*1e6))
# pl.show()





from mayavi import mlab 

plot_data = err_S_array.copy()

# for idx in np.ndindex(plot_data.shape):
# 	if idx[1] > idx[2]:
# 		plot_data[idx] = 1

# plot_data[0:plot_data.shape[0]//2] = 1

# src = mlab.pipeline.scalar_field(plot_data, origin=[0, 0.2, 0.2])
src = mlab.pipeline.scalar_field(plot_data, extent=[min_f, max_f, 1e6*min_R, 1e6*max_R, 1e6*min_R, 1e6*max_R], origin=[min_f, 1e6*min_R, 1e6*min_R], spacing=[step_f, 1e6*step_R, 1e6*step_R])
mlab.axes(src, extent=[min_f, max_f, 1e6*min_R, 1e6*max_R, 1e6*min_R, 1e6*max_R], xlabel='f_in', ylabel='R_1', zlabel='R_2')
# mlab.pipeline.iso_surface(src, contours=[s.min()+0.1*s.ptp(), ], opacity=0.3) 
# mlab.pipeline.iso_surface(src, contours=[0.001, ],) 
mlab.pipeline.iso_surface(src, contours=[tt, ], opacity=0.4) 
 
mlab.show()                                                                                                                                                                                                                                                            

















# X,Y,Z = np.indices((err_S_array.shape[0]+1, err_S_array.shape[1]+1, err_S_array.shape[2]+1))


# # everything red
# colors = np.zeros(err_S_array.shape + (4,))
# colors[..., 0] = 1
# colors[..., 1] = 0
# colors[..., 2] = 0

# # need to set transparency based on error
# # err <= 1% -> alpha = 1
# lowT = 0.0099
# # err >= 5% -> alpha = 0
# highT = 0.011

# colors[..., 3] = (err_S_array - lowT) / (highT - lowT)
# colors[err_S_array<lowT, 3] = 1
# colors[err_S_array>highT, 3] = 0










# # This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# # def midpoints(x):
# #     sl = ()
# #     for i in range(x.ndim):
# #         x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
# #         sl += np.index_exp[:]
# #     return x

# # # prepare some coordinates, and attach rgb values to each
# # r, g, b = np.indices((17, 17, 17)) / 16.0
# # rc = midpoints(r)
# # gc = midpoints(g)
# # bc = midpoints(b)

# # # define a sphere about [0.5, 0.5, 0.5]
# # sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

# # # combine the color components
# # colors = np.zeros(sphere.shape + (3,))
# # colors[..., 0] = rc
# # colors[..., 1] = gc
# # colors[..., 2] = bc

# # and plot everything
# fig = pl.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(X,Y,Z, colors[...,3] > 0,
#           facecolors=colors,
#           # edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
#           linewidth=0.5)
# # ax.set(xlabel='r', ylabel='g', zlabel='b')

# pl.show()


dico_params = []
dico_values = []

for iter_f, f in enumerate(fs):
	for iter_r1, r1 in enumerate(Rs):
		for iter_r2, r2 in enumerate(Rs):
			dico_params.append((f, r1, r2))
			dico_values.append(f*signals[iter_r1] + (1-f)*signals[iter_r2])

dico_values = np.array(dico_values)







trial = 1000


RR1 = 0.5*3.501e-6
RR2 = 0.5*4.501e-6
ff1 = 0.35

SNR = 30.

fs = []
R1 = []
R2 = []



S1 = vangelderen_cylinder_perp_acq(D, RR1, acq)
S2 = vangelderen_cylinder_perp_acq(D, RR2, acq)
S = ff1*S1 + (1-ff1)*S2



for i in range(trial):
	Sn = S + (1/SNR) * np.random.randn()

	tmp = np.argmin(np.abs(dico_values-Sn).sum(axis=1))
	a,b,c = dico_params[tmp]

	if b < c:
		fs.append(a)
		R1.append(b)
		R2.append(c)
	else:
		fs.append(1-a)
		R1.append(c)
		R2.append(b)




fs = np.array(fs)
R1 = 1e6*np.array(R1)
R2 = 1e6*np.array(R2)




pl.figure()
pl.hist(fs, 100)
pl.title('Fitted Signal Fraction for small axon')
pl.axvline(ff1, color='red', label='{:.2f}'.format(ff1))
pl.legend()

pl.figure()
pl.hist(R1, 100)
pl.title('Fitted diameter for small axon')
pl.axvline(1e6*RR1, color='red', label='{:.2f} um'.format(1e6*RR1))
pl.legend()

pl.figure()
pl.hist(R2, 100)
pl.title('Fitted diameter for big axon')
pl.axvline(1e6*RR2, color='red', label='{:.2f} um'.format(1e6*RR2))
pl.legend()

pl.show()



