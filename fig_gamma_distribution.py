import numpy as np
from scipy.special import jnp_zeros
# import pylab as pl
import matplotlib.pyplot as pl
import matplotlib.colors as colors
from matplotlib.patches import Circle
import scipy.stats as ss
from scipy.interpolate import interp1d


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
acq = np.array([[300e-3, 30e-3, 50e-3],
                [300e-3, 40e-3, 50e-3],
                [300e-3, 50e-3, 50e-3]])


D0 = 2.0e-9



# arbitrary but if qmax is too big, there will be non sense
# in practice, I will tweak the parameter range so that qmax give me something sensible
qmin = 0.01 # min quantile
qmax = 0.99 # max quantile



def reff_gamma(k, theta):
    return theta*((k+5)*(k+4)*(k+3)*(k+2))**0.25


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


# def get_gamma_binned_pdf(k, theta, minq=0.01, maxq=0.99, N=100):
#     rv = ss.gamma(k, loc=0, scale=theta) # define random variable for the diameters
#     dmin = rv.ppf(minq) # find d corresponding to min quartile
#     dmax = rv.ppf(maxq) # find d corresponding to max quartile
#     ds_center = np.linspace(dmin, dmax, N, endpoint=True) # center of "bins"
#     spacing = ds_center[1] - ds_center[0] # get "bins" width
#     ds_left = np.linspace(dmin, dmax, N, endpoint=True) - (spacing/2) # left limit of "bins"
#     ds_right = np.linspace(dmin, dmax, N, endpoint=True) + (spacing/2) # right limit of "bins"
#     # Tai's method XD
#     areas = (rv.pdf(ds_left) + rv.pdf(ds_right))*spacing/2.
#     areas /= areas.sum() # normalize to sum 1 for our a purpose, so not technically a density anymore
#     return ds_center, areas

def get_gamma_binned_pdf(k, theta, maxq=0.99, N=100):
    rv = ss.gamma(k, loc=0, scale=theta) # define random variable for the diameters
    dmin = 0.1
    dmax = rv.ppf(maxq) # find d corresponding to max quartile

    ds_left = np.linspace(dmin, dmax, N, endpoint=True)
    spacing = ds_left[1] - ds_left[0] # get "bins" width
    ds_right = ds_left + spacing 
    ds_center = ds_left + (spacing/2)
    # Tai's method XD
    areas = (rv.pdf(ds_left) + rv.pdf(ds_right))*spacing/2.
    areas /= areas.sum() # normalize to sum 1 for our a purpose, so not technically a density anymore
    return ds_center, areas




def get_signal_fraction(radii, prob):
    # "volume" weigths
    crosssections = radii**2
    # signal fractions
    return (prob*crosssections)/crosssections



# GROUND TRUTH

# param
k = 3.0
theta = 0.5

# count probabilities
# ds_center, areas = get_gamma_binned_pdf(k, theta, minq=qmin, maxq=qmax, N=100)
ds_center, areas = get_gamma_binned_pdf(k, theta, maxq=qmax, N=100)
# signal fractions
signal_f = get_signal_fraction(ds_center/2., areas)
# generate signal for each diameter
signal = np.array([vangelderen_cylinder_perp_acq(D0, d*0.5e-6, acq, m_max=5) for d in ds_center]) # convert um diameter into m radius
# sum signals with the weigths
signal_full = (signal_f[:, None] * signal).sum(axis=0)

gt_deff = 2*reff_gamma(k, theta)
gt_deff_true = 2*r_eff(areas, ds_center)

pl.figure()
pl.plot(ds_center, areas)
pl.title('GT, signal decay = [{:.1e}, {:.1e}, {:.1e}]'.format(*(1-signal_full)))
pl.show()




# all params for dictionary generation
krange = np.linspace(0.1, 5, 64, endpoint=True)
thetarange = np.linspace(0.1, 2, 64, endpoint=True)

# signal storage
data = np.zeros((krange.shape[0], thetarange.shape[0], acq.shape[0]))
dmin_data = np.zeros((krange.shape[0], thetarange.shape[0]))
dmax_data = np.zeros((krange.shape[0], thetarange.shape[0]))
k_data = np.zeros((krange.shape[0], thetarange.shape[0]))
t_data = np.zeros((krange.shape[0], thetarange.shape[0]))
data_deff_true = np.zeros((krange.shape[0], thetarange.shape[0]))
p_data = np.zeros((krange.shape[0], thetarange.shape[0]))

from time import time
startt = time()
for k_i, k_1 in enumerate(krange):
    print('{} / {}'.format(k_i, krange.shape[0]))
    for t_i, theta_1 in enumerate(thetarange):
        peak_1 = max((k_1-1)*theta_1,0)
        # count probabilities
        # ds_center_1, areas_1 = get_gamma_binned_pdf(k_1, theta_1, minq=qmin, maxq=qmax, N=100)
        ds_center_1, areas_1 = get_gamma_binned_pdf(k_1, theta_1, maxq=qmax, N=100)
        data_deff_true[k_i, t_i] = 2*r_eff(areas_1, ds_center_1)
        # signal fractions
        signal_f_1 = get_signal_fraction(ds_center_1/2., areas_1)
        # generate signal for each diameter
        signal_1 = np.array([vangelderen_cylinder_perp_acq(D0, d*0.5e-6, acq, m_max=5) for d in ds_center_1]) # convert um diameter into m radius
        # sum signals with the weigths
        signal_full_1 = (signal_f_1[:, None] * signal_1).sum(axis=0)
        # logging
        data[k_i, t_i] = signal_full_1
        dmin_data[k_i, t_i] = ds_center_1.min()
        dmax_data[k_i, t_i] = ds_center_1.max()
        k_data[k_i, t_i] = k_1
        t_data[k_i, t_i] = theta_1
        p_data[k_i, t_i] = peak_1

endt = time()
print('time = {:.0f} seconds for {} gammas'.format(endt-startt, krange.shape[0]*thetarange.shape[0]))

errorfunc=lambda S,gt: np.sum((S-gt)**2)/np.sum(gt**2)


# compute NMSE error
errors = np.sum((data - signal_full)**2, axis=2) / np.sum((signal_full)**2)

data_deff = 2*reff_gamma(k_data, t_data)


f_k = interp1d(krange, range(len(krange)))
ycc = f_k(k)

f_t = interp1d(thetarange, range(len(thetarange)))
xcc = f_t(theta)


pl.figure()
pl.imshow(errors)
pl.title('errors vs (k={:.2f} t={:.2f}) for k in [{:.2f} {:.2f}] and t in [{:.2f} {:.2f}]'.format(k, theta, krange.min(), krange.max(), peakrange.min(), peakrange.max()))
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)
# pl.show()


pl.figure()
pl.imshow(dmin_data)
pl.title('dmin')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(dmax_data)
pl.title('dmax')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(k_data)
pl.title('k')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(t_data)
pl.title('theta')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(p_data)
pl.title('peak')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(data_deff)
pl.title('deff')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(data_deff_true)
pl.title('deff_true')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(np.abs(data_deff - gt_deff), vmax=2)
pl.title('abs  deff - gt deff')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(np.abs(data_deff_true - gt_deff_true), vmax=2)
pl.title('abs  deff - gt deff_true')
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)


pl.figure()
pl.imshow(np.log10(errors))
pl.title('log10 errors vs (k={:.2f} t={:.2f}) for k in [{:.2f} {:.2f}] and t in [{:.2f} {:.2f}]'.format(k, theta, krange.min(), krange.max(), peakrange.min(), peakrange.max()))
pl.colorbar()
cc = Circle((xcc, ycc), radius=0.5, color='red')
pl.gca().add_patch(cc)

pl.show()










#         rv = ss.gamma(k_gamma, loc=0, scale=theta_gamma) # define random variable for the diameters
#         dmin = rv.ppf(qmin) # find r corresponding to qmin
#         dmax = rv.ppf(qmax) # find r corresponding to qmax
#         ds = np.linspace(dmin, dmax, 100, endpoint=True)
#         pdf = rv.pdf(ds) # density at those d

#         ds_center, areas = get_gamma_binned_pdf(k_gamma, theta_gamma, minq=0.01, maxq=0.99, N=100)
#         pl.figure()
#         jj = pdf.max()/areas.max()
#         pl.plot(ds, pdf, label='pdf', linewidth=1)
#         pl.plot(ds_center, areas*jj, label='binned pdf')
#         pl.legend()
#         pl.title('k = {:.2f}  theta = {:.2f}  peak = {:.2f}'.format(k_gamma, theta_gamma, (k_gamma-1)*theta_gamma))
#         pl.show()





# krange_gamma = [1.05, 1.1, 1.2, 1.25, 1.5, 2]
# thetarange_gamma = (np.array(krange_gamma)-1)**-1



# for (k_gamma, theta_gamma) in zip(krange_gamma, thetarange_gamma):
#     rv = ss.gamma(k_gamma, loc=0, scale=theta_gamma) # define random variable for the diameters
#     dmin = rv.ppf(qmin) # find r corresponding to qmin
#     dmax = rv.ppf(qmax) # find r corresponding to qmax
#     ds = np.linspace(dmin, dmax, 1000, endpoint=True)
#     pdf = rv.pdf(ds) # density at those d

#     pl.figure()
#     pl.plot(ds, pdf)
#     pl.title('k = {:.2f}  theta = {:.2f}  peak = {:.2f}'.format(k_gamma, theta_gamma, (k_gamma-1)*theta_gamma))
#     pl.show()






