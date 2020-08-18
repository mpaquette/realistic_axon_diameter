import numpy as np
from scipy.special import jnp_zeros
# import pylab as pl
import matplotlib.pyplot as pl
import matplotlib.colors as colors
from matplotlib.patches import Circle

#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'

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










# Test signal

# BIG / IN-vivo
D = 2.0e-9
RR1 = 0.5*4.5e-6
RR2 = 0.5*3.5e-6
ff1 = 0.3

# # MEDIUM / IN-vivo
# D = 2.0e-9
# RR1 = 0.5*3.5e-6
# RR2 = 0.5*2.5e-6
# ff1 = 0.3

# # SMALL / IN-vivo
# D = 2.0e-9
# RR1 = 0.5*2.5e-6
# RR2 = 0.5*1.5e-6
# ff1 = 0.3

# # MEDIUM / EX-vivo
# D = 0.66e-9
# RR1 = 0.5*2.5e-6
# RR2 = 0.5*1.5e-6
# ff1 = 0.3











# min_R = 0.05e-6/2.
min_R = 0.025e-6/2.
max_R = 6.0e-6/2.
# Rs = np.linspace(min_R, max_R, 120, endpoint=True)
# Rs = np.linspace(min_R, max_R, 239, endpoint=True)
# Rs = np.linspace(min_R, max_R, 596, endpoint=True)
Rs = np.linspace(min_R, max_R, 1196, endpoint=True)

# generate Radius dictionary
signals = []
for R in Rs:
    S = vangelderen_cylinder_perp_acq(D, R, acq)
    signals.append(S)


# def compute_dico_diff(S, dico, errorfunc=lambda S,gt:np.mean((np.abs(S-gt)/gt))):
#     err = []
#     for dico_S in dico:
#         err.append(errorfunc(S,dico_S))
#     return np.array(err)


# the default error function is too generous because it scale down with multiple times (because of mean and sensitivity skewness)
# def compute_2d_slice_dico_diff(S, dico, f1, errorfunc=lambda S,gt: np.sum((S-gt)**2)/np.sum(gt**2)):
def compute_2d_slice_dico_diff(S, dico, f1, errorfunc=lambda S,gt: np.sum(np.abs(S-gt))/3):
# def compute_2d_slice_dico_diff(S, dico, f1, errorfunc=lambda S,gt: np.min(np.abs(S-gt))):
# def compute_2d_slice_dico_diff(S, dico, f1, errorfunc=lambda S,gt: np.max(np.abs(S-gt))):
    err1 = []
    for dico_S1 in dico:
        err2 = []
        for dico_S2 in dico:
            dico_S = f1*dico_S1 + (1-f1)*dico_S2
            err2.append(errorfunc(S,dico_S))
        err1.append(err2)
    return np.array(err1)




# setting up fractions for 2 cylinders experiment
min_f = 0.1
max_f = 0.5
x1 = 3
x2 = 3
fs = np.linspace(min_f, max_f, x1*x2, endpoint=True)




S1 = vangelderen_cylinder_perp_acq(D, RR1, acq)
S2 = vangelderen_cylinder_perp_acq(D, RR2, acq)
S = ff1*S1 + (1-ff1)*S2

err_S = []
for f in fs:
    err_S_f = compute_2d_slice_dico_diff(S, signals, f)
    err_S.append(err_S_f)



err_S_array = np.array(err_S)
tt = 0.03
ttt = np.min(err_S_array) + tt
minV = err_S_array.min()
maxV = err_S_array.max()




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



# for i,f in enumerate(fs):
# #     fig = plot_err(err_S[i], minV, maxV, tt)
#     fig = plot_err(err_S[i], minV, maxV, ttt)
#     # pl.scatter([RR*1e6], [RR*1e6], c='r')
#     pl.title('{:.1f} % HOR   radius: {:.1f} % {:.2f} um +AND+ {:.1f} % {:.2f} um'.format(f*100, 100*ff1, RR1*1e6, 100*(1-ff1), RR2*1e6))
# pl.show()






def mn(f1, R1, f2, R2, n):
    return f1*R1**n+f2*R2**n

def reff_count(f1, R1, f2, R2):
    return (mn(f1, R1, f2, R2, 6) / mn(f1, R1, f2, R2, 2))**0.25

def reff_f(sf1, R1, sf2, R2):
    # # this work but its pointless
    # R2_norm = R1**2 + R2**2
    # f1=sf1/(R1**2/R2_norm)
    # f2=sf2/(R2**2/R2_norm)
    # return (mn(f1, R1, f2, R2, 6) / mn(f1, R1, f2, R2, 2))**0.25
    return mn(sf1, R1, sf2, R2, 4)**0.25

# Deff = 2*reff_count(ff1, RR1, 1-ff1, RR2)
Deff = 2*reff_f(ff1, RR1, 1-ff1, RR2)








# pl.figure()
# pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.5)
# for i,f in enumerate(fs):
#     elev_min = minV
#     elev_max = maxV
#     mid_val = ttt
#     cmap=pl.cm.RdBu_r # set the colormap to something diverging
#     pl.subplot(x1,x2,i+1)
#     pl.imshow(err_S[i], cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max),extent=[(Rs*2e6).min(),(Rs*2e6).max(),(Rs*2e6).max(),(Rs*2e6).min()])
#     cc_deff = Circle((Deff*1e6, Deff*1e6), radius=0.04, color='lightgreen')
#     pl.gca().add_patch(cc_deff)
#     if np.abs(f-ff1)<0.01:
#         cc_gt = Circle((RR2*2e6, RR1*2e6), radius=0.04, color='red')
#         pl.gca().add_patch(cc_gt)
#     # pl.colorbar()
#     pl.xlabel('D2 (um)', fontsize=14)
#     pl.ylabel('D1 (um)', fontsize=14)
#     pl.xticks(fontsize=12)
#     pl.yticks(fontsize=12)

#     pl.title('{:.1f}\% D1 + {:.1f}\% D2'.format(f*100, (1-f)*100), fontsize=14)

# pl.tight_layout()
# pl.show()


# import matplotlib.ticker as ticker
# def fmt(x, pos):
#     a, b = '{:.0e}'.format(x).split('e')
#     b = int(b)+2
#     return r'${} \times 10^{{{}}}$ \%'.format(a, b)




# levels = [0, 0.001, 0.005, 0.01, 0.05]
# # mycolormap = pl.cm.hsv
# mycolormap = pl.cm.jet
# # mycolormap = pl.cm.plasma
# # mycolormap = pl.cm.viridis
# _colors = [mycolormap(i) for i in np.linspace(0, 1, len(levels))]

# pl.figure(figsize=(14,12))

# pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.5)
# for i,f in enumerate(fs):
#     pl.subplot(x1,x2,i+1)
#     cp = pl.contourf(Rs*2e6, Rs*2e6, err_S[i], levels, colors=_colors)
#     cbar = pl.colorbar(cp, format=ticker.FuncFormatter(fmt))
#     # cbar = pl.colorbar(cp, format=ticker.FuncFormatter(fmt))
#     # cbar.ax.tick_params(labelsize=textfs)

#     cc_deff = Circle((Deff*1e6, Deff*1e6), radius=0.10, color='black')
#     pl.gca().add_patch(cc_deff)

#     if np.abs(f-ff1)<0.01:
#         cc_gt = Circle((RR2*2e6, RR1*2e6), radius=0.12, color='red')
#         pl.gca().add_patch(cc_gt)

#     pl.gca().set_aspect(aspect=1)

#     pl.xlabel(r'$d_2$ ($\mu$m)', fontsize=14)
#     pl.ylabel(r'$d_1$ ($\mu$m)', fontsize=14)
#     pl.xticks(fontsize=12)
#     pl.yticks(fontsize=12)

#     pl.title(r'${:.0f}\% \,d_1 + {:.0f}\% \,d_2$'.format(f*100, (1-f)*100), fontsize=14)

# # pl.tight_layout()
# pl.show()




import matplotlib.ticker as ticker
def fmt(x, pos):
    a, b = '{:.1f}'.format(100*x).split('.')
    return r'{:}.{:} \%'.format(a, b)

# TODO handle case with not digit
# # multiply by 100 and print all digit before decimal and up to the first 2 decimal
# def fmt(x, pos):
#     a, b = '{:.15f}'.format(100*x).split('.')
#     # pre decimal
#     a = int(a)
#     # search for first non zero decimal
#     notZero = np.array([digt!='0' for digt in b])
#     posFirstDigit = np.where(notZero)[0][0]
#     # check for rounding with 3rd digit
#     if int(b[posFirstDigit+2]) < 5:
#         c = '0'*posFirstDigit + b[posFirstDigit] + b[posFirstDigit+1]
#     else:
#         # we SHOULD check if b[posFirstDigit+1]+1 is 10, and if it is we increase b[posFirstDigit] by one and if it is also 10 now ....
#         # but I wont
#         c = '0'*posFirstDigit + b[posFirstDigit] + str(int(b[posFirstDigit+1])+1)

#     return r'{:}.{:} \%'.format(a, c)



levels = [0, 0.001, 0.005, 0.01, 0.05]
# mycolormap = pl.cm.hsv
mycolormap = pl.cm.jet
# mycolormap = pl.cm.plasma
# mycolormap = pl.cm.viridis
# _colors = [mycolormap(i) for i in np.linspace(0, 1, len(levels))]

# np.random.seed(2)
# _colors = [(np.random.rand(), np.random.rand(), np.random.rand(), 1.0) for i in range(5)] 

# nicely distinguisable from set1 categorical
_colors = ['#984ea3', '#4daf4a', '#377eb8', '#ff7f00', '#e41a1c']

fig, axes = pl.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(14,12))
fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.5)


for ix,iy in np.ndindex((x1,x2)):
    i = ix*x2 + iy
    f = fs[i]
    axs = axes[ix, iy]
    cp = axs.contourf(Rs*2e6, Rs*2e6, err_S[i], levels, colors=_colors)
    # cbar = pl.colorbar(cp, format=ticker.FuncFormatter(fmt))
    # cbar = pl.colorbar(cp, format=ticker.FuncFormatter(fmt))
    # cbar.ax.tick_params(labelsize=textfs)

    # cc_deff = Circle((Deff*1e6, Deff*1e6), radius=0.10, color='black')
    # pl.gca().add_patch(cc_deff)

    if np.abs(f-ff1)<0.01:
        cc_gt = Circle((RR2*2e6, RR1*2e6), radius=0.12, color='red')
        axs.add_patch(cc_gt)

    # pl.gca().set_aspect(aspect=1)
    axs.set_aspect(aspect=1)

    axs.set_xlabel(r'$d_2$ ($\mu$m)', fontsize=16)
    axs.set_ylabel(r'$d_1$ ($\mu$m)', fontsize=16)
    axs.set_xticks(range(1,7))
    axs.set_yticks(range(1,7))
    axs.set_xticklabels(range(1,7), fontsize=14)
    axs.set_yticklabels(range(1,7), fontsize=14)

    axs.set_title(r'{:.0f}\% $d_1$ + {:.0f}\% $d_2$'.format(f*100, (1-f)*100), fontsize=16)

cbar = fig.colorbar(cp, ax=axes.ravel().tolist(), format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=18)



# pl.tight_layout()
pl.show()

