import numpy as np
import pylab as pl
from scipy.stats import gamma 

from matplotlib import rc
rc('text', usetex=True)





def D_perp_intra_world(DELTA, delta, f_ex, D_inf_ex, r_app, D_0):
    # DELTA: big delta (ms)
    # delta: small delta (ms)
    # f_ex: extra axonal space volume fraction
    # f_in: intra axonal space volume fraction
    f_in = 1-f_ex
    # D_inf_ex: long-time limit extra axonal space diffusivity (um^2/ms)
    # r_app: apparent axon radius (um)
    # for distributions: r_app = <r^6>/<r^2>
    # D_0: intra axonal unrestricted diffusivity (um^2/ms)
    c = (7/48.)*(f_in*r_app**4)/D_0
    tau = DELTA - delta/3.
    return f_ex*D_inf_ex + (c/(delta*tau))


def D_perp_extra_world(DELTA, delta, f_ex, D_inf_ex, A):
    # DELTA: big delta (ms)
    # delta: small delta (ms)
    # f_ex: extra axonal space volume fraction
    # D_inf_ex: long-time limit extra axonal space diffusivity (um^2/ms)
    # A: (for now mysterious) disorder parameter
    c_prime = f_ex*A
    tau = DELTA - delta/3.
    return f_ex*D_inf_ex + c_prime*((np.log(DELTA/delta)+1.5)/tau)



def r_app_from_extra(DELTA, delta, f_ex, D_inf_ex, D_0, A):
    f_in = 1-f_ex
    c_prime = f_ex*A
    c = c_prime*(np.log(DELTA/delta)+1.5)*delta
    r_app = ((48/7.)*(D_0/f_in)*c)**0.25
    return r_app


D_0 = 2.
D_inf_ex = 0.5


N_DELTAs = 96
DELTAs = np.linspace(5,100,N_DELTAs)
N_deltas = 46
deltas = np.linspace(5,50,N_deltas)


# # D_perp_intra_world
# r_apps = np.array([1., 2., 3., 4.])
# N_r_apps = r_apps.shape[0]
# f_exs = np.array([0.25, 0.5, 0.75])
# N_f_exs = f_exs.shape[0]


# intra_D = np.zeros((N_r_apps, N_f_exs, N_DELTAs, N_deltas))
# intra_D_vmin = np.inf*np.ones((N_r_apps, N_f_exs))


# for i_r_apps in range(N_r_apps):
#   r_app = r_apps[i_r_apps]
#   for i_f_exs in range(N_f_exs):
#       f_ex = f_exs[i_f_exs]
#       for i_DELTAs in range(N_DELTAs):
#           DELTA = DELTAs[i_DELTAs]
#           for i_deltas in range(N_deltas):
#               delta = deltas[i_deltas]
#               if DELTA >= delta:
#                   tmp = D_perp_intra_world(DELTA, delta, f_ex, D_inf_ex, r_app, D_0)
#                   intra_D[i_r_apps, i_f_exs, i_DELTAs, i_deltas] = tmp
#                   if tmp < intra_D_vmin[i_r_apps, i_f_exs]:
#                       intra_D_vmin[i_r_apps, i_f_exs] = tmp


## D_perp from intra world
# pl.figure()
# for i_f_exs in range(N_f_exs):
#   f_ex = f_exs[i_f_exs]
#   for i_r_apps in range(N_r_apps):
#       r_app = r_apps[i_r_apps]
#       pl.subplot(N_f_exs, N_r_apps, i_f_exs*N_r_apps + i_r_apps + 1)
#       pl.imshow(intra_D[i_r_apps, i_f_exs].T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=intra_D_vmin[i_r_apps, i_f_exs])
#       pl.title(r'$f_{{ex}} = {} \,\,\,\, r_{{app}} = {}$'.format(f_ex, r_app))
#       pl.colorbar()

# pl.show()



unphysical_mask = np.zeros((N_DELTAs, N_deltas), dtype=np.bool)


# D_perp_extra_world
As = np.array([0.25, 0.5, 1., 2.])
N_As = As.shape[0]
f_exs = np.array([0.25, 0.5, 0.75])
N_f_exs = f_exs.shape[0]


extra_D = np.zeros((N_As, N_f_exs, N_DELTAs, N_deltas))
fake_r_app_extra = np.zeros((N_As, N_f_exs, N_DELTAs, N_deltas))

extra_D_vmin = np.inf*np.ones((N_As, N_f_exs))
fake_r_app_extra_vmin = np.inf*np.ones((N_As, N_f_exs))

fake_r_app_extra_vmax = -np.inf*np.ones((N_As, N_f_exs))


for i_As in range(N_As):
    A = As[i_As]
    for i_f_exs in range(N_f_exs):
        f_ex = f_exs[i_f_exs]
        for i_DELTAs in range(N_DELTAs):
            DELTA = DELTAs[i_DELTAs]
            for i_deltas in range(N_deltas):
                delta = deltas[i_deltas]
                if DELTA >= delta:
                    tmp = D_perp_extra_world(DELTA, delta, f_ex, D_inf_ex, A)
                    extra_D[i_As, i_f_exs, i_DELTAs, i_deltas] = tmp
                    if tmp < extra_D_vmin[i_As, i_f_exs]:
                        extra_D_vmin[i_As, i_f_exs] = tmp

                    tmp = r_app_from_extra(DELTA, delta, f_ex, D_inf_ex, D_0, A)
                    fake_r_app_extra[i_As, i_f_exs, i_DELTAs, i_deltas] = tmp
                    if tmp < fake_r_app_extra_vmin[i_As, i_f_exs]:
                        fake_r_app_extra_vmin[i_As, i_f_exs] = tmp
                    if tmp > fake_r_app_extra_vmax[i_As, i_f_exs]:
                        fake_r_app_extra_vmax[i_As, i_f_exs] = tmp

                else:
                    unphysical_mask[i_DELTAs, i_deltas] = True


## D_perp from extra world
# pl.figure()
# for i_f_exs in range(N_f_exs):
#   f_ex = f_exs[i_f_exs]
#   for i_As in range(N_As):
#       A = As[i_As]
#       pl.subplot(N_f_exs, N_As, i_f_exs*N_As + i_As + 1)
#       pl.imshow(extra_D[i_As, i_f_exs].T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=extra_D_vmin[i_As, i_f_exs])
#       pl.title(r'$D: f_{{ex}} = {} \,\,\,\, A = {}$'.format(f_ex, A))
#       pl.colorbar()

# pl.show()


## apparant radius from fitting dperp_intra_world formula to derp_extra_world signal
## basic figure
# pl.figure()
# for i_f_exs in range(N_f_exs):
#   f_ex = f_exs[i_f_exs]
#   for i_As in range(N_As):
#       A = As[i_As]
#       pl.subplot(N_f_exs, N_As, i_f_exs*N_As + i_As + 1)
#       pl.imshow(fake_r_app_extra[i_As, i_f_exs].T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=fake_r_app_extra_vmin[i_As, i_f_exs])
#       pl.title(r'$r_{{fake}}: f_{{ex}} = {} \,\,\,\, A = {}$'.format(f_ex, A))
#       pl.colorbar()

# pl.show()













## apparant radius from fitting dperp_intra_world formula to derp_extra_world signal
## unified x and y axis label and commun colorbar
# import matplotlib.ticker as ticker
# def fmt(x, pos):
#     return r'{:.0f} $\mu$m'.format(x)




# fig, axes = pl.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(8, 6))
# for i_f_exs in range(N_f_exs):
#   f_ex = f_exs[i_f_exs]
#   for i_As in range(N_As):
#       A = As[i_As]
#       # pl.subplot(N_f_exs, N_As, i_f_exs*N_As + i_As + 1)
#       axs = axes[i_f_exs, i_As]
#       im = axs.imshow(2*fake_r_app_extra[i_As, i_f_exs].T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=fake_r_app_extra_vmin.min(), vmax=fake_r_app_extra_vmax.max())
#       axs.tick_params(axis='both', which='major', labelsize=14)
#       axs.tick_params(axis='both', which='minor', labelsize=14)
#       axs.title.set_text(r'$f_{{ex}} = {} \,\,\,\, A = {}$'.format(f_ex, A))
#       axs.title.set_fontsize(16)
#       # cb.ax.tick_params(labelsize=14)

# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format=ticker.FuncFormatter(fmt))
# cbar.ax.tick_params(labelsize=14)
# fig.text(0.5, 0.08, r'$\Delta$ (ms)', ha='center', fontsize=16)
# fig.text(0.08, 0.5, r'$\delta$ (ms)', va='center', fontsize=16, rotation='vertical')

# pl.show()










## apparant radius from fitting dperp_intra_world formula to derp_extra_world signal
## basic structure but same color scalling
# pl.figure()
# for i_f_exs in range(N_f_exs):
#   f_ex = f_exs[i_f_exs]
#   for i_As in range(N_As):
#       A = As[i_As]
#       pl.subplot(N_f_exs, N_As, i_f_exs*N_As + i_As + 1)
#       pl.imshow(2*fake_r_app_extra[i_As, i_f_exs].T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=fake_r_app_extra_vmin.min(), vmax=fake_r_app_extra_vmax.max())
#       pl.gca().tick_params(axis='both', which='major', labelsize=12)
#       pl.gca().tick_params(axis='both', which='minor', labelsize=12)
#       # pl.title(r'$r_{{fake}}: f_{{ex}} = {} \,\,\,\, A = {}$'.format(f_ex, A), fontsize=16)
#       pl.title(r'$f_{{ex}} = {} \,\,\,\, A = {}$'.format(f_ex, A), fontsize=16)
#       # pl.colorbar()
#       cb = pl.colorbar()
#       cb.ax.tick_params(labelsize=14)
# # pl.title('Apparent Radius from Extra-Axonal tTme Dependent Diffusion')
# pl.show()





## apparant radius from fitting dperp_intra_world formula to derp_extra_world signal
## final version with transparency for unphysical (delta,DELTA) combo and better titles
import matplotlib.ticker as ticker
def fmt(x, pos):
    return r'{:.0f} $\mu$m'.format(x)


fig, axes = pl.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(16, 7))
for i_f_exs in range(N_f_exs):
    f_ex = f_exs[i_f_exs]
    for i_As in range(N_As):
        A = As[i_As]
        # pl.subplot(N_f_exs, N_As, i_f_exs*N_As + i_As + 1)
        axs = axes[i_f_exs, i_As]

        plot_data = np.ma.array(2*fake_r_app_extra[i_As, i_f_exs], mask=unphysical_mask)

        im = axs.imshow(plot_data.T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=fake_r_app_extra_vmin.min(), vmax=fake_r_app_extra_vmax.max())
        # im = axs.imshow(2*fake_r_app_extra[i_As, i_f_exs].T, interpolation='none', extent=[DELTAs.min(),DELTAs.max(),deltas.max(),deltas.min()], vmin=fake_r_app_extra_vmin.min(), vmax=fake_r_app_extra_vmax.max())
        # axs.tick_params(axis='both', which='major', labelsize=14)
        # axs.tick_params(axis='both', which='minor', labelsize=14)
        # axs.title.set_text(r'$f_{{ex}} = {} \,\,\,\, A = {}$'.format(f_ex, A))
        # axs.title.set_fontsize(16)
        # cb.ax.tick_params(labelsize=14)

        axs.set_xticks([5, 25, 50, 75, 100])
        axs.set_yticks([5, 25, 50])
        axs.set_xticklabels([5, 25, 50, 75, 100], fontsize=14)
        axs.set_yticklabels([5, 25, 50], fontsize=14)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=16)
# fig.text(0.5, 0.08, r'$\Delta$ (ms)', ha='center', fontsize=18)
# fig.text(0.08, 0.5, r'$\delta$ (ms)', va='center', fontsize=18, rotation='vertical')

for i_f_exs in range(N_f_exs):
    f_ex = f_exs[i_f_exs]
    fig.text(0.06, (i_f_exs+1)/float(N_f_exs+1), r'$f_{{ex}} = {}$'.format(f_ex), va='center', fontsize=18, rotation='vertical')

for i_As in range(N_As):
    A = As[i_As]
    fig.text((i_As+1.1)/float(N_As+2), 0.88, r'$A = {}$'.format(A), ha='center', fontsize=18)

axes[N_f_exs-1, 0].set_xlabel(r'$\Delta$ (ms)', fontsize=16)
axes[N_f_exs-1, 0].set_ylabel(r'$\delta$ (ms)', fontsize=16)

pl.show()






