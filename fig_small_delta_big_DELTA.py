# Van Gelderen formula for diffusion in cylinders
import numpy as np
from scipy.special import jnp_zeros

import pylab as pl
import matplotlib.ticker as ticker

from vangelderen import vangelderen_cylinder_perp
from scheme import expand_scheme, remove_unphysical

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}\usepackage{{siunitx}}'




D_invivo = 2e-9

list_G = [0.3]
list_DELTA = np.linspace(10e-3, 50e-3, 9)
spacing_DELTA = list_DELTA[1] - list_DELTA[0]
list_delta = np.linspace(10e-3, 50e-3, 9)
spacing_delta = list_delta[1] - list_delta[0]
scheme = expand_scheme(list_G, list_DELTA, list_delta)
# scheme = remove_unphysical(scheme)

# Rs = np.linspace(0.5e-6, 3.0e-6, 6)
Rs = np.array([0.25, 0.5, 1, 2])*1e-6
n = np.sqrt(len(Rs))
ny = int(np.ceil(n))
nx = int(np.floor(n))
if nx*ny < len(Rs):
    nx += 1




# pl.figure()
# pl.title(r'\num{{1.1e-4}}')
# pl.plot([1,2,3], [3,2,1])
# pl.show()


# pl.figure()
# pl.title('\num{{1.1e-4}}')
# pl.plot([1,2,3], [3,2,1])
# pl.show()

# pl.figure()
# pl.title('$\num{{1.1e-4}}$')
# pl.plot([1,2,3], [3,2,1])
# pl.show()

# pl.figure()
# pl.title(r'$\num{{1.1e-4}}$')
# pl.plot([1,2,3], [3,2,1])
# pl.show()






# # version 2: diameter, 2x2, signal decay in sci notation with only min and max, fixed spacing
# textfs = 20
# pl.figure()
# pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
# pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters', fontsize=textfs+2)
# # pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)
# for iR,R in enumerate(Rs):
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

#   pl.subplot(nx, ny, iR+1)

#   tmp = S.reshape(len(list_DELTA), len(list_delta))
#   mask = np.isinf(tmp)
#   tmp2 = np.ma.array(1-tmp, mask=mask)

#   pl.imshow(tmp2, interpolation='nearest', extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
#   cbar = pl.colorbar()

#   cbar.set_ticks([tmp2.min(), tmp2.max()])
#   cbar.set_ticklabels(['{:.1e}'.format(tmp2.min()), '{:.1e}'.format(tmp2.max())])

#   cbar.ax.tick_params(labelsize=textfs)
#   # pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
#   pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs)
#   pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
#   pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
#   pl.xticks(fontsize=textfs-4)
#   pl.yticks(fontsize=textfs-4)

# pl.show()




# # version 3: diameter, 2x2, signal decay in sci notation with only min and max, fixed spacing, decay in PERCENT
# textfs = 20
# pl.figure()
# pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
# pl.suptitle(r'Percent signal decay for multiple ($\Delta$, $\delta$) and diameters', fontsize=textfs+2)
# # pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)
# for iR,R in enumerate(Rs):
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

#   pl.subplot(nx, ny, iR+1)

#   tmp = S.reshape(len(list_DELTA), len(list_delta))
#   mask = np.isinf(tmp)
#   # percent
#   tmp2 = 100*np.ma.array(1-tmp, mask=mask)

#   pl.imshow(tmp2, interpolation='nearest', extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
#   cbar = pl.colorbar()

#   cbar.set_ticks([tmp2.min(), tmp2.max()])
#   cbar.set_ticklabels(['{:.1e} \%'.format(tmp2.min()), '{:.1e} \%'.format(tmp2.max())])
#   cbar.ax.tick_params(labelsize=textfs)
#   # pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
#   pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs)
#   pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
#   pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
#   pl.xticks(fontsize=textfs-4)
#   pl.yticks(fontsize=textfs-4)

# pl.show()



# def fmt(x, pos):
#     a, b = '{:.1e}'.format(x).split('e')
#     b = int(b)
#     return r'${} \times 10^{{{}}}$'.format(a, b)


# # version 4: diameter, 2x2, signal decay in sci notation with only min and max, fixed spacing, WITH \num{}
# textfs = 20
# pl.figure()
# pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
# pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters', fontsize=textfs+2)
# # pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)
# for iR,R in enumerate(Rs):
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

#   pl.subplot(nx, ny, iR+1)

#   tmp = S.reshape(len(list_DELTA), len(list_delta))
#   mask = np.isinf(tmp)
#   tmp2 = np.ma.array(1-tmp, mask=mask)

#   pl.imshow(tmp2, interpolation='nearest', extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
#   cbar = pl.colorbar(format=ticker.FuncFormatter(fmt))

#   cbar.set_ticks([tmp2.min(), tmp2.max()])
#   # cbar.set_ticklabels(['{:.1e}'.format(tmp2.min()), '{:.1e}'.format(tmp2.max())])


#   cbar.ax.tick_params(labelsize=textfs)
#   # pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
#   pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs)
#   pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
#   pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
#   pl.xticks(fontsize=textfs-4)
#   pl.yticks(fontsize=textfs-4)

# pl.show()




# def fmt(x, pos):
#     a, b = '{:.1e}'.format(x).split('e')
#     b = int(b)
#     return r'${} \times 10^{{{}}}$ \%'.format(a, b)



# # version 5: diameter, 2x2, signal decay in sci notation with only min and max, fixed spacing, decay in PERCENT, WITH \num{}
# textfs = 20
# pl.figure()
# pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
# pl.suptitle(r'Percent signal decay for multiple ($\Delta$, $\delta$) and diameters', fontsize=textfs+2)
# # pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)
# for iR,R in enumerate(Rs):
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

#   pl.subplot(nx, ny, iR+1)

#   tmp = S.reshape(len(list_DELTA), len(list_delta))
#   mask = np.isinf(tmp)
#   # percent
#   tmp2 = 100*np.ma.array(1-tmp, mask=mask)

#   pl.imshow(tmp2, interpolation='nearest', extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
#   cbar = pl.colorbar(format=ticker.FuncFormatter(fmt))

#   cbar.set_ticks([tmp2.min(), tmp2.max()])
#   # cbar.set_ticklabels(['{:.1e} \%'.format(tmp2.min()), '{:.1e} \%'.format(tmp2.max())])

#   cbar.ax.tick_params(labelsize=textfs)
#   # pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
#   pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs)
#   pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
#   pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
#   pl.xticks(fontsize=textfs-4)
#   pl.yticks(fontsize=textfs-4)

# pl.show()





# # version 6: diameter, 2x2, signal decay in sci notation with only ROUNDED min and max, fixed spacing, decay in PERCENT, WITH \num{}
# textfs = 20
# pl.figure()
# pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
# pl.suptitle(r'Percent signal decay for multiple ($\Delta$, $\delta$) and diameters', fontsize=textfs+2)
# # pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)
# for iR,R in enumerate(Rs):
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

#   pl.subplot(nx, ny, iR+1)

#   tmp = S.reshape(len(list_DELTA), len(list_delta))
#   mask = np.isinf(tmp)
#   # percent
#   tmp2 = 100*np.ma.array(1-tmp, mask=mask)

#   # lowR = np.ceil(tmp2.min(), int(np.ceil(-np.log10(tmp2.min()))))
#   # higR = np.round(tmp2.max(), int(np.ceil(-np.log10(tmp2.max()))))


#   a, b = '{:.3e}'.format(tmp2.min()).split('e')
#   c, d = a.split('.') 
#   c = int(c)
#   b = int(b)
#   lowR = c*10**b

#   a, b = '{:.3e}'.format(tmp2.max()).split('e')
#   c, d = a.split('.') 
#   if d != '000':
#       c = int(c) + 1
#       b = int(b)
#       if c == 10:
#           c = 1
#           b += 1
#   else:
#       c = int(c)
#       b = int(b)
#   higR = c*10**b

#   pl.imshow(tmp2, interpolation='nearest', vmin=lowR, vmax=higR, extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
#   cbar = pl.colorbar(format=ticker.FuncFormatter(fmt))

    
#   # cbar.set_ticks([tmp2.min(), tmp2.max()])
#   cbar.set_ticks([lowR, higR])
#   # cbar.set_ticklabels(['{:.1e} \%'.format(tmp2.min()), '{:.1e} \%'.format(tmp2.max())])

#   cbar.ax.tick_params(labelsize=textfs)
#   # pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
#   pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs)
#   pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
#   pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
#   pl.xticks(fontsize=textfs-4)
#   pl.yticks(fontsize=textfs-4)

# pl.show()





# print all digit before decimal and up to the first 2 decimal
def fmt(x, pos):
    a, b = '{:.15f}'.format(x).split('.')
    # pre decimal
    a = int(a)
    # search for first non zero decimal
    notZero = np.array([digt!='0' for digt in b])
    posFirstDigit = np.where(notZero)[0][0]
    # check for rounding with 3rd digit
    if int(b[posFirstDigit+2]) < 5:
        c = '0'*posFirstDigit + b[posFirstDigit] + b[posFirstDigit+1]
    else:
        # we SHOULD check if b[posFirstDigit+1]+1 is 10, and if it is we increase b[posFirstDigit] by one and if it is also 10 now ....
        # but I wont
        c = '0'*posFirstDigit + b[posFirstDigit] + str(int(b[posFirstDigit+1])+1)

    return r'{:}.{:} \%'.format(a, c)



# version 7 and final: diameter, 2x2, signal decay in sci notation with only min and max, fixed spacing, decay in PERCENT, WITH \num{}
# v5 with bigger text, no title and no sci notation
textfs = 20
pl.figure()
pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.35)
# pl.suptitle(r'Percent signal decay for multiple ($\Delta$, $\delta$) and diameters', fontsize=textfs+2)
# pl.suptitle(r'Signal decay for multiple ($\Delta$, $\delta$) and diameters, using in-vivo diffusivity and maximum Connectom gradient strength', fontsize=textfs+2)
for iR,R in enumerate(Rs):
    S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)

    pl.subplot(nx, ny, iR+1)

    tmp = S.reshape(len(list_DELTA), len(list_delta))
    mask = np.isinf(tmp)
    # percent
    tmp2 = 100*np.ma.array(1-tmp, mask=mask)

    pl.imshow(tmp2, interpolation='nearest', extent=[1e3*(list_delta.min()-spacing_delta/2.),1e3*(list_delta.max()+spacing_delta/2.),1e3*(list_DELTA.max()+spacing_DELTA/2.),1e3*(list_DELTA.min()-spacing_DELTA/2.)])
    cbar = pl.colorbar(format=ticker.FuncFormatter(fmt))

    cbar.set_ticks([tmp2.min(), tmp2.max()])
    # cbar.set_ticklabels(['{:.1e} \%'.format(tmp2.min()), '{:.1e} \%'.format(tmp2.max())])

    cbar.ax.tick_params(labelsize=textfs)
    # pl.title(r'R = {:.1f} $\mu$m'.format(R*1e6), fontsize=textfs)
    pl.title(r'd = {:.1f} $\mu$m'.format(R*2e6), fontsize=textfs+2)
    pl.xlabel(r'$\delta$ (ms)', fontsize=textfs)
    pl.ylabel(r'$\Delta$ (ms)', fontsize=textfs)
    pl.xticks(fontsize=textfs-4)
    pl.yticks(fontsize=textfs-4)

pl.show()











# D_invivo = 2e-9

# list_G = [0.3]
# list_DELTA = np.linspace(5e-3, 15e-3, 96)
# list_delta = [5e-3]
# scheme = expand_scheme(list_G, list_DELTA, list_delta)


# Rs = np.linspace(0.5e-6, 3.0e-6, 6)


# pl.figure()

# for R in Rs:
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)
#   pl.plot(1e3*list_DELTA, S, label='R = {:.1f} um'.format(R*1e6))
# pl.xlabel('DELTA (ms)')
# pl.ylabel('Signal')
# pl.title('delta = {} ms'.format(list_delta[0]*1e3))

# pl.show()










# D_invivo = 2e-9

# list_G = [0.3]
# list_DELTA = np.linspace(5e-3, 15e-3, 96)
# list_delta = [5e-3]
# scheme = expand_scheme(list_G, list_DELTA, list_delta)


# Rs = np.linspace(0.5e-6, 3.0e-6, 6)


# pl.figure()

# for R in Rs:
#   S = vangelderen_cylinder_perp(D_invivo, R, scheme, 50)
#   pl.plot(1e3*list_DELTA, 1 - (S / S[0]), label='R = {:.1f} um'.format(R*1e6))

# pl.axhline(0.01)
# pl.xlabel('DELTA (ms)')
# pl.ylabel('Signal')
# pl.title('delta = {} ms'.format(list_delta[0]*1e3))
# pl.legend()

# pl.show()







