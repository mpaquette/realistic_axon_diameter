import numpy as np
import pylab as pl 

import MC_2D as mc 

# from matplotlib.animation import FuncAnimation


# desired_D = 2.0 # um^2/ms
desired_D = 0.66 # um^2/ms
maximum_dx = 0.01 # um
maximum_dt = 0.05 # ms

# figure out weither dx or dt is the bottle neck
if mc.compute_dt(desired_D, maximum_dx) <= maximum_dt:
    dx = maximum_dx
    dt = mc.compute_dt(desired_D, maximum_dx)
elif mc.compute_dx(desired_D, maximum_dt) <= maximum_dx:
    dt = maximum_dt
    dx = mc.compute_dx(desired_D, maximum_dt)
else:
    # dx and dt are very weird w.r.t. D if we get here...
    print('D, dx, dt value are probably wrong')

# sanity check
print('desired  D = {:.2e}'.format(desired_D))
print('computed D = {:.2e}'.format(mc.compute_D(dx, dt)))
print('dx within limt? {}     {:.2e}  <=  {:.2e}'.format(dx <= maximum_dx, dx, maximum_dx))
print('dt within limt? {}     {:.2e}  <=  {:.2e}'.format(dt <= maximum_dt, dt, maximum_dt))


desired_circle_diameter = np.array([4.0, 2.0]) # um
# compute circle radius in canvas pixel
radius_pixel = (np.ceil((desired_circle_diameter/2.) / dx)).astype(np.int)
# radius_pixel = int(np.floor((circle_diameter/2.) / dx))
actual_circle_diameter =  2*radius_pixel*dx # um

print('Ellipse "diameter"')
print('desired = {:.2f} {:.2f} um'.format(*desired_circle_diameter))
print('actual  = {:.2f} {:.2f} um'.format(*actual_circle_diameter))
print('canvas radius  = {} {} pixels'.format(*radius_pixel))

canvas = mc.canvas_single_ellipse(radius_pixel, side_pixels=None, center_pixel=None)
pl.figure()
pl.imshow(canvas)
pl.show()


# uniform sampling
# count positions
N_pos = canvas.sum()
minimum_N_particule = 50000
# compute number of particule per positions
N_per_pos = int(np.ceil(minimum_N_particule/float(N_pos)))
actual_N_particule = N_pos * N_per_pos
init_particule = mc.initialize_uniform_particule(canvas, N_per_pos)
print('Desired vs Actual #particule {}   {}'.format(minimum_N_particule, actual_N_particule))
print('{} positions with {} particules each'.format(N_pos, N_per_pos))


# tmp = np.zeros_like(canvas)
# tmp[canvas.shape[0]//2 - 3:canvas.shape[0]//2 + 3, canvas.shape[1]//2 - 3:canvas.shape[1]//2 + 3] = 1
## particule = np.array(np.where(tmp)).T.astype(np.uint16)
# particule = np.array(np.where(tmp)).T.astype(np.int16)
# N_pos = particule.shape[0]
# minimum_N_particule = 1000
# N_per_pos = int(np.ceil(minimum_N_particule/float(N_pos)))
# actual_N_particule = N_pos * N_per_pos
# # duplicate positions N_per_pos times
## init_particule = np.repeat(particule, N_per_pos, axis=0).astype(np.uint16)
# init_particule = np.repeat(particule, N_per_pos, axis=0).astype(np.int16)

# computation num of time steps
desired_simulation_length = 20 # ms
num_dt = int(np.ceil(desired_simulation_length/float(dt)))
actual_simulation_length = num_dt*dt

print('Simulation duration')
print('desired = {:.2f} ms'.format(desired_simulation_length))
print('actual  = {:.2f} ms'.format(actual_simulation_length))
print('number of timestep  = {}'.format(num_dt))



desired_logging_dt = 0.1 # ms
sample_rate = int(np.floor(desired_logging_dt / float(dt)))
actual_logging_rate = dt*sample_rate

print('Simulation logging')
print('desired = {:.2f} ms'.format(desired_logging_dt))
print('actual  = {:.2f} ms'.format(actual_logging_rate))
print('timestep resolution = {}'.format(sample_rate))
print('History size will be ({}, {}, ndim={})'.format(int(np.ceil(num_dt / float(sample_rate))), actual_N_particule,canvas.ndim))

# this is a crucial point, we have to determines if it's too much memory-wise
## positions use np.uint16 datatype (ergo the 2 in the formula)
# positions use np.int16 datatype (ergo the 2 in the formula)
# 2 bytes per value, np.prod(history.shape) values (one for each particules at each logged time times ndim))
approx_GB_history = (int(np.ceil(num_dt / float(sample_rate))) * actual_N_particule * canvas.ndim * 2 ) / (1024.**3)
print('History size will be roughly {:.2f} GBytes'.format(approx_GB_history))




# Simulate
if approx_GB_history < 0.5:
    time_history, particule_history = mc.perform_MC_2D(canvas, init_particule, num_dt, sample_rate, True)
else:
    print('No-simulation, history seems on the big side, make sure you are not busting your memory')




timestamp = time_history*dt
time_interval = timestamp[1:] - timestamp[:-1]







delta = actual_simulation_length/2. # ms
Delta = actual_simulation_length - delta # ms

GMAX = 0.3 # T/m
# convert ms to s, result in T/m
g_norm = mc.square_gradient(GMAX, delta*1e-3, Delta*1e-3, timestamp[1:]*1e-3)

pl.figure()
pl.plot(timestamp[1:], g_norm)
pl.xlabel('time (ms)')
pl.ylabel('gradient (T/m)')
pl.show()

# echo condition
# (g_norm*time_interval).sum() == 0



#  T^-1 s^-1
gamma = 42.515e6 * 2*np.pi







# relative_position = (particule_history[1:].astype(np.int16) - particule_history[0].astype(np.int16)) * dx # um
relative_position = (particule_history[1:] - particule_history[0]) * dx # um


# g_orient = np.array([1., 1])
t = np.linspace(0, 2*np.pi, 100, endpoint=False)
v = []
for angle in t:
    g_orient = np.array([np.cos(angle), np.sin(angle)])

    g_orient = g_orient / np.linalg.norm(g_orient)

    absolute_distance = relative_position.dot(g_orient) * time_interval[:,None] # um ms
    total_dephasing = np.sum(absolute_distance * 1e-6*g_norm[:,None], axis=0) # T ms


    # all of those are still at the individual spin level
    effective_dephasing = gamma * 1e-3*total_dephasing # no unit
    complex_signal = np.exp(1j*(-effective_dephasing))
    magn_signal = np.abs(np.mean(complex_signal))

    # print(magn_signal)
    v.append(magn_signal)

v = np.array(v)
# print(np.mean(v))


pl.figure()
pl.plot(t, v)
pl.xlabel('gradient angle (rad)')
pl.xlabel('signal (normalized)')
pl.title('Signal as a function of gradient orientation')
pl.show()


# compute directional radius

# compute min max and mean radius VG

# compare with mean?




# ellipse
aa, bb = desired_circle_diameter/2.
rt = np.sqrt(aa**2*np.cos(t)**2 + bb**2*np.sin(t)**2)


pl.figure()
pl.plot(rt, v)
# pl.plot(rt[np.argsort(rt)], v[np.argsort(rt)])
pl.xlabel('distance to center (um)')
pl.ylabel('signal (normalized)')
pl.title('Signal as a function of distance to center')
pl.show()





import vangelderen as vg
E_theory = []
for rad in rt:
	tmp = vg.vangelderen_cylinder_perp(mc.compute_D(dx, dt)*1e-9, rad*1e-6, np.array([[GMAX, Delta*1e-3, delta*1e-3]]))
	E_theory.append(tmp)
E_theory = np.array(E_theory)



pl.figure()
pl.scatter(E_theory, v, label='data')
pl.xlabel('Theoretical signal for circle')
pl.ylabel('Simulated signal for ellipse')
line = np.linspace(0.95*E_theory.min(), 1.05*E_theory.max(), 100)
pl.plot(line, line, '--r', label='identity line')
pl.title('Vangelderen vs MC for ellipse at various orientation (matching distance and radius)')
pl.legend()
pl.show()





d_dict = 1e-6*np.arange(0.01,5.01,0.01)[::-1]
S_vg = np.array([vg.vangelderen_cylinder_perp(mc.compute_D(dx, dt)*1e-9, 0.5*d, np.array([[GMAX, Delta*1e-3, delta*1e-3]]), m_max=50) for d in d_dict])[:,0]


def estimate_diameter_from_dict(value, dico_signal=S_vg, dico_radius=d_dict):
	# assume dico_signal is sorted ascending
	# return linear interpolated radius

	# dico_signal[idx-1] < value <= dico_signal[idx]
	idx = np.searchsorted(S_vg, value, 'left')
	s_low = dico_signal[idx-1]
	s_high = dico_signal[idx]
	r_low = dico_radius[idx-1]
	r_high = dico_radius[idx]

	k = (value - s_low) / (s_high - s_low)
	return r_low + (r_high - r_low)*k





d_v = estimate_diameter_from_dict(v)

# mean mesured signal
mean_mesured_signal = v.mean()
# VG radius of mean measured signal
radius_from_mean_measured_signal = 0.5e6*estimate_diameter_from_dict(mean_mesured_signal)
# mean radius of single direction perceived VG radius
mean_radius_from_measured_signal = 0.5e6*estimate_diameter_from_dict(E_theory.mean())






pl.figure()

n = 7
mycolormap = pl.cm.hsv
_colors = (mycolormap(i) for i in np.linspace(0, 1, n))


# for each orientation, take measured signal and fit VG to get radius
pl.plot(0.5e6*d_v, label='VG radius from fitting', color=next(_colors))
# for each orientation, centerline "radius"
pl.plot(rt, label='"radius" from geometry', color=next(_colors))

pl.plot(0.5e6*d_v - rt, label='diff VG and Geom', color=next(_colors))

pl.axhline(radius_from_mean_measured_signal, label='Radius from VG on mean signal', color=next(_colors))
pl.plot(0.5e6*d_v - radius_from_mean_measured_signal, label='Diff signal', color=next(_colors))

pl.axhline(mean_radius_from_measured_signal, label='Mean geomtric "radius"', color=next(_colors))
pl.plot(rt - mean_radius_from_measured_signal, label='"Diff geometry', color=next(_colors))

pl.xlabel('gradient angle (rad)')
pl.ylabel('radius (um)')
pl.title('"perceived" Radius')
pl.legend()
pl.show()




# signal with E = E_perp*E_par
# bvalues = (gamma*delta*G)**2 * (DELTA - (delta/3.)) # s/m^2









# def estimate_diameter_from_dict_4(value, dico_signal=S_vg, dico_radius=d_dict):
# 	# assume dico_signal is sorted ascending
# 	# return quartic interpolated radius

# 	# dico_signal[idx-1] < value <= dico_signal[idx]
# 	idx = np.searchsorted(S_vg, value, 'left')
# 	s_low = dico_signal[idx-1]
# 	s_high = dico_signal[idx]
# 	d_low = dico_radius[idx-1]
# 	d_high = dico_radius[idx]

# 	# Let's keep it linear ....
# 	# k = (value**0.25 - s_low**0.25) / (s_high**0.25 - s_low**0.25) # i think theory point to this one
# 	k = (value**4 - s_low**4) / (s_high**4 - s_low**4) # but the experimental data point to this one
# 	return d_low + (d_high - d_low)*k


# test_d = []                                                                                                                                                                                                                                                         
# test_vg = []                                                                                                                                                                                                                                                         
# test_est = []
# test_est_4 = []

# for i in range(100):
# 	# random diameter
# 	test_d.append(0.01e-6 + 4.5e-6*np.random.rand())
# 	test_vg.append(vg.vangelderen_cylinder_perp(mc.compute_D(dx, dt)*1e-9, 0.5*test_d[-1], np.array([[GMAX, Delta*1e-3, delta*1e-3]]), m_max=50))
# 	test_est.append(estimate_diameter_from_dict(test_vg[-1]))
# 	test_est_4.append(estimate_diameter_from_dict_4(test_vg[-1]))

# err_lin = np.abs(np.array(test_d) - np.array(test_est).ravel())
# print('error linear = {}'.format(np.sum(err_lin)))
# err_qur = np.abs(np.array(test_d) - np.array(test_est_4).ravel())
# print('error quartic = {}'.format(np.sum(err_qur)))

# print('LIN MINUS QUR = {}'.format(np.sum(err_lin) - np.sum(err_qur)))




# pl.scatter(err_lin, test_d)
# pl.scatter(err_qur, test_d)















# ## gif package

# # compute the canvas at each logging point
# images = np.zeros((particule_history.shape[0], canvas.shape[0], canvas.shape[1]), dtype=np.uint32)
# for i in range(particule_history.shape[0]):
#     images[i] = mc.draw_particule(canvas.shape[0], particule_history[i])



# datas = images.copy()
# # maxv = datas.max()
# maxv = 2
# fps = 30.

# fig = pl.figure(figsize=(8,8))
# im = pl.imshow(np.zeros_like(datas[0]), interpolation='none', vmin=0, vmax=maxv)
# pl.axis('off')
# pl.colorbar()

# def animate_func(i):
#     im.set_array(datas[i])
#     return [im]

# anim = FuncAnimation(fig, animate_func, frames = range(datas.shape[0]), interval = 1000 / fps, blit = True)
# # anim.save('/home/raid2/paquette/Pictures/discrete_mc/inner/test_circle_4.mp4', fps=fps,  dpi=300)
# pl.show()






## distance_origin = particule_history[1:].astype(np.int16) - particule_history[0].astype(np.int16)
# distance_origin = particule_history[1:] - particule_history[0]


# sq_dist_x = (distance_origin[:,:,0]*dx)**2
# sq_dist_y = (distance_origin[:,:,1]*dx)**2

# sq_dist = np.linalg.norm(distance_origin*dx, axis=2)**2


# MSD_x = sq_dist_x.mean(axis=1)
# MSD_y = sq_dist_y.mean(axis=1)
# MSD = sq_dist.mean(axis=1)


# pl.figure()
# pl.plot(time_history[1:]*dt, MSD_x, label='MSD x')
# pl.plot(time_history[1:]*dt, MSD_y, label='MSD y')
# pl.plot(time_history[1:]*dt, MSD, label='MSD')
# pl.legend()
# pl.show()


# pl.figure()
# pl.plot(time_history[1:]*dt, MSD_x/(2*time_history[1:]*dt), label='D x')
# pl.plot(time_history[1:]*dt, MSD_y/(2*time_history[1:]*dt), label='D y')
# pl.plot(time_history[1:]*dt, MSD/(2*time_history[1:]*dt), label='D')
# pl.legend()
# pl.show()











