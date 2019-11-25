import numpy as np
import pylab as pl 

import MC_2D as mc 

from matplotlib.animation import FuncAnimation


desired_D = 2.0 # um^2/ms
# desired_D = 0.66 # um^2/ms
maximum_dx = 0.025 # um
maximum_dt = 0.01 # ms

# figure out wether dx or dt is the bottle neck
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


desired_circle_diameter = 1.0 # um
# compute circle radius in canvas pixel
radius_pixel = int(np.ceil((desired_circle_diameter/2.) / dx))
# radius_pixel = int(np.floor((circle_diameter/2.) / dx))
actual_circle_diameter =  2*radius_pixel*dx # um

print('Circle diameter')
print('desired = {:.2f} um'.format(desired_circle_diameter))
print('actual  = {:.2f} um'.format(actual_circle_diameter))
print('canvas radius  = {} pixels'.format(radius_pixel))


canvas = mc.canvas_single_circle(radius_pixel, side_pixels=None, center_pixel=None)
# pl.figure()
# pl.imshow(canvas)
# pl.show()


# uniform sampling
# count positions
N_pos = canvas.sum()
minimum_N_particule = 10000
# compute number of particule per positions
N_per_pos = int(np.ceil(minimum_N_particule/float(N_pos)))
init_particule = mc.initialize_uniform_particule(canvas, N_per_pos)
# actual_N_particule = N_pos * N_per_pos
actual_N_particule = init_particule.shape[0]
print('Desired vs Actual #particule {}   {}'.format(minimum_N_particule, actual_N_particule))
print('{} positions with {} particules each'.format(N_pos, N_per_pos))


# tmp = np.zeros_like(canvas)
# tmp[canvas.shape[0]//2 - 3:canvas.shape[0]//2 + 3, canvas.shape[1]//2 - 3:canvas.shape[1]//2 + 3] = 1
# particule = np.array(np.where(tmp)).T.astype(np.uint16)
# N_pos = particule.shape[0]
# minimum_N_particule = 1000
# N_per_pos = int(np.ceil(minimum_N_particule/float(N_pos)))
# actual_N_particule = N_pos * N_per_pos
# # duplicate positions N_per_pos times
# init_particule = np.repeat(particule, N_per_pos, axis=0).astype(np.uint16)

# computation num of time steps
desired_simulation_length = 10 # ms
num_dt = int(np.ceil(desired_simulation_length/float(dt)))
actual_simulation_length = num_dt*dt

print('Simulation duration')
print('desired = {:.2f} ms'.format(desired_simulation_length))
print('actual  = {:.2f} ms'.format(actual_simulation_length))
print('number of timestep  = {}'.format(num_dt))


desired_early_length = 2 # ms
num_dt_early = int(np.ceil(desired_early_length/float(dt)))
actual_early_length = num_dt_early*dt

print('Early Simulation duration')
print('desired = {:.2f} ms'.format(desired_early_length))
print('actual  = {:.2f} ms'.format(actual_early_length))
print('number of timestep  = {}'.format(num_dt_early))


desired_logging_dt = 0.1 # ms
sample_rate = int(np.floor(desired_logging_dt / float(dt)))
actual_logging_rate = dt*sample_rate

print('Simulation logging')
print('desired = {:.2f} ms'.format(desired_logging_dt))
print('actual  = {:.2f} ms'.format(actual_logging_rate))
print('timestep resolution = {}'.format(sample_rate))
# print('History size will be ({}, {}, ndim={})'.format(int(np.ceil(num_dt / float(sample_rate))), actual_N_particule,canvas.ndim))


desired_early_logging_dt = 0.01 # ms
sample_rate_early = int(np.floor(desired_early_logging_dt / float(dt)))
actual_early_logging_rate = dt*sample_rate_early

print('Simulation early logging')
print('desired = {:.2f} ms'.format(desired_early_logging_dt))
print('actual  = {:.2f} ms'.format(actual_early_logging_rate))
print('timestep resolution = {}'.format(sample_rate_early))
# print('History size will be ({}, {}, ndim={})'.format(int(np.ceil(num_dt / float(sample_rate))), actual_N_particule,canvas.ndim))


print('History size will be ({}, {}, ndim={})'.format(int(np.ceil(num_dt_early / float(sample_rate_early)) + np.ceil((num_dt-num_dt_early) / float(sample_rate))), actual_N_particule,canvas.ndim))



# this is a crucial point, we have to determines if it's too much memory-wise
# positions use np.uint16 datatype (ergo the 2 in the formula
# 2 bytes per value, np.prod(history.shape) values (one for each particules at each logged time times ndim))
approx_GB_history = (int(np.ceil(num_dt_early / float(sample_rate_early)) + np.ceil((num_dt-num_dt_early) / float(sample_rate))) * actual_N_particule * canvas.ndim * 2 ) / (1024.**3)
print('History size will be roughly {:.2f} GBytes'.format(approx_GB_history))




# Simulate
if approx_GB_history < 0.5:
    time_history, particule_history = mc.perform_MC_2D_2times(canvas, init_particule, num_dt, num_dt_early, sample_rate_early, sample_rate, True)
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





import vangelderen as vg
E_theory = vg.vangelderen_cylinder_perp(mc.compute_D(dx, dt)*1e-9, 0.5*actual_circle_diameter*1e-6, np.array([[GMAX, Delta*1e-3, delta*1e-3]]))
print(E_theory)




relative_position = (particule_history[1:].astype(np.int16) - particule_history[0].astype(np.int16)) * dx # um


# g_orient = np.array([1., 1])
v = []
for angle in np.linspace(0, 2*np.pi, 10, endpoint=False):
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


print(np.mean(v))




