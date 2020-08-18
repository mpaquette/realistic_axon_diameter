import numpy as np
import pylab as pl 

import MC_2D as mc 

from matplotlib.animation import FuncAnimation


desired_D = 2.0 # um^2/ms
# desired_D = 0.66 # um^2/ms
maximum_dx = 0.1 # um
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


desired_circle_diameter = 2.0 # um
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











