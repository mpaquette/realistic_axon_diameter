import numpy as np
import pylab as pl 

import MC_2D as mc 




desired_D = 2.0 # um^2/ms
maximum_dx = 0.05 # um
maximum_dt = 0.05 # ms

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


desired_circle_diameter = 1.5 # um
# compute circle radius in canvas pixel
radius_pixel = int(np.ceil((desired_circle_diameter/2.) / dx))
# radius_pixel = int(np.floor((circle_diameter/2.) / dx))
actual_circle_diameter =  2*radius_pixel*dx # um

print('Circle diameter')
print('desired = {:.2f} um'.format(desired_circle_diameter))
print('actual  = {:.2f} um'.format(actual_circle_diameter))
print('canvas radius  = {} pixels'.format(radius_pixel))


canvas = mc.canvas_single_circle(radius_pixel, side_pixels=None, center_pixel=None)

pl.figure()
pl.imshow(canvas)
pl.show()



# count positions
N_pos = canvas.sum()
minimum_N_particule = 10000
# compute number of particule per positions
N_per_pos = int(np.ceil(minimum_N_particule/float(N_pos)))
actual_N_particule = N_pos * N_per_pos
init_particule = mc.initialize_uniform_particule(canvas, N_per_pos)



# computation num of time steps
desired_simulation_length = 100e3 # ms
num_dt = int(np.ceil(desired_simulation_length/float(dt)))
actual_simulation_length = num_dt*dt

print('Simulation duration')
print('desired = {:.2f} ms'.format(desired_simulation_length))
print('actual  = {:.2f} um'.format(actual_simulation_length))
print('number of timestep  = {}'.format(num_dt))



desired_logging_dt = 0.05 # ms
sample_rate = int(np.floor(logging_dt / float(dt)))
actual_logging_rate = dt*sample_rate

print('Simulation logging')
print('desired = {:.2f} ms'.format(desired_logging_dt))
print('actual  = {:.2f} um'.format(actual_logging_rate))
print('timestep resolution = {}'.format(sample_rate))
print('History size will be ({}, {}, ndim)'.format(int(np.ceil(num_dt / float(sample_rate))), actual_N_particule))



time_history, particule_history = mc.perform_MC_2D(canvas, init_position, num_dt, sample_rate)




# image = mc.draw_particule(size, particule)





