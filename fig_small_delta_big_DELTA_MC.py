import numpy as np
import pylab as pl 

import MC_2D as mc 

# from matplotlib.animation import FuncAnimation

import vangelderen as vg



desired_D = 2.0 # um^2/ms
# desired_D = 0.66 # um^2/ms
maximum_dx = 0.01 # um
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


desired_circle_diameter = 2.5 # um
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
minimum_N_particule = 45000
# compute number of particule per positions
N_per_pos = int(np.ceil(minimum_N_particule/float(N_pos)))
init_particule = mc.initialize_uniform_particule(canvas, N_per_pos)
# actual_N_particule = N_pos * N_per_pos
actual_N_particule = init_particule.shape[0]
print('Desired vs Actual #particule {}   {}'.format(minimum_N_particule, actual_N_particule))
print('{} positions with {} particules each'.format(N_pos, N_per_pos))


# computation num of time steps
desired_simulation_length = 100 # ms
num_dt = int(np.ceil(desired_simulation_length/float(dt)))
actual_simulation_length = num_dt*dt

print('Simulation duration')
print('desired = {:.2f} ms'.format(desired_simulation_length))
print('actual  = {:.2f} ms'.format(actual_simulation_length))
print('number of timestep  = {}'.format(num_dt))


desired_early_length = 5 # ms
num_dt_early = int(np.ceil(desired_early_length/float(dt)))
actual_early_length = num_dt_early*dt

print('Early Simulation duration')
print('desired = {:.2f} ms'.format(desired_early_length))
print('actual  = {:.2f} ms'.format(actual_early_length))
print('number of timestep  = {}'.format(num_dt_early))


desired_logging_dt = 0.05 # ms
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






timestamp = time_history*dt # ms
time_interval = timestamp[1:] - timestamp[:-1] # ms
relative_position = (particule_history[1:].astype(np.int16) - particule_history[0].astype(np.int16)) * dx # um



# np.save('/home/raid2/paquette/work/realistic_axon_diameter/MC_trajectory/'+'timestamp'+'__Diff_2p00'+'__Diam_2p50'+'.npy', timestamp)
# np.save('/home/raid2/paquette/work/realistic_axon_diameter/MC_trajectory/'+'relpos'+'__Diff_2p00'+'__Diam_2p50'+'.npy', relative_position)







G = 0.3 # T/m
g_orient_x = np.array([1., 0.])
g_orient_y = np.array([0., 1.])



def fancy_round(arr, val):
    return val*(np.round(arr / val)).astype(np.int)



list_DELTA = np.linspace(10e-3, 50e-3, 9) 
list_delta = np.linspace(5e-3, 50e-3, 19)

list_DELTA = np.array([fancy_round(x, 0.0025) for x in list_DELTA])
list_delta = np.array([fancy_round(x, 0.0025) for x in list_delta])





param = []
MC_E = []
vg_E = []

for bigdelta in list_DELTA:
    for smalldelta in list_delta:
        if smalldelta-1e-6 <= bigdelta:
            # if (fancy_round(bigdelta,0.0025), fancy_round(smalldelta,0.0025)) not in param:
            if (bigdelta, smalldelta) not in param:
                print((bigdelta, smalldelta))
                param.append((bigdelta, smalldelta))
                g_norm = mc.square_gradient(G, smalldelta, bigdelta, 1e-3*timestamp)
                # minimize variance a little bit
                E_x = mc.signal_from_MC(g_norm[1:], g_orient_x, 1e-6*relative_position, 1e-3*time_interval)
                E_y = mc.signal_from_MC(g_norm[1:], g_orient_y, 1e-6*relative_position, 1e-3*time_interval)
                MC_E.append((E_x+E_y)/2.)
                E_theory = vg.vangelderen_cylinder_perp(mc.compute_D(dx, dt)*1e-9, 0.5*actual_circle_diameter*1e-6, np.array([[G, bigdelta, smalldelta]]))
                vg_E.append(E_theory)



MC_data = np.array(MC_E)
vg_data = np.array(vg_E)[:,0]






grid_MC = np.nan*np.zeros((len(list_DELTA), len(list_delta)))
grid_vg = np.nan*np.zeros((len(list_DELTA), len(list_delta)))


for iDel, bigdelta in enumerate(list_DELTA):
    for idel, smalldelta in enumerate(list_delta):
        if (bigdelta, smalldelta) in param:
            idx = param.index((bigdelta, smalldelta))
            grid_MC[iDel, idel] = MC_E[idx]
            grid_vg[iDel, idel] = vg_E[idx]



minv = min(MC_data.min(), vg_data.min())
maxv = max(MC_data.max(), vg_data.max())


pl.figure()
pl.imshow(grid_MC, interpolation='nearest', extent=[1e3*list_delta.min(),1e3*list_delta.max(),1e3*list_DELTA.max(),1e3*list_DELTA.min()], vmin=minv, vmax=maxv)
pl.title('MC')
pl.xlabel('delta (ms)')
pl.ylabel('DELTA (ms)')
pl.colorbar()


pl.figure()
pl.imshow(grid_vg, interpolation='nearest', extent=[1e3*list_delta.min(),1e3*list_delta.max(),1e3*list_DELTA.max(),1e3*list_DELTA.min()], vmin=minv, vmax=maxv)
pl.title('VG')
pl.xlabel('delta (ms)')
pl.ylabel('DELTA (ms)')
pl.colorbar()


pl.figure()
pl.imshow(grid_vg - grid_MC, interpolation='nearest', extent=[1e3*list_delta.min(),1e3*list_delta.max(),1e3*list_DELTA.max(),1e3*list_DELTA.min()])
pl.title('VG - MC')
pl.xlabel('delta (ms)')
pl.ylabel('DELTA (ms)')
pl.colorbar()



pl.show()




# smalldelta = 10e-3 # s
# bigdelta = 30e-3 # s

# g_norm = mc.square_gradient(G, smalldelta, bigdelta, 1e-3*timestamp)
# E = mc.signal_from_MC(g_norm[1:], g_orient, 1e-6*relative_position, 1e-3*time_interval)
# E_theory = vg.vangelderen_cylinder_perp(mc.compute_D(dx, dt)*1e-9, 0.5*actual_circle_diameter*1e-6, np.array([[G, bigdelta, smalldelta]]))






