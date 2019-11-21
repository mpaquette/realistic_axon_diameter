import numpy as np
import warnings



def canvas_single_circle(radius_pixel, side_pixels=None, center_pixel=None):
	# draw canvas for 2D monte carlo simulation of a single circle
	# circle has radius radius_pixel pixels
	# circle is centered at center_pixel (None = canvas center)
	# canvas has size side_pixels[0] by side_pixels[1] (None = np.ceil(2*radius_pixel+1))

	if side_pixels is None:
		side_pixels = (int(np.ceil(2*radius_pixel+1)),)*2

	if center_pixel is None:
		center_pixel = (side_pixels[0]//2, side_pixels[1]//2)

	if 2*radius_pixel >= np.min(side_pixels):
		warnings.warn('Circle is too big for canvas, returning NaN')
		return np.nan

	# circle boundaries
	b_min_x = center_pixel[0] - radius_pixel
	b_max_x = center_pixel[0] + radius_pixel
	b_min_y = center_pixel[1] - radius_pixel
	b_max_y = center_pixel[1] + radius_pixel

	if (b_min_x < 0) or (b_max_x > side_pixels[0]) or (b_min_y < 0) or (b_max_y > side_pixels[1]):
		warnings.warn('Circle is poking out of canvas, returning NaN')
		return np.nan

	canvas = np.zeros(side_pixels, dtype=np.bool)

	for ix in range(side_pixels[0]):
		for iy in range(side_pixels[1]):
			if (ix-center_pixel[0])**2 + (iy-center_pixel[1])**2 <= radius_pixel**2:
				canvas[ix, iy] = True

	return canvas


def compute_dt(known_D, known_dx):
	# D = MSD / (2 dt) = ((dx^2)/ndim) / (2 dt)
	# dt = ((dx^2)/ndim) / (2 D)
	num_dim = 2
	return (known_dx**2 / num_dim) / (2*known_D)


def compute_dx(known_D, known_dt):
	# D = MSD / (2 dt) = ((dx^2)/ndim) / (2 dt)
	# dx = (2 dt D ndim)^0.5
	num_dim = 2
	return np.sqrt(2*known_dt*known_D*num_dim)


def compute_D(known_dx, known_dt):
	# D = MSD / (2 dt) = ((dx^2)/ndim) / (2 dt)
	num_dim = 2
	return (known_dx**2 / num_dim) / (2*known_dt)


# encodes unit vector x, -x, y, -y 
vec_2D = np.zeros((4,2))
vec_2D[0][0] = 1
vec_2D[1][0] = -1
vec_2D[2][1] = 1
vec_2D[3][1] = -1
def _hop(canvas, particule, moves=vec_2D):
	# Performs one timestep of discrete 2D montecarlo
	# 1/ndim chance of diffusing in each dimension
	# 1/2 of diffusing in each direction of given dimension
	# particule is (N,ndim)
	# In practice, this simply gives equal probability to everything in the moves vector
	c = np.random.choice(range(moves.shape[0]), particule.shape[0])
	# assumes square canvas and proper moves vector for the dimensionality
	new_particule = np.clip(particule+moves[c], 0, canvas.shape[0]-1).astype(np.uint16)
	return new_particule


def draw_particule(size, particule):
	# ndim "square" canvas
	image = np.zeros((size,)*particule.shape[1], dtype=np.uint32)
	# loop over all particules
	for i in np.arange(particule.shape[0]):
		# image[particule[i,0], particule[i,1]] += 1
		image[tuple(particule[i,:])] += 1

		# # loop over all dimensions
		# pos = (particule[i,0], )
		# for j in range(1, particule.shape[1]):
		# 	pos += (particule[i,j], )

		# # increment canvas value
		# image[pos] += 1
	return image


def initialize_uniform_particule(canvas, N_per_pos):
	# 1 particule per canvas positions
	particule = np.array(np.where(canvas)).T.astype(np.uint16)
	# count positions
	N_part = particule.shape[0]
	# duplicate positions N_per_pos times
	particule = np.repeat(particule, N_per_pos, axis=0).astype(np.uint16)
	return particule



# maybe?
# uint8	Unsigned integer (0 to 255)
# uint16	Unsigned integer (0 to 65535)
def perform_MC_2D(canvas, init_position, num_dt, sample_rate):
	# Perform discrete Monte-Carlo
	# canvas encodes the boundaries [(N1, N2, ...) boolean]
	# init_position has the canvas position of each particules [(#particule, #dimensions) integer]
	# num_dt is the number of times steps
	# return a subsampled particules history (every sample_rate timestep)
	num_samples = int(np.ceil(num_dt / float(sample_rate)))
	time_history = np.empty((num_samples, ))
	particule_history = np.empty((num_samples, init_position.shape[0], init_position.shape[1]), dtype=np.uint16)
	i_log = 1
	# init
	particule_history[0] = init_position
	new_particule = init_position
	# iterate over all time step (timestep 0 is the init)
	for it in range(1, num_dt+1):
		# rotate values
		old_particule = new_particule.copy()
		# perform 1 timestep
		new_particule = _hop(canvas, old_particule, moves=vec_2D)
		# log new positions (sometime)
		if not (it%sample_rate):
			# log iteration so we can compute time outside of the function with the known dt
			time_history[i_log] = it
			particule_history[i_log] = new_particule
			i_log += 1

	return time_history, particule_history





