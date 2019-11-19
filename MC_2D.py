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
	return (known_dx**2 / num_dim) / (2*known_D)












