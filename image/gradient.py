import numpy as np
import convolve as convolve

def gradient(img, x_f, y_f):
	"""Return the gradient magnitude of img given the partial x and partial y
	filters x_f and y_f (could be prewitt or sobel) and their gradient direction angles.

	Args:
		img: The grayscale image.
		x_f: The partial x filter.
		y_f: The partial y filter.
	Returns: 
		A tuple consisting of
		(1) an image with the same shape as the argument img that 
		is the the gradient magnitude of the edges.
		(2) A matrix same size as img but each entry contains 
		the angle of direction of the gradient at that pixel.

	"""
	# compute the partial derivatives of the image.
	x_partial = convolve(img, x_f)
	y_partial = convolve(img, y_f)

	grad =  np.sqrt(np.multiply(x_partial,x_partial) + np.multiply(y_partial,y_partial))
	angles = np.arctan2(y_partial, x_partial)
	return grad, angles