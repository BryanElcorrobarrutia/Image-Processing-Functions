import numpy as np
import convolve_1D


def fast_convolve(img, f):
	"""Returns the convolution of a given 2D filter over a grayscale image. 

	This function really computes the correlation of the 
	flipped horizontally and vertically filter over img
	which is the equivalent operation.

	Faster function since it assumes filter is separable and
	take advantage of that.

	Args:
		img: The grayscale image.
		f: The (2k+1)x(2k+1) separable filter.
	Returns: 
		If the filter is separable, it returns an image with the same shape 
		as the argument img that is the result of the convolution between filter and img.
		Otherwise it returns false.
	"""
	U, s, V = np.linalg.svd(f)
	# checks if the filter is separable. 
	if s[0] > 0 and s[1] < 0.0000001: # chose this small number arbitarily.
		u = (np.sqrt(s[0])*U[:,0]).reshape((3,1))
		v = (np.sqrt(s[0])*V[0,:]).reshape((1,3))

		f_separated = np.outer(u,v)
		# asserts that the outer product between them actually is the filter.
		assert(np.allclose(f_separated, f))

		# first convolve v over the image and then u.
		return convolve_1D(convolve_1D(img, v), u)

	else:
		print("not separable")
		return False