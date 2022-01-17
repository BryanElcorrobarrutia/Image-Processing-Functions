import numpy as np
def non_maxima_supression(R, window_size):
	""" Returns the given image with applied non-maxima supression. 
	Args:
		img: The grayscale image we want the corners of.
		window_size: Assumed to be an odd number to represent the dimension of the 
		window we use for non-maxima supression.
	Returns: 
		An image with the same shape as the argument R that
		contains the remaining R values. 
		
	"""
	# asserting assumptions of odd dimension  and R-channel image.
	assert(window_size % 2 == 1)
	assert(len(R.shape) == 2)
	k = (window_size - 1)//2

	# Padding the input image.
	padded = np.pad(R, k, mode='constant', constant_values=0)
	output = np.zeros(R.shape)

	# Step size of window_size so we don't 
	# have overlapping patches we apply 
	# non_maxima supression to.
	for x in range(0,R.shape[0], window_size):
		for y in range(0,R.shape[1], window_size):
			# i,j for account for the padding.
			i = x + k
			j = y + k

			# extract the patch we will get the local max of.
			patch = padded[i-k:i+k+1,j-k:j+k+1]

			# replace that patch with all 0s except where the local max is.
			max_indices = np.where(patch == np.amax(patch))
			rep = np.zeros(patch.shape)
			rep[max_indices[0][0],max_indices[1][0]] = patch[max_indices[0][0],max_indices[1][0]]
			padded[i-k:i+k+1,j-k:j+k+1] = rep

	# extract the image out of the padded image.
	output =  padded[k:-k, k:-k]

	# we binarify the image.
	output[output > 0] = 255

	return output