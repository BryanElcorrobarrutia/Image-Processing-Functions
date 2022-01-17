import numpy as np

def convolve(img, f):
	"""Returns the convolution of a given 2D filter over a grayscale image. 

	This function really computes the correlation of the 
	flipped horizontally and vertically filter over img
	which is the equivalent operation.

	Args:
		img: The grayscale image.
		f: The (2k+1)x(2k+1) filter.
	Returns: 
		An image with the same shape as the argument img that 
		is the result of the convolution between filter and img.
		
	"""

	# asserting assumptions of odd dimension filter and 1-channel image.
	assert(len(f.shape) == 2 and f.shape[0] % 2 == 1 and f.shape[1] % 2 == 1)
	assert(len(img.shape) == 2)

	k = (f.shape[0] - 1) // 2
	flipped = f[::-1,::-1]	

	# flatted the filter.
	f_flat = flipped.ravel()

	# Padding the input image.
	padded = np.pad(img, k, mode='constant', constant_values=0)
	output = np.zeros(img.shape)

	# Computes the correlation in matrix form (from lecture 2 slides)
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			# to account for padding of k zeros.
			i = row + k
			j = col + k
			output[row,col] = f_flat.T.dot(padded[i-k:i+k+1,j-k:j+k+1].ravel())

	return output