import numpy as np

def convolve_1D(img, f):
	"""Returns the convolution of a given 1D filter over a grayscale image. 

	This function really computes the correlation of the 
	flipped filter over img which is the equivalent operation.

	Args:
		img: The grayscale image.
		f: The kx1 or 1xk filter where k is an odd number.
	Returns: 
		An image with the same shape as the argument img that 
		is the result of the convolution between filter and img.
		
	"""

	if (f.shape[0] == 1):
		k = (f.shape[1] - 1) // 2
		flipped = f[:,::-1]	
	else: 
		k = (f.shape[0] - 1) // 2
		flipped = f[::-1, :]	
	

	# flatted the filter.
	flat = flipped.ravel()

	# Padding the input image.
	padded = np.pad(img, k, mode='constant', constant_values=0)
	output = np.zeros(img.shape)

	# Computes the correlation in matrix form (from lecture 2 slides)


	if (f.shape[0] == 1):
		for row in range(img.shape[0]):
			for col in range(img.shape[1]):
				# to account for padding of k zeros.
				i = row + k
				j = col + k
				output[row,col] = flat.dot(padded[i,j-k:j+k+1])
	elif (f.shape[1] == 1):
		for row in range(img.shape[0]):
			for col in range(img.shape[1]):
				# to account for padding of k zeros.
				i = row + k
				j = col + k
				output[row,col] = flat.dot(padded[i-k:i+k+1,j].ravel())

	return output