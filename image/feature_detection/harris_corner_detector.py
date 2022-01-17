import numpy as np

def harris_corner_detector(img, threshold, alpha, window_size):
	"""Returns a grayscale image that shows all the corners
	contained in the given img 

	Args:
		img: The grayscale image we want the corners of.
		threshold: The threshold we will apply to the "cornerness" measure 
		R to determine if there is a corner at a pixel or not.
		alpha: Is the value between 0.04-0.06 used to compute R the "cornerness"
		measure.
		window_size: Assumed to be an odd number to represent the dimension of the 
		window we use for non-maxima supression.
	Returns: 
		An image with the same shape as the argument img that 
		shows all the corners contains in img. 
	"""
	sobel_x  = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
	sobel_y  = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
		
	# create a gaussian filter of size 5x5 and a standard deviation of 1.
	gaussian = gaussian_filter(1, 5) 

	# we get the x and y derivatives of the image.
	img_x = convolve(img, sobel_x)
	img_y = convolve(img, sobel_y)

	# this will contain all the M matrices per pixel.
	M = np.zeros((img.shape[0], img.shape[1], 2,2))

	# we compute the values needed to fill in each M.
	img_x_2 = np.multiply(img_x, img_x)
	img_y_2 = np.multiply(img_y, img_y)
	img_x_y = np.multiply(img_x, img_y)

	# We convolve a gaussian filter over these as required.
	M_x_2 = convolve(img_x_2, gaussian)
	M_y_2 = convolve(img_y_2, gaussian)
	M_x_y = convolve(img_x_y, gaussian)

	# Fill out each of the M matrices per pixel.
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			M[x,y,0,0] = M_x_2[x,y]
			M[x,y,1,1] = M_y_2[x,y]
			M[x,y,0,1] = M_x_y[x,y]
			M[x,y,1,0] = M_x_y[x,y]

	# We compute the "cornerness" value at each pixel.
	R = np.zeros(img.shape)
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			g = (np.linalg.det(M[x,y]) - alpha*(np.trace(M[x,y]))**2)
			# we apply the threshold here.
			if g > threshold:
				R[x,y] = g

	# Apply non maxima supression to the "cornerness" scores and return it.
	return non_maxima_supression(R, window_size)