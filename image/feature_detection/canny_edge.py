import numpy as np

def canny_edge(img, threshold):

	""" Returns the edges of the given image applying non-maxima suppression.

	Args:
		img: The image we apply the canny edge detector to.
		threshold: The minimum edge strength edges we will consider. 
	Returns: 
		A grayscale image consisting of the edges in the image.
	"""
	sobel_x = np.array([[-1,0,1],
				 [-1,0,1],
				 [-1,0,1]])
	sobel_y = np.array([[-1,-1,-1],
				 [0,0,0],
				 [1,1,1]])

	img_grad, img_angles = gradient(img, sobel_x, sobel_y)
	#save_img(img_grad,"gradient_image.jpg")
	# from https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
	# this changes the range of the matrix to be an integer between 0 and 255 (including 0 and 255).
	norm_image = cv2.normalize(img_grad, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint)

	# We apply a threshholding here to get rid of the weak edges in the image.
	img_grad[norm_image<threshold] = 0

	# set borders to 0 edge strength so we don't have to account for this case
	# when considering pixels neighbors.
	img_grad[:,0] = 0
	img_grad[:,-1] = 0
	img_grad[0,:] = 0
	img_grad[-1,:] = 0

	output = np.zeros(img.shape)

	pi = np.pi
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			if (img_grad[row,col] != 0):

				# figure out which of the pixels neighbors to consider.
				# very brute force method

				a = img_angles[row,col]
				if (-pi/8 <= a and a <= pi/8) or  (7*pi/8 <= a and a <= pi) or (-pi <= a and a <= -7*pi/8):
					# left and right pixels
					x1 = img_grad[row,col-1]
					x2 = img_grad[row,col+1]
				elif (pi/8 <= a and a <= 3*pi/8) or (-7*pi/8 <= a and a <= -5*pi/8):
					# top left and bot right pixels
					x1 = img_grad[row-1,col-1]
					x2 = img_grad[row+1,col+1]
				elif (3*pi/8 <= a and a <= 5*pi/8) or (-5*pi/8 <= a and a <= -3*pi/8):
					# top and bottom pixels	
					x1 = img_grad[row-1,col]
					x2 = img_grad[row+1,col]
				elif (5*pi/8 <= a and a <= 7*pi/8) or (-3*pi/8 <= a and a <= -1*pi/8):
					#top right and bottom left pixels.
					x1 = img_grad[row-1,col+1]
					x2 = img_grad[row+1,col-1]


				if img_grad[row,col] > x1 and img_grad[row,col] > x2:
					output[row,col] = img_grad[row,col]


	return output