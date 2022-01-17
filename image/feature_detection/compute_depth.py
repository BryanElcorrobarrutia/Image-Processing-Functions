def compute_depth(img, f, T):
	""" Computes the depth of the image given a focal length f and baseline T.
	Args:
		img: An image to compute the depth of.
		f: The focal lengh of the camera in meters.
		T: The baseline length between cameras in meters.
	Returns:
		A 2D matrix the same dimensions as img where each entry contains the depth
		of the pixel.
	"""
	depth = np.zeros(img.shape)
	num = f*T # so we don't ahve to compute this more than once.
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			# if the disparity is 0 we cap the depth at 40 meters.
			if img[x,y] != 0: 
				# if the depth is greater than 40 meters we clip it to be 40 meters.
				depth[x,y] = num / img[x,y] 
				if depth[x,y] > 40:
					depth[x,y] = 40
			else:
				depth[x,y] = 40 
	return depth