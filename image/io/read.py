def read_image(filename):
	""" Reads the image with the given filename in the current
	directory and returns the grayscale version of it.


	Args:
		filename: The name of the file we want to read.
	Returns:
		The grayscale version of the image named filename.
	"""	
	# Discarding the images alpha channel if it exist.
	img = cv2.imread(filename)[:,:,:3]
	# opencv actually reads in images in BGR order instead of RGB. We convert it to RGB.
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return color.rgb2gray(img)
