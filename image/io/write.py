def save_img(img, filename):
	""" Saves the given grayscale image to the given filename in the
	current directory.

	If the filename doesn't exist in the current directory then it 
	creates it and writes to it.

	Args:
		img: The grayscale image.
		filename: The name of the file we want to write to.
	Returns:
		It doesn't return anything.
	"""	

	# from https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
	# this changes the range of the matrix to be an integer between 0 and 255 (including 0 and 255).
	norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint)
	cv2.imwrite(filename, norm_image)