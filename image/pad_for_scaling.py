def pad_for_scaling(img, d, direction):
	"""
	If d = 1 then this function
	will insert columns of 0s between the columns of image,
	namely d-1 columns for later upsizing.

	Returning this new padded img.

	If d = 0 then this function
	will insert rows of 0s between the rows of image,
	namely d-1 rows for later upsizing.

	"""
	if direction == 1:
		# first column
		padded = img[:,0]
		pad = np.zeros((img.shape[0], d-1))
		for i in range(1,img.shape[1]):
			padded = np.column_stack((padded,pad,img[:,i]))
		return padded

	elif direction == 0:
		# first column
		img = img.T
		padded = img[:,0]
		pad = np.zeros((img.shape[0], d-1))
		for i in range(1,img.shape[1]):
			padded = np.column_stack((padded,pad,img[:,i]))
		return padded.T