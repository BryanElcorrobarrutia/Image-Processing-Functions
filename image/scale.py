def scale(img, d):
	""" This function will upscale the given img by a factor of d.
	"""


	# It turns out that bilinear interpolation
	# can be done by doing 1 dimensional
	# linear interpolation one after the other.

	# full disclosure I looked at the wikipedia
	# https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
	# and read that this was the case.

	# create the kernel
	# we are using the linear discrete filter from lecture.
	# here I create it. From slide 29 of the image pyramid lecture.
	k = np.zeros((1, 2*d-1))
	k[0,d-1] = 1
	index = d
	for i in range(1,d):
		k[0,d-1+i] = (d-i)/ float(d)
	k[0,:d-1] = np.flip(k,1)[0,:d-1]


	# this we simply pad in the appropriate direction
	# and convolve our 1D kernel.
	img = pad_for_scaling(img, d, 1)
	img = convolve_1D(img, k)

	# pad in the appropriate direction
	# and convolve our 1D kernel.
	img = pad_for_scaling(img, d, 0)

	# need to reshape this to be a column vector
	# since we convolve this kernel vertically.
	k = k.reshape((2*d-1,1))
	img = convolve_1D(img, k)


	return img