def LoG(x, y, sigma):
	""" Computes the laplacian of an isotropic 2D gaussian with mean 0 
	and a given standard deviation sigma at the point (x,y).

	Args:
		(x,y): The coordinate to evaluate the laplacian of a gaussian.
		sigma: The standard deviation of the isotropic 2D gaussian.
	Returns: 
		The output of the laplacian of gaussian evaluated at (x,y).		
	"""
	# I derived the laplacian of gaussian with pen and pencil.

	# breaking computation into parts for readabilty.
	p1 = 1/(2*np.pi*np.power(sigma,4))
	p2 = np.exp(-0.5*(x**2 + y**2)/np.power(sigma,2))
	p3 = -2 + (x**2 + y**2)/np.power(sigma,2)

	return p1*p2*p3

def LoG_filter(sigma, n):
	""" Returns a discretization of the laplacian of an isotropic 
	2D gaussian with mean 0 and a given standard deviation sigma in
	the form of a nxn filter centered at (0,0).


	Args:
		sigma: The standard deviation of the isotropic 2D gaussian.
		n: The size of the filter returned. Assumed to be odd. 
	Returns: 
		A Laplacian of Gaussian filter with sigma standard deviation.
	"""

	# assert n is an odd number.
	assert(n > 0 and n % 2 == 1)


	k = (n - 1)//2

	LoG_filter = np.zeros((n,n))

	for row in range(n):
		for col in range(n):
			# we drink the indices into the laplacian by a factor of 0.8
			# for a slightly better discretization of the laplacian of gaussians.
			LoG_filter[row,col] = LoG((row - k)*0.8, (col - k)*0.8, sigma)


	# To make sure the LoG filter sums to 0 I just normalize
	# all the postive values to 1 and all the negative values
	# to -1.
	pos_sum = np.sum(LoG_filter[LoG_filter>0])
	neg_sum = np.sum(LoG_filter[LoG_filter<0]) * -1
	LoG_filter = LoG_filter.astype(float)
	LoG_filter[LoG_filter>0] = LoG_filter[LoG_filter>0] / pos_sum
	LoG_filter[LoG_filter<0] = LoG_filter[LoG_filter<0] / neg_sum

	# return a multiple of 8 of the filter so its entries 
	# are not all between -1 and 1. Still sums to 0.
	return LoG_filter * 8 