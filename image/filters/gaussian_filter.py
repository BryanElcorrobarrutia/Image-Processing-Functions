def gaussian_1D(x, sigma):
  	""" Computes the 1D gaussian with mean 0 
	and a given standard deviation sigma at the point x.

	Args:
		x: The value to evaluate the gaussian.
		sigma: The standard deviation of the 1D gaussian
	Returns: 
		The output of the gaussian at x.		
	"""
  	return (1.0 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*(np.power(x,2))/np.power(sigma,2))

def gaussian_filter(sigma, n):
	""" Returns a discretization of an isotropic 
	2D gaussian with mean 0 and a given standard deviation sigma in
	the form of a nxn filter centered at (0,0).
	Args:
		sigma: The standard deviation of the isotropic 2D gaussian.
		n: The size of the filter returned. Assumed to be odd. 
	Returns: 
		A Gaussian Filter with dimensions nxn.
	"""

	# Since a 2D guassian is separable, what we do is 
	# generate two 1D guassian filters and compute
	# the outer product between them to get our 2D gaussian.


	# assert n is an odd number.
	assert(n > 0 and n % 2 == 1)

	k = (n - 1)//2

	gaussian = np.zeros(n)

	for i in range(n):
		# very similiar to the LoG_filter function.
		gaussian[i] = gaussian_1D((i - k)*0.8,sigma)

	output =  np.outer(gaussian, gaussian) 

	# we make sure it sums to 1.
	return output / np.sum(output)