import numpy as np

def separable(f):
	"""Checks if the passed filter is separable or not. 
	Args:
		f: The filter in question. 
	Returns: 
		True if the filter is separable else false.
	"""
	U, s, V = np.linalg.svd(f)
	if s[0] > 0 and s[1] < 0.000001: # chose 0.000001 arbitrarily.
		return True
	else:
		return False