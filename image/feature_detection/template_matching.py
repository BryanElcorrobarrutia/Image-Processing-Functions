import numpy as np


def template_matching(img, template):
	""" Performs template matching with the given grayscale image template.

	Specifically it does the sliding window algorithm, computing 
	the normalized dot product between the template and patches
	of the image.

	Args:
		img: The grayscale img we will try to find the template in.
		template: The grayscale template we try to find in the image.
	Returns: 
		A grayscale image where the location of the brightest pixel
		is where the center of the template is in the image. 
	"""

	# creates a new template t with odd dimensions so a center is defined.
	if (template.shape[0] % 2 == 0) and (template.shape[1] % 2 == 0):
		#pad with a new bottom row and right column of zeros.
		t = np.pad(template, ((0,1), (0,1)), mode ='constant', constant_values = (0,0))
	elif (template.shape[0] % 2 == 0):
		#pad with a new bottom row of zeros.
		t = np.pad(template, ((0,1), (0,0)), mode ='constant', constant_values = (0,0))
	elif (template.shape[1] % 2 == 0):
		#pad with a new right column of zeros.
		t = np.pad(template, ((0,0), (0,1)), mode ='constant', constant_values = (0,0))


	# these numbers define the size of the patch we take from img.
	# namely a (2k+1)x(2g+1) dimensioned patch.
	k = (t.shape[0] - 1) // 2
	g = (t.shape[1] - 1) // 2


	# flatten the template.
	t = t.ravel() 

	t = t / np.linalg.norm(t)

	# Padding the input image.
	padded = np.pad(img, ((k,k), (g,g)), mode ='constant', constant_values = (0,0))
	output = np.zeros(img.shape)

	# Computes the normalized cross correlation in matrix form (from lecture 2 slides)
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			# to account for padding of zeros
			i = row + k
			j = col + g

			patch = padded[i-k:i+k+1,j-g:j+g+1].ravel()
			output[row,col] = t.T.dot(patch) / np.linalg.norm(patch)

	return output