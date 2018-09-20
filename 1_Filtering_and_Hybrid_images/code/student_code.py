import numpy as np
from time import time

def my_imfilter(image, filter):
	"""
	Apply a filter to an image. Return the filtered image.

	Args
	- image: numpy nd-array of dim (m, n, c)
	- filter: numpy nd-array of dim (k, k)
	Returns
	- filtered_image: numpy nd-array of dim (m, n, c)

	HINTS:
	- You may not use any libraries that do the work for you. Using numpy to work
	with matrices is fine and encouraged. Using opencv or similar to do the
	filtering for you is not allowed.
	- I encourage you to try implementing this naively first, just be aware that
	it may take an absurdly long time to run. You will need to get a function
	that takes a reasonable amount of time to run so that the TAs can verify
	your code works.
	- Remember these are RGB images, accounting for the final image dimension.
	"""

	assert filter.shape[0] % 2 == 1
	assert filter.shape[1] % 2 == 1

	# Check if image is grayscale or color
	num_dims = len(image.shape)
	if num_dims > 2: # 3rd dimension = number of channels
		print("color image received")
		img_dims = image.shape[-1] 
	else: # Grayscale images generally have 2 dimensions = rows, columns
		print("grayscale image received")
		img_dims = 1
		image = np.expand_dims(image, -1)

	img_r = image.shape[0]
	img_c = image.shape[1]
	filter_r = filter.shape[0]
	filter_c = filter.shape[1]

	# Determine amount of padding for image based on filter dimensions:
	pad_y = int((filter.shape[0] - 1) * 0.5)
	pad_x = int((filter.shape[1] - 1) * 0.5)

	filtered_image_list = []
	start = time()
	for c in range(img_dims):
		padded_img = np.copy( np.pad(image[:,:,c], ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect') )
		temp_img = np.zeros((img_r, img_c))
		for m in range(img_r):
			for n in range(img_c):
				patch = padded_img[m:(m+filter_r), n:(n+filter_c)]
				temp_img[m,n] = np.sum(np.multiply(patch, filter))
		filtered_image_list.append(np.expand_dims(temp_img, -1))
	print("filtered image generated in", time()-start,"seconds\n")

	if num_dims > 2:
		filtered_image = np.concatenate(filtered_image_list, axis=-1)
		return filtered_image
	else:
		return np.squeeze(filtered_image_list[0])

def create_hybrid_image(image1, image2, filter):
	"""
	Takes two images and creates a hybrid image. Returns the low
	frequency content of image1, the high frequency content of
	image 2, and the hybrid image.

	Args
	- image1: numpy nd-array of dim (m, n, c)
	- image2: numpy nd-array of dim (m, n, c)
	Returns
	- low_frequencies: numpy nd-array of dim (m, n, c)
	- high_frequencies: numpy nd-array of dim (m, n, c)
	- hybrid_image: numpy nd-array of dim (m, n, c)

	HINTS:
	- You will use your my_imfilter function in this function.
	- You can get just the high frequency content of an image by removing its low
	frequency content. Think about how to do this in mathematical terms.
	- Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
	as 'clipping'.
	- If you want to use images with different dimensions, you should resize them
	in the notebook code.
	"""

	assert image1.shape[0] == image2.shape[0]
	assert image1.shape[1] == image2.shape[1]
	assert image1.shape[2] == image2.shape[2]

	# --------------------Extract the low frequency content of image1:
	low_frequencies = my_imfilter(image1, filter)

	# --------------------Extract the low frequency content of image2:	
	
	low_frequencies_image2 = my_imfilter(image2, filter)
	temp_high_freq = image2 - low_frequencies_image2
	high_frequencies = np.clip(temp_high_freq, 0, 1)
	
	# --------------------Add low freq + high freq
	hybrid_image = np.clip(low_frequencies + temp_high_freq, 0, 1)

	return low_frequencies, high_frequencies, hybrid_image


