import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from time import time 
import matplotlib.pyplot as plt

def get_tiny_images(image_paths):
	"""
	This feature is inspired by the simple tiny images used as features in
	80 million tiny images: a large dataset for non-parametric object and
	scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
	Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
	pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

	To build a tiny image feature, simply resize the original image to a very
	small square resolution, e.g. 16x16. You can either resize the images to
	square while ignoring their aspect ratio or you can crop the center
	square portion out of each image. Making the tiny images zero mean and
	unit length (normalizing them) will increase performance modestly.

	Useful functions:
	-   cv2.resize
	-   use load_image(path) to load a RGB images and load_image_gray(path) to
	  load grayscale images

	Args:
	-   image_paths: list of N elements containing image paths

	Returns:
	-   feats: N x d numpy array of resized and then vectorized tiny images
	        e.g. if the images are resized to 16x16, d would be 256
	"""
	# dummy feats variable
	feats = []
	standardize_pixels 	= True
	unit_norm			= False

	if standardize_pixels:
		print("Standardizing pixels")
	elif unit_norm:
		print("Transforming pixels to zero mean and unit length")
	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################

	for path in image_paths:
		curr_img 	= load_image_gray(path)
		curr_img 	= cv2.resize(curr_img,(16, 16), interpolation=cv2.INTER_AREA)
		pixels_vec	= curr_img.flatten()

		if standardize_pixels:
			pixels_mean = np.mean(pixels_vec)
			pixels_std 	= np.std(pixels_vec)
			pixels_vec 	= (pixels_vec - pixels_mean)/pixels_std 
		elif unit_norm:
			pixels_mean = np.mean(pixels_vec)
			pixels_norm = np.linalg.norm(pixels_vec)
			pixels_vec 	= (pixels_vec - pixels_mean)/pixels_norm 

		feats.append(np.expand_dims(pixels_vec, 0))

	feats = np.concatenate(feats,0)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return feats

def build_vocabulary(image_paths, vocab_size):
	"""
	This function will sample SIFT descriptors from the training images,
	cluster them with kmeans, and then return the cluster centers.

	Useful functions:
	-   Use load_image(path) to load RGB images and load_image_gray(path) to load
	      grayscale images
	-   frames, descriptors = vlfeat.sift.dsift(img)
	    http://www.vlfeat.org/matlab/vl_dsift.html
	      -  frames is a N x 2 matrix of locations, which can be thrown away
	      here (but possibly used for extra credit in get_bags_of_sifts if
	      you're making a "spatial pyramid").
	      -  descriptors is a N x 128 matrix of SIFT features
	    Note: there are step, bin size, and smoothing parameters you can
	    manipulate for dsift(). We recommend debugging with the 'fast'
	    parameter. This approximate version of SIFT is about 20 times faster to
	    compute. Also, be sure not to use the default value of step size. It
	    will be very slow and you'll see relatively little performance gain
	    from extremely dense sampling. You are welcome to use your own SIFT
	    feature code! It will probably be slower, though.
	-   cluster_centers = vlfeat.kmeans.kmeans(X, K)
	      http://www.vlfeat.org/matlab/vl_kmeans.html
	        -  X is a N x d numpy array of sampled SIFT features, where N is
	           the number of features sampled. N should be pretty large!
	        -  K is the number of clusters desired (vocab_size)
	           cluster_centers is a K x d matrix of cluster centers. This is
	           your vocabulary.

	Args:
	-   image_paths: list of image paths.
	-   vocab_size: size of vocabulary

	Returns:
	-   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
	  cluster center / visual word
	"""
	# Load images from the training set. To save computation time, you don't
	# necessarily need to sample from all images, although it would be better
	# to do so. You can randomly sample the descriptors from each image to save
	# memory and speed up the clustering. Or you can simply call vl_dsift with
	# a large step size here, but a smaller step size in get_bags_of_sifts.
	#
	# For each loaded image, get some SIFT features. You don't have to get as
	# many SIFT features as you will in get_bags_of_sift, because you're only
	# trying to get a representative sample here.
	#
	# Once you have tens of thousands of SIFT features from many training
	# images, cluster them with kmeans. The resulting centroids are now your
	# visual word vocabulary.

	# dim = 128      # length of the SIFT descriptors that you are going to compute.
	# vocab = np.zeros((vocab_size,dim))

	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################
	descriptors_list = []
	start_time = time()

	for path in image_paths:
		curr_img = load_image_gray(path)
		_, descriptors = vlfeat.sift.dsift(curr_img, step=20, fast=True)
		# _, descriptors = vlfeat.sift.dsift(curr_img, step=10, fast=False)
		descriptors = descriptors.astype(np.float32)
		descriptors_list.append(descriptors)
	descriptors_list = np.concatenate(descriptors_list, 0)
	vocab = vlfeat.kmeans.kmeans(descriptors_list, vocab_size)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	print("Time taken to build vocabulary:", time() - start_time)

	return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
	"""
	This feature representation is described in the handout, lecture
	materials, and Szeliski chapter 14.
	You will want to construct SIFT features here in the same way you
	did in build_vocabulary() (except for possibly changing the sampling
	rate) and then assign each local feature to its nearest cluster center
	and build a histogram indicating how many times each cluster was used.
	Don't forget to normalize the histogram, or else a larger image with more
	SIFT features will look very different from a smaller version of the same
	image.

	Useful functions:
	-   Use load_image(path) to load RGB images and load_image_gray(path) to load
	      grayscale images
	-   frames, descriptors = vlfeat.sift.dsift(img)
	      http://www.vlfeat.org/matlab/vl_dsift.html
	    frames is a M x 2 matrix of locations, which can be thrown away here
	      (but possibly used for extra credit in get_bags_of_sifts if you're
	      making a "spatial pyramid").
	    descriptors is a M x 128 matrix of SIFT features
	      note: there are step, bin size, and smoothing parameters you can
	      manipulate for dsift(). We recommend debugging with the 'fast'
	      parameter. This approximate version of SIFT is about 20 times faster
	      to compute. Also, be sure not to use the default value of step size.
	      It will be very slow and you'll see relatively little performance
	      gain from extremely dense sampling. You are welcome to use your own
	      SIFT feature code! It will probably be slower, though.
	-   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
	      finds the cluster assigments for features in data
	        -  data is a M x d matrix of image features
	        -  vocab is the vocab_size x d matrix of cluster centers
	        (vocabulary)
	        -  assignments is a Mx1 array of assignments of feature vectors to
	        nearest cluster centers, each element is an integer in
	        [0, vocab_size)

	Args:
	-   image_paths: paths to N images
	-   vocab_filename: Path to the precomputed vocabulary.
	      This function assumes that vocab_filename exists and contains an
	      vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
	      or visual word. This ndarray is saved to disk rather than passed in
	      as a parameter to avoid recomputing the vocabulary every run.

	Returns:
	-   image_feats: N x d matrix, where d is the dimensionality of the
	      feature representation. In this case, d will equal the number of
	      clusters or equivalently the number of entries in each image's
	      histogram (vocab_size) below.
	"""
	# load vocabulary
	with open(vocab_filename, 'rb') as f:
		vocab = pickle.load(f)
	vocab_size = vocab.shape[0]
	feats = []

	start_time = time()
	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################
	for path in image_paths:
		curr_img = load_image_gray(path)
		_, descriptors = vlfeat.sift.dsift(curr_img, step=3, fast=True)
		# _, descriptors = vlfeat.sift.dsift(curr_img, step=5, fast=False)
		descriptors = descriptors.astype(np.float32)
		assignments = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)
		hist, _ = np.histogram(assignments, bins=np.arange(vocab_size+1), density=True)
		feats.append( np.expand_dims(hist, axis=0) )

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	feats = np.concatenate(feats, axis=0)	
	
	print("Time taken to construct bag-of-SIFT features:", time() - start_time)

	return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):
	"""
	This function will predict the category for every test image by finding
	the training image with most similar features. Instead of 1 nearest
	neighbor, you can vote based on k nearest neighbors which will increase
	performance (although you need to pick a reasonable value for k).

	Useful functions:
	-   D = sklearn_pairwise.pairwise_distances(X, Y)
	    computes the distance matrix D between all pairs of rows in X and Y.
	      -  X is a N x d numpy array of d-dimensional features arranged along
	      N rows
	      -  Y is a M x d numpy array of d-dimensional features arranged along
	      N rows
	      -  D is a N x M numpy array where d(i, j) is the distance between row
	      i of X and row j of Y

	Args:
	-   train_image_feats:  N x d numpy array, where d is the dimensionality of
	      the feature representation
	-   train_labels: N element list, where each entry is a string indicating
	      the ground truth category for each training image
	-   test_image_feats: M x d numpy array, where d is the dimensionality of the
	      feature representation. You can assume N = M, unless you have changed
	      the starter code
	-   metric: (optional) metric to be used for nearest neighbor.
	      Can be used to select different distance functions. The default
	      metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
	      well for histograms

	Returns:
	-   test_labels: M element list, where each entry is a string indicating the
	      predicted category for each testing image
	"""
	test_labels = []
	perform_kNN = True
	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################
	k = 18
	D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats)

	if perform_kNN:
		train_labels = np.array(train_labels)
		for i in range(D.shape[0]):
			sorted_indices 			= np.argsort(D[i,:])[:k]
			nearest_labels 			= train_labels[sorted_indices]
			unique_labels, counts 	= np.unique(nearest_labels, return_counts=True)
			max_count_index			= np.argmax(counts)
			test_labels.append(unique_labels[max_count_index])

	else:
		for i in range(D.shape[0]):
			test_labels.append( train_labels[np.argmin(D[i,:])] )

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return  test_labels

def get_targets(target_label, labels_list):

	targets, weights = [], []
	for l in labels_list:
		if target_label == l:
			targets.append(1.0)
			weights.append(100.0)
		else:
			targets.append(-1.0)
			weights.append(1.0)
	return np.array(targets), np.array(weights) 


def svm_classify(train_image_feats, train_labels, test_image_feats):
	
	"""
	This function will train a linear SVM for every category (i.e. one vs all)
	and then use the learned linear classifiers to predict the category of
	every test image. Every test feature will be evaluated with all 15 SVMs
	and the most confident SVM will "win". Confidence, or distance from the
	margin, is W*X + B where '*' is the inner product or dot product and W and
	B are the learned hyperplane parameters.

	Useful functions:
	-   sklearn LinearSVC
	    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
	-   svm.fit(X, y)
	-   set(l)

	Args:
	-   train_image_feats:  N x d numpy array, where d is the dimensionality of
	      the feature representation
	-   train_labels: N element list, where each entry is a string indicating the
	      ground truth category for each training image
	-   test_image_feats: M x d numpy array, where d is the dimensionality of the
	      feature representation. You can assume N = M, unless you have changed
	      the starter code
	Returns:
	-   test_labels: M element list, where each entry is a string indicating the
	      predicted category for each testing image
	"""
	# categories
	categories = list(set(train_labels))

	# construct 1 vs all SVMs for each category
	lambda_value = 970.0
	print("lambda:", lambda_value)
	svms = {cat: LinearSVC(random_state=0, tol=1e-5, loss='hinge', C=lambda_value) for cat in categories}

	test_labels = []

	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################
	num_test_points = test_image_feats.shape[0]
	predictions = []
	W_mat = []
	b_vec = []
	# Iterate through categories and train each SVM:
	for cat_ in svms:
		# obtain targets to train SVM:
		y, w_s = get_targets(cat_, train_labels)
		svms[cat_].fit(train_image_feats, y, sample_weight=w_s)
		W_mat.append(svms[cat_].coef_)
		b_vec.append(svms[cat_].intercept_)
		predictions.append(np.expand_dims(svms[cat_].decision_function(test_image_feats), -1))

	# predictions = np.concatenate(predictions, axis=-1)
	# predicted_indices = np.argmax(predictions, axis=-1)	
	# test_labels = [categories[index] for index in predicted_indices]

	W_mat = np.concatenate(W_mat, 0)
	b_vec = np.expand_dims(np.concatenate(b_vec, -1),-1)
	predictions = W_mat.dot(test_image_feats.T) + b_vec	
	predicted_indices = np.argmax(predictions, axis=0)	
	test_labels = [categories[index] for index in predicted_indices]	
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return test_labels
