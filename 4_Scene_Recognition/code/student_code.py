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
from sklearn.metrics import confusion_matrix
import os.path as osp
from utils import *
import sys

def get_tiny_images(image_paths, standardize_pixels=False, unit_norm=False):
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

	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################
	i = 0
	for path in image_paths:
		curr_img 	= load_image_gray(path)
		curr_img 	= cv2.resize(curr_img,(16, 16), interpolation=cv2.INTER_AREA)

		pixels_vec	= curr_img.flatten()

		if standardize_pixels:
			if i==0:
				print("applying standardization transform")
				i+=1
			pixels_mean = np.mean(pixels_vec)
			pixels_std 	= np.std(pixels_vec)
			pixels_vec 	= (pixels_vec - pixels_mean)/pixels_std 
		elif unit_norm:
			if i==0:
				print("applying unit norm transform")
				i+=1			
			pixels_mean = np.mean(pixels_vec)
			pixels_vec 	= (pixels_vec - pixels_mean)
			pixels_norm = np.linalg.norm(pixels_vec)
			pixels_vec  = pixels_vec/pixels_norm
		else:
			if i==0:
				print("using un-transformed pixel intensities")
				i+=1			

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
		_, descriptors = vlfeat.sift.dsift(curr_img, step=10, fast=True)
		
		descriptors = descriptors.astype(np.float32)
		descriptors_list.append(descriptors)
	descriptors_list = np.concatenate(descriptors_list, 0)
	vocab = vlfeat.kmeans.kmeans(descriptors_list, vocab_size)

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	# print("Time taken to build vocabulary:", time() - start_time)

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
		_, descriptors = vlfeat.sift.dsift(curr_img, step=5, fast=True)

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
    metric='euclidean', perform_kNN=False, k=1):
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
	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	#############################################################################
	D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats)

	if perform_kNN:
		train_labels_array = np.array(train_labels)
		for i in range(D.shape[0]):
			sorted_indices 						= np.argsort(D[i,:])[:k]
			nearest_labels 						= train_labels_array[sorted_indices]
			unique_labels, l_indices, counts 	= np.unique(nearest_labels, return_index=True, return_counts=True)
			max_count_indicies      			= np.argwhere(counts == np.amax(counts))
			
			if max_count_indicies.shape[0] > 1:
				smaller_distance_array	= D[i,sorted_indices[l_indices[max_count_indicies]]]
				min_dist_indx		  	= np.argmin(smaller_distance_array)
				min_indx  				= sorted_indices[l_indices[max_count_indicies]][min_dist_indx]
				test_labels.append(train_labels[min_indx[0]])
			else:
				test_labels.append(unique_labels[np.argmax(counts)])

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


def svm_classify(train_image_feats, train_labels, test_image_feats, lambda_value=591.0):
	
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
	# print("lambda:", lambda_value)
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
		# predictions.append(np.expand_dims(svms[cat_].decision_function(test_image_feats), -1))

	W_mat = np.concatenate(W_mat, 0)
	b_vec = np.expand_dims(np.concatenate(b_vec, -1),-1)
	predictions = W_mat.dot(test_image_feats.T) + b_vec	
	predicted_indices = np.argmax(predictions, axis=0)	
	test_labels = [categories[index] for index in predicted_indices]	
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return test_labels


def cross_validation_vocab(categories, iters_per_param_value, candidate_vocab_sizes, candidate_lambda_values, train_percentage):

	num_train_per_cat_cross_val = 25
	data_path = osp.join('..', 'data')
	train_image_paths_cross_val, _, train_labels_cross_val, _ = get_image_paths(data_path, \
	                                                                  categories, num_train_per_cat_cross_val);

	num_training_images      =  len(train_image_paths_cross_val)
	num_training_split       =  int(train_percentage*num_training_images)

	np_paths_array           =  np.array(train_image_paths_cross_val) # for being able to randomly access indices
	np_labels_array          =  np.array(train_labels_cross_val)
	cat2idx                  =  {cat: idx for idx, cat in enumerate(categories)}

	mean_acc_list          = [] # for plotting error bars
	std_acc_list           = [] # for plotting error bars
	best_cand_lambdas_list = []

	for i in range(len(candidate_vocab_sizes)):
	    accuracy_list = [] # store accuracy values to compute mean and std acc over iterations 
	    lambdas_vote = np.zeros((len(candidate_lambda_values),1))
	    print("trying candidate vocab size:",candidate_vocab_sizes[i])
	    for j in range(iters_per_param_value):
	        
	        # Generate random indices for training images:
	        rnd_indxs = np.random.choice(num_training_images, num_training_images, replace=False)
	        
	        # Depending on train/validation percentage split, separate training and validation images:
	        train_image_paths_split       =  np_paths_array[rnd_indxs[:num_training_split]]
	        validation_image_paths_split  =  np_paths_array[rnd_indxs[num_training_split:]]
	        train_labels_split            =  np_labels_array[rnd_indxs[:num_training_split]]
	        validation_labels_split       =  np_labels_array[rnd_indxs[num_training_split:]]
	        
	        # Create vocabulary for current train/validation split:
	        vocab_filename = 'cross_validation_data/vocab_' + \
	                         str(candidate_vocab_sizes[i]) + '_iter_' + str(j+1) + '.pkl'
	        if not osp.isfile(vocab_filename):
	            vocab = build_vocabulary(train_image_paths_split, candidate_vocab_sizes[i])
	            with open(vocab_filename, 'wb') as f:
	                pickle.dump(vocab, f)

	        # Use the generated vocabulary to extract bags-of-sift features from train & validation images:
	        train_features        =  get_bags_of_sifts(train_image_paths_split, vocab_filename)
	        validation_features   =  get_bags_of_sifts(validation_image_paths_split, vocab_filename)  
	        
	        y_true                =  [cat2idx[cat] for cat in validation_labels_split]
	        
	        # for current vocabulary size test possible candidate lambdas:
	        testing_lambdas_list  =  []
	        for k in range(len(candidate_lambda_values)):
	            predicted_categories  =  svm_classify(train_features, train_labels_split, \
	                                                     validation_features, lambda_value=candidate_lambda_values[k])
	        
	            # Create a confusion matrix, compute accuracy and store in a list:
	            y_pred  =  [cat2idx[cat] for cat in predicted_categories]
	            cm      =  confusion_matrix(y_true, y_pred)
	            cm      =  cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
	            acc     =  np.mean(np.diag(cm))
	            testing_lambdas_list.append(acc)
	        best_lambda_indx = np.argmax(np.array(testing_lambdas_list))
	        lambdas_vote[best_lambda_indx,0] += 1
	        
	        accuracy_list.append(testing_lambdas_list[best_lambda_indx]) # Store accuracy corresponding to best lambda
	        print("Iteration:", j+1, "best acc:", accuracy_list[-1],\
	              "best lambda:", candidate_lambda_values[best_lambda_indx])
	        
	    acc_list_array = np.array(accuracy_list)    
	    mean_acc_list.append(np.mean(acc_list_array))
	    std_acc_list.append(np.std(acc_list_array))
	        
	    highest_voted_lambda_indx = np.argmax(lambdas_vote[:,0])
	    best_cand_lambdas_list.append(candidate_lambda_values[highest_voted_lambda_indx])
	    
	    print("Stats: mean acc = ", mean_acc_list[-1], "std acc = ", std_acc_list[-1], \
	          "highest voted lambda = ", best_cand_lambdas_list[-1])

	cross_val_means   = np.array(mean_acc_list)
	cross_val_stds    = np.array(std_acc_list)
	cross_val_lambdas = np.array(best_cand_lambdas_list) 
	np.savez('cross_validation_results', means=cross_val_means, stds=cross_val_stds, lambdas=cross_val_lambdas)	


def cross_validation_lambda(categories, candidate_lambda_values, train_labels, iters, train_split, train_image_feats):
	
	train_labels_array            =  np.array(train_labels)

	total_images                  =  train_image_feats.shape[0]
	num_train_images_split        =  int(train_split*total_images)
	cat2idx                       =  {cat: idx for idx, cat in enumerate(categories)}

	mean_acc_list          = [] # for plotting error bars
	std_acc_list           = [] # for plotting error bars

	for i in range(candidate_lambda_values.shape[0]):
	    temp_acc_list=[]
	    for j in range(iters):
	        rnd_indxs                     =  np.random.choice(total_images, total_images, replace=False)
	        train_image_feats_split       =  train_image_feats[rnd_indxs[:num_train_images_split], :]
	        validation_image_feats_split  =  train_image_feats[rnd_indxs[num_train_images_split:], :]
	        train_labels_split            =  train_labels_array[rnd_indxs[:num_train_images_split]]
	        validation_labels_split       =  train_labels_array[rnd_indxs[num_train_images_split:]]
	    
	        y_true                        =  [cat2idx[cat] for cat in validation_labels_split]
	        predicted_categories          =  svm_classify(train_image_feats_split, train_labels_split, \
	                                           validation_image_feats_split, \
	                                           lambda_value=candidate_lambda_values[i])
	        
	        # build confusion matrix and compute prediction accuracy on validation data:
	        y_pred  =  [cat2idx[cat] for cat in predicted_categories]
	        cm      =  confusion_matrix(y_true, y_pred)
	        cm      =  cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
	        acc     =  np.mean(np.diag(cm))
	        temp_acc_list.append(acc)
	        sys.stdout.write("trial:%d/%d, lambda:%f, iter:%d, acc:%f\r" % (i+1, candidate_lambda_values.shape[0], \
	                                                             candidate_lambda_values[i], j+1, temp_acc_list[-1]))
	        sys.stdout.flush()
	    mean_acc_list.append(np.mean(np.array(temp_acc_list)))
	    std_acc_list.append(np.std(np.array(temp_acc_list)))
	    
	cross_val_lambda_means = np.array(mean_acc_list)
	cross_val_lambda_stds  = np.array(std_acc_list)
	np.savez('cross_validation_lambda_results', means=cross_val_lambda_means, stds=cross_val_lambda_stds)


def plot_cross_validation_results_lambda(candidate_lambda_values):

	cross_val_results_data = np.load('cross_validation_lambda_results.npz')

	X    = candidate_lambda_values
	Y    = cross_val_results_data['means']
	Yerr = cross_val_results_data['stds']

	print("Highest mean acc:",Y[np.argmax(Y)],"for lambda:", X[np.argmax(Y)])

	plt.figure()
	plt.errorbar(X,Y,Yerr)
	plt.xlabel('lambda')
	plt.ylabel('accuracy')

def plot_cross_validation_results_vocab_size():

	cross_val_results_data = np.load('cross_validation_results.npz')

	X    = np.array([10, 20, 50, 100, 150, 200, 400, 600, 1000, 5000, 10000])
	Y    = cross_val_results_data['means']
	Yerr = cross_val_results_data['stds']

	print("Highest accuracy:", np.amax(Y), "for vocabulary size of ", X[np.argmax(Y)])

	f, (ax, ax2, ax3) = plt.subplots(1, 3, sharey=True)
	ax.errorbar(x=X, y=Y, yerr=Yerr, fmt='--o')
	ax2.errorbar(x=X, y=Y, yerr=Yerr, fmt='--o')
	ax3.errorbar(x=X, y=Y, yerr=Yerr, fmt='--o')
	ax.set_xlim(0,450)
	ax2.set_xlim(500,1100)
	ax3.set_xlim(2500,10500)
	ax.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax3.spines['left'].set_visible(False)
	d = 0.015
	kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
	ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # top-right 
	ax.plot((1 - d, 1 + d), (-d, +d), **kwargs) # bottom-right

	kwargs.update(transform=ax2.transAxes)  
	ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) # top-left
	ax2.plot((-d, +d), (-d, +d), **kwargs) # bottom-left
	ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # top-right 
	ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs) # bottom-right

	kwargs.update(transform=ax3.transAxes)  
	ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs) # top-left
	ax3.plot((-d, +d), (-d, +d), **kwargs) # bottom-left

	ax2.tick_params(left='off')
	ax3.tick_params(left='off')
	ax2.set_xlabel('vocabulary size')
	ax.set_ylabel('accuracy')
	ax2.set_title('Cross-validation results')
	plt.show()	

def monte_carlo_testing(categories, vocab_filename, num_trials, num_train_per_cat, best_lambda_value):

	cat2idx        =  {cat: idx for idx, cat in enumerate(categories)}
	accuracy_list  = []
	data_path = osp.join('..', 'data')

	for i in range(num_trials):
	    train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path,
	                                                                                 categories,
	                                                                                 num_train_per_cat);
	    train_image_feats       =  get_bags_of_sifts(train_image_paths, vocab_filename)
	    test_image_feats        =  get_bags_of_sifts(test_image_paths, vocab_filename)
	    predicted_categories    =  svm_classify(train_image_feats, train_labels, test_image_feats, lambda_value=best_lambda_value)
	    
	    y_true  =  [cat2idx[cat] for cat in test_labels]
	    y_pred  =  [cat2idx[cat] for cat in predicted_categories]
	    cm      =  confusion_matrix(y_true, y_pred)
	    cm      =  cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
	    acc     =  np.mean(np.diag(cm))    
	    accuracy_list.append(acc)
	    sys.stdout.write("trial:%d/%d, acc:%f\r" % (i+1, num_trials, acc))
	    sys.stdout.flush()

	print("Statistics: Mean=", np.mean(np.array(accuracy_list)), " Std:", np.std(np.array(accuracy_list)))