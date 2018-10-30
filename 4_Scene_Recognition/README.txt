
### Running code for final evaluation ###

In order to run the final evaluation of Linear SVM + bag-of-SIFT feature representation, you need to run the following cells in the listed order:

1.) Setup cell
2.) Cell in Section 2a : this should use the vocab.pkl file corresponding to vocab_size of 400 which is tuned value from my cross-validation experiment. This will also extract bag-of-SIFT features for training and testing images. With my current values of step size for vlfeat.sift.dsift(), it takes about 86 seconds to construct bag-of-SIFT features. 
3.) Cell in Section 3a : this will use the extracted features from the previous step and train SVMs for each category.
4.) Cell in Section 3b : builds and plots confusion matrix and reports accuracy percentage. 

I listed these steps so that you avoid running the cross-validation cells in between as they are really time consuming. I have saved the cross-validation results in .npz files which I load to plot the errorbar plots in my report. 


### Changed function signatures with some arguments added with default values ###

1.) get_tiny_images(image_paths, standardize_pixels = False, unit_norm = False)
	
	Added arguments:-
	standardize_pixels : Applies transform by subtracting mean and scaling by standard deviation
	unit_norm 		   : Applies transform by subtracting mean and scaling by norm of zero-centered vector

2.) nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, metric='euclidean', perform_kNN = False, k = 1)

	Added arguments:-
	perform_kNN : Boolean to use k nearest neighbors classifier vs 1 nearest neighbor classifier
	k 		    : hyper-parameter for k nearest neighbors classifier

3.) svm_classify(train_image_feats, train_labels, test_image_feats, lambda_value = 591.0)

	Added argument:-
	lambda_value : penalty parameter for training linear SVMs


### Running code cross-validation ###

This function needs the following import :
from sklearn.metrics import confusion_matrix

I did not read the instruction regarding TAs not using our jupyter notebook while grading. I had implemented cross-validation code in the jupyter notebook itself. But while preparing my submission, I realized this and shifted all of the code for cross-validation to the following 2 functions:

1.) cross_validation_vocab(categories, iters_per_param_value, candidate_vocab_sizes, candidate_lambda_values, train_percentage)
	
	Args:-
	categories 				: list of scene categories as strings (as in starter code)
	iters_per_param_value	: Number of iterations of randomly sampling training and validation data per candidate value of vocabulary size
	candidate_vocab_sizes	: list of candidate vocabulary sizes 
	candidate_lambda_values	: list of coarse lambda values (see report for details)
	train_percentage		: percentage of randomly sampled indices to use to training. Validation percentage = 1.0 - train_percentage

	example use of function:- 
	
	iters_per_param_value     = 10
	candidate_vocab_sizes     = [10, 20, 50, 100, 150, 200, 400, 600, 1000, 5000, 10000]
	candidate_lambda_values   = [0.1, 1, 10, 100, 200, 500, 750, 900, 1000, 1200, 1500, 2000]
	train_percentage          = 0.6 # validation_percentage = 1.0 - train_percentage
	
	sc.cross_validation_vocab(categories, iters_per_param_value, candidate_vocab_sizes, candidate_lambda_values, train_percentage)	


2.) cross_validation_lambda(categories, candidate_lambda_values, train_labels, iters, train_split, train_image_feats)

	Args:-
	categories 				: list of scene categories as strings (as in starter code)
	candidate_lambda_values	: numpy array of candidate lambda values 
	train_labels 			: N element list, where each entry is a string indicating the ground truth category for each training image
	iters 					: Number of iterations of randomly sampling training and validation data per candidate value of lambda
	train_split				: percentage of randomly sampled indices to use to training. Validation percentage = 1.0 - train_percentage
	train_image_feats		: N x d matrix, where d is the dimensionality of the feature representation. d is the number of clusters or equivalently the number of entries in each image's
	      					  histogram (vocab_size). Run cell in Section 2a to obtain this.

	example use of function:-

	candidate_lambda_values  =  np.arange(1,1000,10)
	iters                    =  20
	train_split              =  0.7
	sc.cross_validation_lambda(categories, candidate_lambda_values, train_labels, iters, train_split, train_image_feats)

Functions to plot errorbar plots:

3.) plot_cross_validation_results_vocab_size()

4.) plot_cross_validation_results_lambda(candidate_lambda_values)

	Args:- 
	candidate_lambda_values	: numpy array of candidate lambda values (must be same array as used for function 2 above)

Function used for Monte Carlo testing with tuned hyper-parameters (Monte-Carlo because the shuffle(pth) results in different datasets everytime. So we need statistics of the chosen hyper-params on test data by sampling a number of datasets and evaluating performance on test data):

5.) monte_carlo_testing(categories, vocab_filename, num_trials, num_train_per_cat, best_lambda_value)
	
	Args:- 
	categories 				: list of scene categories as strings (as in starter code)
	vocab_filename			: pickle file of vocabulary constructed using tuned hyper-parameter vocabulary size from cross-validation
	num_trials				: Number of Monte Carlo trails of sampling 1500 training and testing images
	num_train_per_cat		: (from starter code) number of images per scene category
	best_lambda_value		: Value of hyper-parameter lambda tuned by cross-validation

	example use of funtion:-

	vocab_filename    = 'vocab.pkl'
	num_trials        = 20
	best_lambda_value = 591
	sc.monte_carlo_testing(categories, vocab_filename, num_trials, num_train_per_cat, best_lambda_value)	

I understand this is not the best way to implement a python function, but I was working under the assumption that our notebooks will be used while grading. These functions can be called from the notebook and they should perform cross-validation and save the results in .npz files that can be used to plot the errorbar plots in the html report.