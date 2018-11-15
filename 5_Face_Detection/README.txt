
### Added Global Variables ###

I have the following global variables (which I refer to in the report by their respective names) which are used in the functions in student_code.py:

return_all 			= False # Returns all of the extracted negative HoG features instead of num_samples randomly sampled HoG features 
step_size 			= 15    # sliding window step / stride for extracting negative HoG features and for hard negative mining
detect_step_size 	= 10    # sliding window step / stride for running detector on test images
pos_ws				= 1.0   # sample_weights for positive HoG features for training SVM
topk_value			= 50    # top k detections to consider for NMS after detection

scales              = [1.0] # down-scaling values for extracting negative HoG features and hard negative mining

# down-scaling values for running detectors on test images
detection_scales    = [1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]


### Modified function signatures with default values ###

(1.) mine_hard_negs(non_face_scn_path, svm, feature_params, conf_thres=-1.0):

(2.) run_detector(test_scn_path, svm, feature_params, verbose=False, conf_thres=-1.0):

In both the functions above, I added the conf_thres argument to test different values of confidence threshold against the predicted confidence by SVM


### extra credit ###

1.) Run 'source activate cs6476p5'
2.) Install tensorflow by running 'conda install -c conda-forge tensorflow' 

example code to train the neural network:

batch_size        = 128
feature_dim       = features_pos.shape[1]
fnn_layer_size    = 8
max_gradient_norm = 1.0
learning_rate     = 1e-4
training_epochs   = 500

face_nn           = sc.NN_classifier( batch_size, feature_dim, fnn_layer_size, \
                                      max_gradient_norm, learning_rate, training_epochs)
 
input_data  = np.concatenate((features_pos, features_neg), axis=0)
target_data = np.concatenate((np.ones((features_pos.shape[0],1)),\
                              -1.0 * np.ones((features_neg.shape[0],1))), axis=0)

face_nn.train_model(input_data, target_data)

3.) I have implemented a function that will perform sliding window detection using the above trained neural network. The function signature is similar to the original run_detector()

run_detector_nn(test_scn_path, nn, feature_params, verbose=False, conf_thres=-1.0)

The only difference is that an instance of the NN_classifier class has to be passed to this function. 