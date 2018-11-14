
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