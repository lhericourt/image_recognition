# -*- coding: utf-8 -*-

######################## Dataset management ########################
# Directory path with all images (male and female)
DIR_IMG = "data_v2/img/"

# Directory path with all images (male and female)
NEW_DIR_IMG = "data_v3/img/"

# Path to the json file with references of train images
PATH_JSON_TRAIN = "data_v2/meta/train.json"

# Path to the json file with references of test images
PATH_JSON_TEST = "data_v2/meta/test.json"

# Path to the list of images that are incorrectly labeled
IMG_WITH_WRONG_LABEL = "dataset/data_for_training/misclassified_data.pkl"

# Path to the json file with references of train images that have been processed to crop the people
PATH_NEW_JSON_TRAIN = "data_v3/meta/train.json"

# Path to the json file with references of train images that have been processed to crop the people
PATH_NEW_JSON_TEST = "data_v3/meta/test.json"

# Directory path with dataset as numpy array (images and labels)
DIR_DATASET_NDARRAY = "dataset/data_for_training/"

# Path to the file that list all images that are not taking into account because of their format
IMG_KO = "dataset/data_for_training/images_ko.pkl"

######################## Models management ########################
# Directory with all generated models
DIR_GEN_MODELS = "models/generated_models/"

# Path to the model used to detect the full body of a person in an image
PATH_PERSON_DETECTION_MODEL = "models/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb"

# Path to the weight of a VGG16 model trained on ImageNet
PATH_VGG16_WEIGHTS = 'models/VGG16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Path to the detector of profile faces
PATH_PROFILEFACE  = "models/face_classification/haarcascade_profileface.xml"

# Path to the detector of frontal faces
PATH_FRONTALFACE = "models/face_classification/haarcascade_frontalface_alt.xml"

# Path to the model of faces
PATH_FACE_MODEL = "models/face_classification/simple_CNN.81-0.96.hdf5"

# Path to the VGG model trained on Fashwell images
PATH_TRAINED_VGG = "models/generated_models/vgg_crop_img_4.09-0.44.hdf5"

######################## Model training results ########################
# Directory with tensorboard logs
DIR_LOG_TB = "results/logs_tb/"

# Directory with detailed indicator (by class)
DIR_DETAILED_RESULTS = "results/detailed_results/"