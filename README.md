################################################################
# Prerequisites
To run the program you need to run it on Python 2 and have the following libraries installed :
- tensorflow
- keras
- openCV
- skimage

All scripts have to be run from the root directory of the project (/fashwell)

################################################################
# Code architecture

## A notebook
At the root of the project you can find a notebook used to study the original dataset (Dataset_analyse.ipynb)

## Package conf
It defines all the parameters used by the program

## Package dataset
It defines all the modules used to generate final dataset used for training a model (reclass images that are misclassified, crop persons in images, generate numpy arrays from images).
We also find in this package the directory where the final dataset used for training is supposed to be (directory data_for_training)

## Directory data_v3
It contains images and json files resulting of the different transformations made on the original images.

## Package models
It defines the different models used by the program :
- Transfer learning (VGG16)
- Person detection (mobileNet SSD on Coco dataset)
- face and gender detection (haar cascade and CNN)
- The global model that combines all these three models

This package contains also the module to train the transfer learning model

## Package my_utils
It defines some common utilities used by different modules of the programm

## Directory results
Directory where the different results of the models are saved

################################################################
# Running the model

## Change the dataset path in config file
No image is present in the project directory also to see the result of the model on the test set you need to change the path of the test set in the file: conf/config.py.

The two parameters to change are:
- NEW_DIR_IMG, with the path of directory of all images (e.g: /home/laurent/projects/fashwell/data_v3/img/)
- PATH_NEW_JSON_TEST, with the path of the json file containing the list of test images (e.g: /home/laurent/projects/fashwell/data_v3/meta/test.json)

## Run the model
- Go to the project root (inside fashwell directory)
- Run the command line: python models/global_model.py

## See the results
The results are saved in the directory results/detailed_results/.
- One csv file lists for all images the prediction
- One pickle file saves the metrics (precision, recall, f1score)


################################################################
# Generate dataset for training
If you want to generate the dataset from the initial images for training you need to follow theses steps:

## Change the configuration
Change the values of the following parameters in the file conf/config.py
- DIR_IMG: path of the directory with all images  (e.g: /home/laurent/projects/fashwell/data_v2/img/)
- PATH_JSON_TRAIN: path of the file that references training dataset (e.g: /home/laurent/projects/fashwell/data_v3/meta/train.json)
- PATH_JSON_TEST: path of the file that references test dataset (e.g: /home/laurent/projects/fashwell/data_v3/meta/test.json)

## Remove noise in labels
Some females are in the male directory and some males are in the female directory. To move the pictures to their correct folder run the script dataset/reclass_train_dataset.py

## Generate new dataset with persons cropped for the training
To remove the noise from the background in some pictures I have made a script to detect persons in a picture and to crop the images to keep only the persons.
To do so run the script dataset/crop_person.py
It will generate new images in the folder data/v3 (it will copy the images from the test set, and generate new images for the training with the cropped persons

## Generate images as numpy array
The last step is to generate the dataset images and labels as numpy arrays. Run the script dataset/generate_dataset.py to do so.
It will generate the numpy files in the directory dataset/data_for_training/

################################################################
# Train the pre-trained VGG16 model
You first need to generate the dataset as explained in the previous paragraph.
Then if you wish you can modify some parameters (activation function, number of neurones in the fc layer..).
To do so open the module models/train_model.py and modify the values of the dictionary named "param_values" at the top of the main.
Finally you can run this script (models/train_model.py).
The results are saved in the folder results/logs_tb and can be visualized with tensorboard.