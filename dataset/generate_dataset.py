# -*- coding: utf-8 -*-

import sys
import os
import logging
import pickle
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json
from skimage.io import imread
from skimage.transform import resize
import numpy as np

from conf import config
from my_utils.image_processing import remove_white_background

# Set up python logging format
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

# Declare parameters
DIR_IMG = config.DIR_IMG
NEW_DIR_IMG = config.NEW_DIR_IMG
PATH_JSON_TRAIN = config.PATH_JSON_TRAIN
PATH_JSON_TEST = config.PATH_JSON_TEST
PATH_NEW_JSON_TRAIN = config.PATH_NEW_JSON_TRAIN
PATH_NEW_JSON_TEST = config.PATH_NEW_JSON_TEST
DIR_DATASET_NDARRAY = config.DIR_DATASET_NDARRAY
IMG_KO = config.IMG_KO
SIZE = 128
DICT_LABEL = {
    "male": 0,
    "female": 1
}


class DetectImgKO(object):
    """Detect images that cannot be read"""

    @staticmethod
    def find_images_with_unusual_format(json_filepath):
        """Detect the images that cannot be read
        
        :param json_filepath: Path of the json file that contains path for male and female pictures
        :return: a list a image names that cannot be read
        """
        data = json.load(open(json_filepath))
        male_img_names = data["male"]
        female_img_names = data["female"]
        images_ko = []
        for img_name in male_img_names + female_img_names:
            img = imread(DIR_IMG + img_name)
            if len(img.shape) != 3:
                logging.info("Only 2 dimension image : {}".format(img_name))
                images_ko.append(img_name)
            elif img.shape[0] < 10:
                logging.info("Image to small : {}".format(img_name))
                images_ko.append(img_name)
        return images_ko

    def save_img_ko(self, output_dir):
        """ Save for the training and the test the list of images that cannot be read.
        
        :param output_dir: Path of the directory where to save the pickle object
        :return: None, only save the list of ko images into a pickle file
        """
        train_img_ko = self.find_images_with_unusual_format(PATH_JSON_TRAIN)
        test_img_ko = self.find_images_with_unusual_format(PATH_JSON_TEST)
        img_ko = train_img_ko + test_img_ko

        with open(output_dir + 'images_ko.pkl', 'wb') as f:
            pickle.dump(img_ko, f)


class DataSet(object):
    """Class to convert the images are numpy arrays that can be used to train a model"""

    def __init__(self, img_ko, dir_img):
        """Constructor
        
        :param img_ko: List of names of images that cannot be read 
        :param dir_img: Path of the directory that contains all images (male and female)
        """
        self.img_ko = img_ko
        self.dir_img = dir_img

    def generate_ndarray_dataset(self, json_filepath):
        """Use a json file containing dataset information to generate a numpy array containing 
        all the corresponding images, and all the corresponding labels
        
        :param json_filepath: Path of the json file that contains path for male and female pictures
        :return: a numpy array with all images, and a numpy array with all labels (1d array)
        """
        logging.info("Beginning of generating dataset for : {}".format(json_filepath))
        logging.info("Retrieving all images with right format")
        data = json.load(open(json_filepath))
        male_img_names = [x for x in data["male"] if x not in self.img_ko]
        female_img_names = [x for x in data["female"] if x not in self.img_ko]

        nb_img = len(male_img_names + female_img_names)
        nb_male_img = len(male_img_names)

        X = np.zeros((nb_img, SIZE, SIZE, 3))
        y = np.zeros((nb_img, 1))

        logging.info("Converting all images as numpy array")
        for i, img_name in enumerate(male_img_names + female_img_names):
            img = imread(self.dir_img + img_name)
            img = remove_white_background(img, 20)
            img = resize(img, (SIZE, SIZE))
            X[i] = np.expand_dims(img, axis=0)

        y[:nb_male_img] = DICT_LABEL["male"]
        y[nb_male_img:] = DICT_LABEL["female"]

        logging.info("Shuffling data")
        random.seed(42)
        random_index = random.sample(range(nb_img), nb_img)
        X_shuffled = X[random_index]
        y_shuffled = y[random_index]
        logging.info("End of generating dataset for {}".format(json_filepath))

        return X_shuffled, y_shuffled

    def save_dataset_as_ndarray(self, name, output_dir, path_json_train, path_json_test):
        """ Save the training set and the test set as numpy array
        
        :param name: name used as suffix of the files that are generated
        :param output_dir: Directory where to save the numpy arrays
        :param path_json_train: Path to the JSON that references training images
        :param path_json_test: Path to the JSON that references test images
        :return: None, only save the numpy arrays
        """
        logging.info("Beginning of processing train dataset")
        X_train, y_train = self.generate_ndarray_dataset(path_json_train)
        np.save(output_dir + "X_train_" + name, X_train)
        np.save(output_dir + "y_train_" + name, y_train)

        logging.info("Beginning of processing test dataset")
        X_test, y_test = self.generate_ndarray_dataset(path_json_test)
        np.save(output_dir + "X_test_" + name, X_test)
        np.save(output_dir + "y_test_" + name, y_test)


if __name__ == '__main__':
    # Detect images that cannot be read
    # di = DetectImgKO()
    # di.save_img_ko(DIR_DATASET_NDARRAY)

    # Generate dataset as numpy arrays
    with open(DIR_DATASET_NDARRAY + 'images_ko.pkl', 'rb') as f:
        img_ko = pickle.load(f)
    ds = DataSet(img_ko, NEW_DIR_IMG)
    ds.save_dataset_as_ndarray("crop", DIR_DATASET_NDARRAY, PATH_NEW_JSON_TRAIN, PATH_NEW_JSON_TEST)
