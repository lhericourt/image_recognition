# -*- coding: utf-8 -*-
import sys
import os
import logging
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from conf import config
import cv2
from tensorflow.contrib.keras import models
from keras.models import load_model
import numpy as np
from skimage.io import imread
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from models.face_and_person_models import PersonDetection, FaceRecognition
from models.performance_indicator import *
from my_utils.image_processing import resize_list_of_img, remove_white_background

# Set up python logging format
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

# Declare parameters
NEW_DIR_IMG = config.NEW_DIR_IMG
PATH_NEW_JSON_TEST = config.PATH_NEW_JSON_TEST
DIR_DETAILED_RESULTS = config.DIR_DETAILED_RESULTS


class GlobalModel(object):
    """Complete model using face detection, person detection and VGG"""

    PATH_PERSON_DETECTION_MODEL = config.PATH_PERSON_DETECTION_MODEL
    PATH_FACE_MODEL = config.PATH_FACE_MODEL
    PATH_FRONTALFACE = config.PATH_FRONTALFACE
    PATH_PROFILEFACE = config.PATH_PROFILEFACE
    PATH_TRAINED_VGG = config.PATH_TRAINED_VGG
    INPUT_SIZE = 128

    def __init__(self):
        """Constructor that loads all used models"""
        self.fr = self.init_face_model()
        self.pr = PersonDetection(self.PATH_PERSON_DETECTION_MODEL)
        self.vgg = models.load_model(self.PATH_TRAINED_VGG, custom_objects={'precision_male': precision_male,
                                                                            "precision_female": precision_female,
                                                                            "precision": precision,
                                                                            'recall_male': recall_male,
                                                                            "recall_female": recall_female,
                                                                            "recall": recall,
                                                                            "f1_score_male": f1_score_male,
                                                                            "f1_score_female": f1_score_female,
                                                                            "f1_score": precision})

    def init_face_model(self):
        """Initialize the face recognition model
        
        :return: The instance of the face recognition model
        """
        profileface = cv2.CascadeClassifier(self.PATH_PROFILEFACE)
        face = cv2.CascadeClassifier(self.PATH_FRONTALFACE)
        gender_classifier = load_model(self.PATH_FACE_MODEL)
        fr = FaceRecognition(face, profileface, gender_classifier)
        return fr

    def predict_gender_by_face(self, list_img, list_img_name):
        """Try to find faces within the list of images and use them to find out the gender. Since we can have
        multiple faces on an image we define the gender as the one with the probability the highest.
        
        :param list_img: List of images as numpy array
        :param list_img_name: List of names of each images
        :return: List of tuple with the result (0: male, 1: female, -1: no class) and the associate image name
        """
        # Find the gender of the images
        pred_face = self.fr.compute_gender_label_from_list_img(list_img)

        # Keep only images that have been classified
        idx_img_classified = np.where(np.array(pred_face) != -1)[0]
        y_pred_face_classified = np.array(pred_face)[idx_img_classified]
        img_name_classified = [x for x in list_img_name if list_img_name.index(x) in idx_img_classified]
        result = zip(y_pred_face_classified, img_name_classified)

        return result

    def predict_gender_by_person(self, list_img, list_img_name):
        """Try to find persons within the images and use them to find out the gender. Since we can have
        multiple persons on an image we define the gender as the one with the probability the highest.
        
        :param list_img: List of images as numpy array
        :param list_img_name: List of names of each images
        :return: List of tuple with the result (0: male, 1: female, -1: no class) and the associate image name
        """
        # Find the gender of the images
        persons_in_all_img = self.pr.get_persons_from_list_img(list_img)
        y_pred_from_person = []
        for i, persons_in_one_img in enumerate(persons_in_all_img):
            best_pred = -1
            if len(persons_in_one_img) > 0:
                img_persons_resized = resize_list_of_img(persons_in_one_img, self.INPUT_SIZE)
                pred_by_person = self.vgg.predict(img_persons_resized)
                best_pred = np.unravel_index(pred_by_person.argmax(), pred_by_person.shape)[1]
            y_pred_from_person.append(best_pred)

        # Keep only images that have been classified
        idx_img_classified = np.where(np.array(y_pred_from_person) != -1)[0]
        y_pred_person = np.array(y_pred_from_person)[idx_img_classified]
        img_name_classified = [x for x in list_img_name if list_img_name.index(x) in idx_img_classified]
        result = zip(y_pred_person, img_name_classified)

        return result

    def predict_gender_by_whole_img(self, list_img, list_img_name):
        """Find the gender using the whole image (no face, no person).
        :param list_img: List of images as numpy array
        :param list_img_name: List of names of each images
        :return: List of tuple with the result (0: male, 1: female) and the associate image name
        """
        list_img_resized = resize_list_of_img(list_img, self.INPUT_SIZE)
        pred_whole_img = self.vgg.predict(list_img_resized)
        y_pred_whole_img = pred_whole_img.argmax(axis=1)
        return zip(y_pred_whole_img, list_img_name)

    def predict_gender_on_test_set(self, dir_img, list_img_name):
        """Predict the gender for all images of the test set. This method can take other data however they have 
        to follow this rule : the names of the images (2nd argument of the method) must correspond to images within the
        specified directory (1st argument)
        
        :param dir_img: Directory where the images are
        :param list_img_name: list of names of each images
        :return: List of tuple with the result (0: male, 1: female) and the associate image name
        """
        # Loading dataset
        list_img = []
        for i, img_path in enumerate(list_img_name):
            img = imread(dir_img + img_path)
            img = remove_white_background(img, 20)
            list_img.append(img)
        list_img = np.array(list_img)
        logging.info("The dataset has been loaded")

        # Try to find out the gender by face detection
        logging.info("Beginning of prediction by face detection")
        result_pred_by_face = self.predict_gender_by_face(list_img, list_img_name)

        # For images not yet classified try to find out gender by person detection
        logging.info("Beginning of prediction by person detection")

        img_classified = zip(*result_pred_by_face)[1]
        img_name_to_classify = [x for x in list_img_name if x not in img_classified]
        idx_img_to_classify = [x for x in range(len(list_img_name)) if list_img_name[x] in img_name_to_classify]
        result_pred_by_person = self.predict_gender_by_person(list_img[idx_img_to_classify], img_name_to_classify)

        # For images not yet classified find out gender using the whole image
        logging.info("Beginning of prediction by the whole image")
        img_classified += zip(*result_pred_by_person)[1]
        img_name_to_classify = [x for x in list_img_name if x not in img_classified]
        idx_img_to_classify = [x for x in range(len(list_img_name)) if list_img_name[x] in img_name_to_classify]
        result_pred_whole_img = self.predict_gender_by_whole_img(list_img[idx_img_to_classify], img_name_to_classify)

        return result_pred_by_face + result_pred_by_person + result_pred_whole_img


def save_results(prediction, test_name):
    """Save the result as a dataframe (each prediction for each image), and scores in a pickle file
    
    :param prediction: predictions as list af tuple (prediction, image name)
    :param test_name: name used to save the files
    :return: Nothing just save the results in files
    """
    # save predictions in a Dataframe
    prediction_df = pd.DataFrame(prediction, columns=["y_pred", "img_name"])
    prediction_df["y_true"] = prediction_df.apply(axis=1, func=lambda x: 0 if x["img_name"].startswith("male") else 1)
    prediction_df.to_csv(DIR_DETAILED_RESULTS + test_name + ".csv", index=False)

    # save score
    scores = precision_recall_fscore_support(prediction_df["y_true"], prediction_df["y_pred"])
    with open(DIR_DETAILED_RESULTS + test_name + ".pkl", "wb") as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(scores)


if __name__ == '__main__':
    # Load model
    gb = GlobalModel()

    # Load test set
    test_dataset = json.load(open(PATH_NEW_JSON_TEST))
    test_dataset_list = test_dataset['male'] + test_dataset["female"]

    # Get predictions
    predictions = gb.predict_gender_on_test_set(NEW_DIR_IMG, test_dataset_list)
    print(predictions)
    # save predictions in a Dataframe
    save_results(predictions, "result_lulu")
