# -*- coding: utf-8 -*-

import sys
import os
from shutil import copyfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pickle
import json

from conf import config

# Declare parameters
PATH_JSON_TRAIN = config.PATH_JSON_TRAIN
IMG_WITH_WRONG_LABEL = config.IMG_WITH_WRONG_LABEL
DIR_IMG = config.DIR_IMG


def update_train_json_with_right_labels(path_json_train, img_wrong_labels):
    """Update the json used for the training set to remove the noise : reclassing correctly the male pictures
    that were identified as females, and the female pictures that where identified as males.
    
    :param path_json_train: Path to the json containing the train dataset
    :param img_wrong_labels: Path to the list of image names that where misclassified
    :return: None, only update the json file
    """

    # Load data
    with open(img_wrong_labels, 'rb') as outfile:
        img_ko = pickle.load(outfile)
    train_json = json.load(open(path_json_train))

    # Update female and male list of pictures
    new_list_img_female = [x for x in train_json["female"] if x not in img_ko["female"]] \
                          + [x.replace("male", "female") for x in img_ko["male"]]
    new_list_img_male = [x for x in train_json["male"] if x not in img_ko["male"]] \
                        + [x.replace("female", "male") for x in img_ko["female"]]

    # Save the new json
    new_train_json = {
        "female": new_list_img_female,
        "male": new_list_img_male
    }
    with open(path_json_train, 'w') as outfile:
        json.dump(new_train_json, outfile)


def move_img_with_wrong_labels(img_wrong_label, dir_img):
    """Move the images that where misclassified to their correct folder
    
    :param img_wrong_label: Path to the list of image names that where misclassified
    :param dir_img: Directory where all male and female images are.
    :return: 
    """
    # Load misclassified images
    with open(img_wrong_label, 'rb') as outfile:
        img_ko = pickle.load(outfile)

    # Move misclassified male pictures from female directory to male directory
    for img_name in img_ko["female"]:
        copyfile(dir_img + img_name, dir_img + "male/" + img_name.split("/")[-1])
        os.remove(dir_img + img_name)

    # Move misclassified female pictures from male directory to female directory
    for img_name in img_ko["male"]:
        copyfile(dir_img + img_name, dir_img + "female/" + img_name.split("/")[-1])
        os.remove(dir_img + img_name)


if __name__ == '__main__':
    update_train_json_with_right_labels(PATH_JSON_TRAIN, IMG_WITH_WRONG_LABEL)
    move_img_with_wrong_labels(IMG_WITH_WRONG_LABEL, DIR_IMG)
