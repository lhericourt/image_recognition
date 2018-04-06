# -*- coding: utf-8 -*-
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import tensorflow as tf
import json
from skimage.io import imsave, imread
import pickle

from conf import config
from my_utils.image_processing import copy_test_set, crop_img

# Set up python logging format
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

# Declare parameters
PATH_PERSON_DETECTION_MODEL = config.PATH_PERSON_DETECTION_MODEL
PATH_JSON_TRAIN = config.PATH_JSON_TRAIN
PATH_JSON_TEST = config.PATH_JSON_TEST
PATH_NEW_JSON_TRAIN = config.PATH_NEW_JSON_TRAIN
PATH_NEW_JSON_TEST = config.PATH_NEW_JSON_TEST
DIR_IMG = config.DIR_IMG
NEW_DIR_IMG = config.NEW_DIR_IMG
DIR_DATASET_NDARRAY = config.DIR_DATASET_NDARRAY
PERSON_LABEL = 1


def save_new_image(img_cropped, index, img_name, new_dir_img):
    """Save the new image corresponding to one person of the original image into a new directory
    
    :param img_cropped: image of the person as a numpy array
    :param index: number of the identified person in the original image
    :param img_name: name of the image as we can find it in the dataset json file (e.g. male/IEUGgdg-z.jpg)
    :param new_dir_img: Path of the directory where to save the image of the person
    :return: the name of the image of the person (e.g. male/IEUGgdg-z_1.jpg)
    """
    img_name_no_extension = img_name.split(".")[0]
    img_new_name = img_name_no_extension + "_" + str(index) + "." + img_name.split(".")[1]
    img_new_path = new_dir_img + img_new_name
    imsave(img_new_path, img_cropped)

    return img_new_name


def crop_person(list_img_name, dir_img, new_dir_img, path_new_json):
    """Create new images from a list of image paths. For each image the function will create 
    as many images as person present on the original image. Each new image will contain only the person detected.
    If no person has been detected on the image the function will just copy the image.
    
    :param list_img_name: List of names of images
    :param dir_img: Path of the directory where all images are present (male and female)
    :param new_dir_img: Path of the directory where all new images are saved (male and female)
    :param path_new_json: Path of the new json that references all new images
    :return: None, only save images and the new json
    """
    new_json = {
        "female": [],
        "male": []
    }

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_PERSON_DETECTION_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for j, image_name in enumerate(list_img_name):
                if j % 100 == 0:
                    logging.info("{} images have been processed".format(j))
                image_np = imread(dir_img + image_name)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                nb_boxes = len(np.where(scores > 0.9)[1])
                idx_person = np.where(classes[0][:nb_boxes] == PERSON_LABEL)[0]

                if len(idx_person) > 0:
                    for i, idx in enumerate(idx_person):
                        img_cropped = crop_img(image_np, boxes[0][idx])
                        img_new_name = save_new_image(img_cropped, i, image_name, new_dir_img)
                        new_json[image_name.split("/")[0]].append(img_new_name)
                else:
                    imsave(new_dir_img + image_name, image_np)
                    new_json[image_name.split("/")[0]].append(image_name)

    # Save the references of the new images into a file
    with open(path_new_json, 'w') as outfile:
        json.dump(new_json, outfile)


if __name__ == '__main__':
    # We load the references of images that cannot be read
    with open(DIR_DATASET_NDARRAY + 'images_ko.pkl', 'rb') as f:
        img_ko = pickle.load(f)

    # We crop all train data and save them into a new directory (same structure as the old one)
    train_json = json.load(open(PATH_JSON_TRAIN))
    images_to_crop = [x for x in train_json["female"] + train_json["male"] if x not in img_ko]
    crop_person(images_to_crop, DIR_IMG, NEW_DIR_IMG, PATH_NEW_JSON_TRAIN)

    # We copy all the test set into the same new directory
    copy_test_set(PATH_JSON_TEST, PATH_NEW_JSON_TEST, DIR_IMG, NEW_DIR_IMG)
