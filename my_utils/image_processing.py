# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import json
from shutil import copyfile
from skimage.transform import resize


def remove_white_background(img, threshold):
    """Crop images to remove useless white borders

    :param img: image as numpy array (RGB)
    :param threshold: level of white to remove, the lower the threshold is the whiter the background has to be 
    to be cropped
    :return: cropped image as numpy array
    """
    new_img = img - 255
    mask1 = np.abs(new_img).sum(axis=2) < threshold
    mask2 = np.abs(new_img).sum(axis=2) > (255 - threshold) * 3
    mask = mask1 + mask2
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    return img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


def copy_test_set(path_json_test, path_new_json_test, old_dir, new_dir):
    """Copy the test set (json and images) in a new directory.
    
    :param path_json_test: Path to the actual json test set
    :param path_new_json_test: Path to the new json test set
    :param old_dir: Path to the directory image all images (male and female)
    :param new_dir: Path to the new directory to copy the images
    :return: None, only copy files
    """

    copyfile(path_json_test, path_new_json_test)
    test_json = json.load(open(path_json_test))

    for img_name in test_json["male"] + test_json["female"]:
        copyfile(old_dir + img_name, new_dir + img_name)


def crop_img(img_np, coord_box):
    """Crop an image from coordinates
    
    :param img_np: image as a numpy array
    :param coord_box: list of coordinates normalized between 0 and 1 - [y_min, x_min, y_max, x_max]
    :return: the cropped image as a numpy array
    """
    ymin, xmin, ymax, xmax = coord_box[0], coord_box[1], coord_box[2], coord_box[3]
    im_width = img_np.shape[1]
    im_height = img_np.shape[0]
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                  int(ymin * im_height), int(ymax * im_height))
    return img_np[top:bottom, left:right, :]


def convert_img_in_float(img):
    """Normalized an image from range [0, 255] to [0, 1]
    
    :param img: image as a numpy array
    :return: normalized image as a numpy array
    """
    img = img.astype('float32')
    img = img / 255.0
    return img


def resize_list_of_img(list_img, size):
    """Resize a list of images
    
    :param list_img: list of images as numpy arrays
    :param size: new size (height and width)
    :return: resized images as numpy array
    """
    img_resized = np.zeros((len(list_img), size, size, 3))
    for i, whole_img in enumerate(list_img):
        img_resized[i] = resize(whole_img, (size, size))
    return img_resized
