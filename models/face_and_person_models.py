# -*- coding: utf-8 -*-

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import cv2
import tensorflow as tf

from my_utils.image_processing import crop_img, convert_img_in_float


class FaceRecognition(object):
    """Object to do face recognition on an image"""
    THRESHOLD = 0.85  # Below this threshold the classifier does not make a decision

    def __init__(self, face_detector, profile_face_detector, gender_classifier):
        """Constructor
        
        :param face_detector: the frontal face detector
        :param profile_face_detector: the profile face detector
        :param gender_classifier: the model to detect if the face is a male or a female
        """
        self.face_detector = face_detector
        self.profile_face_detector = profile_face_detector
        self.gender_classifier = gender_classifier
        self.gender_target_size = gender_classifier.input_shape[1:3]

    def get_faces_coordinates_from_img(self, rgb_img):
        """Get for all faces of the images their coordinates
        
        :param rgb_img: image as a numpy array in RGB
        :return: numpy array, one line per face, 4 columns (x_left, y_top, width, height)
        """
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
        profile_faces = self.profile_face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
        faces_coordinates = np.concatenate(
            [np.array(faces).reshape(-1, 4).astype(int), np.array(profile_faces).reshape(-1, 4).astype(int)])
        return faces_coordinates

    def get_faces_coordinates_from_list_img(self, list_img):
        """Get for all faces of a list of images their coordinates

        :param rgb_img: list of image as a numpy array in RGB
        :return: list of numpy array, each numpy array corresponding to an image : one line per face, 
        4 columns (x_left, y_top, width, height)
        """
        all_faces_coordinates = []
        for i, img in enumerate(list_img):
            if i % 100 == 0:
                logging.info("{} images have been processed by the faces detection".format(i))
            faces_coordinates = self.get_faces_coordinates_from_img(img)
            all_faces_coordinates.append(faces_coordinates)
        return all_faces_coordinates

    def get_faces(self, rgb_img, coordinates, output_size):
        """Get faces as images (numpy array) from an image and the coordinates of the faces
        
        :param rgb_img: image as a numpy array in RGB
        :param coordinates: numpy array, one line per face, 4 columns (x_left, y_top, width, height)
        :param output_size: faces as images in numpy array
        :return: 
        """
        faces = np.zeros((len(coordinates), output_size, output_size, 3))

        for i, (x, y, w, h) in enumerate(coordinates):
            # We take a marge to be sure to have the whole face
            rgb_face = rgb_img[max(0, y - int(0.5 * h)): min(y + int(1.5 * h), rgb_img.shape[0]),
                       max(0, x - int(0.5 * w)): min(x + int(1.5 * w), rgb_img.shape[1]), :]

            rgb_face = cv2.resize(rgb_face, (self.gender_target_size))
            rgb_face = convert_img_in_float(rgb_face)
            rgb_face = np.expand_dims(rgb_face, 0)
            faces[i] = rgb_face

        return faces

    def get_faces_from_list_img(self, list_img, list_coordinates, output_size):
        """Get all faces from a list of images and a list of coordinates

        :param list_img: list of images as a numpy array in RGB
        :param list_coordinates: list of numpy array : one line per face, 4 columns (x_left, y_top, width, height)
        :param output_size: list of faces as images, each numpy array corresponds to the n faces found in the image
        :return: 
        """
        list_faces = []

        for i, img in enumerate(list_img):
            if len(list_coordinates[i]) > 0:
                faces = self.get_faces(img, list_coordinates[i], output_size)
            else:
                faces = []
            list_faces.append(faces)

        return list_faces

    def compute_gender_label_from_img(self, rgb_img):
        """Get the gender of an image.
        
        It identifies all faces on the image, and for each one it predicts the gender. The method returns the 
        gender of the face that has the highest score. If the score is too low, the method does not classify the image.
        
        :param rgb_img: image as a numpy array in RGB
        :return: 0 for male, 1 for female, -1 for no gender
        """
        output_size = 48
        prediction = -1
        faces_coordinates = self.get_faces_coordinates_from_img(rgb_img)
        faces = self.get_faces(rgb_img, faces_coordinates, output_size)

        if len(faces > 0):
            genders = self.gender_classifier.predict(faces)
            idx_max = np.unravel_index(genders.argmax(), genders.shape)
            if genders[idx_max] > self.THRESHOLD:
                # label 0 for male, and 1 for female
                prediction = 1 - idx_max[1]

        return prediction

    def compute_gender_label_from_list_img(self, list_img):
        """Get the gender for a list of images.

        It identifies all faces on a list of images, and for each face it predicts the gender. For each image 
        the method returns the gender of the face that has the highest score. 
        If the score is too low, the method does not classify the image.

        :param list_img: list of image as a numpy array in RGB
        :return: list of prediction for each image (0 for male, 1 for female, -1 for no gender)
        """
        output_size = 48

        list_faces_coordinates = self.get_faces_coordinates_from_list_img(list_img)
        list_faces = self.get_faces_from_list_img(list_img, list_faces_coordinates, output_size)
        list_prediction = []

        for i, faces in enumerate(list_faces):
            if i % 100 == 0:
                logging.info("{} images have been processed by the gender detection".format(i))
            prediction = -1
            if len(faces) > 0:
                genders = self.gender_classifier.predict(faces)
                idx_max = np.unravel_index(genders.argmax(), genders.shape)
                if genders[idx_max] > self.THRESHOLD:
                    # label 0 for male, and 1 for female
                    prediction = 1 - idx_max[1]
            list_prediction.append(prediction)

        return list_prediction


class PersonDetection(object):
    """Object to detect people on an image"""
    PERSON_LABEL = 1  # It corresponds to the label of persons in the COCO dataset
    THRESHOLD = 0.9  # Below this threshold the classifier does not consider the zone as a person

    def __init__(self, model_path):
        """Constructor that automatically load the model
        
        :param model_path: Path of the model used to detect people on an image
        """
        self.model_path = model_path
        self.detection_graph = self.load_model()

    def load_model(self):
        """ Method to load the people detector model
        
        :return: the people detector model
        """
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def get_persons_from_list_img(self, list_img):
        """Detect in an image all the people.

        :param list_img: list of image as a numpy array in RGB
        :return: A list of numpy array, each numpy array corresponding to the persons present on the image
        """
        persons_of_all_img = []
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                for img in list_img:
                    persons_img = []
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: np.expand_dims(img, axis=0)})
                    nb_boxes = len(np.where(scores > self.THRESHOLD)[1])
                    idx_person = np.where(classes[0][:nb_boxes] == self.PERSON_LABEL)[0]

                    if len(idx_person) > 0:
                        for i, idx in enumerate(idx_person):
                            img_cropped = crop_img(img, boxes[0][idx])
                            persons_img.append(img_cropped)
                    persons_of_all_img.append(np.array(persons_img))

        return persons_of_all_img
