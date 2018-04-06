# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from tensorflow.contrib.keras.python.keras import backend as K


def precision_male(y_true, y_pred):
    """ Compute precision for the class "male"
    
    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: precision (float)
    """
    nb_male_pred = K.sum(K.round(K.clip(y_pred[:, 0], 0, 1)))
    male_true_positives = K.sum(K.round(K.clip(y_true[:, 0] * y_pred[:, 0], 0, 1)))
    precision = male_true_positives / (nb_male_pred + K.epsilon())
    return precision


def precision_female(y_true, y_pred):
    """ Compute precision for the class "female"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: precision (float)
    """
    nb_female_pred = K.sum(K.round(K.clip(y_pred[:, 1], 0, 1)))
    male_true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    precision = male_true_positives / (nb_female_pred + K.epsilon())
    return precision


def precision(y_true, y_pred):
    """ Compute global precision"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: the mean of female precision and male precision
    """
    return (precision_male(y_true, y_pred) + precision_female(y_true, y_pred)) / 2


def recall_male(y_true, y_pred):
    """ Compute recall for the class "male"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: recall (float)
    """
    nb_male = K.sum(K.round(K.clip(y_true[:, 0], 0, 1)))
    male_true_positives = K.sum(K.round(K.clip(y_true[:, 0] * y_pred[:, 0], 0, 1)))
    recall = male_true_positives / (nb_male + K.epsilon())
    return recall


def recall_female(y_true, y_pred):
    """ Compute recall for the class "female"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: recall (float)
    """
    nb_female = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
    male_true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    recall = male_true_positives / (nb_female + K.epsilon())
    return recall


def recall(y_true, y_pred):
    """ Compute global recall"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: the mean of female recall and male recall
    """
    return (recall_male(y_true, y_pred) + recall_female(y_true, y_pred)) / 2


def f1_score_male(y_true, y_pred):
    """ Compute f1 score for the class "male"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: f1 score (float)
    """
    return 2 * ((precision_male(y_true, y_pred) * recall_male(y_true, y_pred)) /
                (precision_male(y_true, y_pred) + recall_male(y_true, y_pred)))


def f1_score_female(y_true, y_pred):
    """ Compute f1 score for the class "male"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: f1 score (float)
    """
    return 2 * ((precision_female(y_true, y_pred) * recall_female(y_true, y_pred)) /
                (precision_female(y_true, y_pred) + recall_female(y_true, y_pred)))


def f1_score(y_true, y_pred):
    """ Compute global f1 score"

    :param y_true: true labels (dummy numpy array, column 0 for male, column 1 for female)
    :param y_pred: predicted labels (dummy numpy array, column 0 for male, column 1 for female)
    :return: the mean of female f1 score and male f1 score
    """
    return (f1_score_male(y_true, y_pred) + f1_score_female(y_true, y_pred)) / 2
