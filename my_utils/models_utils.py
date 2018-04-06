# -*- coding: utf-8 -*-

import sys
import os
import logging
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import tensorflow.contrib.keras as ker
from tensorflow.contrib.keras.python.keras import backend as K
from sklearn.metrics import precision_recall_fscore_support

from conf import config

# Set up python logging format
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

# Declare parameters
DIR_DETAILED_RESULTS = config.DIR_DETAILED_RESULTS


def set_last_layers_trainable(model, train_layer):
    """Method which sets all layers after the one defines as trainable. The ones before being not trainable.
    
    :param model: model with its layers
    :param train_layer: layer number from which we want to train them
    :return: the updated model
    """
    if train_layer == 0:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:train_layer]:
            layer.trainable = False
        for layer in model.layers[train_layer:]:
            layer.trainable = True
    return model


class DecayLR(ker.callbacks.Callback):
    """Class to define a learning rate which decreases in time"""

    def __init__(self, decay):
        """Constructor
        
        :param decay: decay between [0, 1] used to reduce the learning rate 
        """
        super(DecayLR, self).__init__()
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        """Processes called at the beginning of each epoch during the training of the model
        
        :param epoch: epoch number
        :param logs: dictionary of logs
        """
        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch % 3 == 0 and epoch > 0:
            new_lr = self.decay * old_lr
            K.set_value(self.model.optimizer.lr, new_lr)
            logging.info("New value for the learning rate : {}".format(K.get_value(self.model.optimizer.lr)))
        else:
            K.set_value(self.model.optimizer.lr, old_lr)


class MetricsDetailedScore(ker.callbacks.Callback):
    """Class to save metrics in a pickle file (precision, recall, f1score and support)"""

    def __init__(self, test_name):
        """Constructor

        :param test_name: name of the training test used to save the file with all the metrics 
        """
        super(MetricsDetailedScore, self).__init__()
        self.test_name = test_name

    def on_train_end(self, logs={}):
        """Processes called at the end of the training of the model"""
        # Compute predictions
        predict = self.model.predict(self.validation_data[0])
        best_idx = np.argmax(predict, axis=1).reshape(len(predict), 1)
        predict_dumm = np.concatenate([1 - best_idx, best_idx], axis=1)

        # Compute and save the metrics using the ground truth and the predictions
        all_indicators = precision_recall_fscore_support(self.validation_data[1], predict_dumm)
        with open(DIR_DETAILED_RESULTS + self.test_name + '.pkl', 'wb') as f:
            pickle.dump(all_indicators, f)
