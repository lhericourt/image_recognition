# -*- coding: utf-8 -*-

import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tensorflow.contrib.keras as ker
import numpy as np

from conf import config
from models.VGG16 import VGG16_local
from models.performance_indicator import *
from my_utils.models_utils import set_last_layers_trainable, DecayLR, MetricsDetailedScore

# Set up python logging format
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

# Declare parameters
DIR_LOG_TB = config.DIR_LOG_TB
DIR_GEN_MODELS = config.DIR_GEN_MODELS
DIR_DATASET_NDARRAY = config.DIR_DATASET_NDARRAY


class TFModel(object):
    """Class to train a model using transfer learning"""

    def __init__(self, tf_model_name, input_size, n_output):
        """Constructor that loads the convolution part of the pretrained model corresponding to the model name
        
        :param tf_model_name: transfer learning model name (only "vgg16" is available)
        :param input_size: size of the input image (integer that corresponds to the length and the width)
        :param n_output: number of classes to predict
        """
        self.tf_model_name = tf_model_name
        self.input_size = input_size
        self.n_output = n_output
        self.conv_layers = self.get_conv_net(self.tf_model_name)
        self.model_to_train = None

    def get_conv_net(self, name):
        """Return the pretrained convolution network asked for

        :param name: name of the pre-trained network
        :return: the convolution part of the pre-trained networks
        """
        if name == "vgg16":
            return VGG16_local.VGG16(weights='imagenet', include_top=False,
                                     input_shape=(self.input_size, self.input_size, 3))
        else:
            logging.error("This model name is not understood : {}".format(self.tf_model_name))
            return None

    def generate_model_with_n_FC_layers(self, n_conv_layer_trainable, dropout, n_layers, n_hidden_unit, activation):
        """Generate a model from a CNN model with fully connected layers on the top
    
        :param n_conv_layer_trainable: number of the first layer from the convolutional 
        layers that we want to set trainable
        :param dropout: dropout ratio (0 means no dropout)
        :param n_layers: number of layers we want to set in the fully connected part
        :param n_hidden_unit: number of hidden unit we want to set for each layer of the fully connected part
        :param activation: activation function to use (relu, tanh,...)
        :return: the model built with all thes parameters
        """

        if self.conv_layers is None:
            logging.error("No convolution network to use")
            return
        # Load first part of the CNN and add batch normalization
        self.conv_layers = set_last_layers_trainable(self.conv_layers, train_layer=n_conv_layer_trainable)
        self.model_to_train = ker.models.Sequential()
        self.model_to_train.add(ker.layers.BatchNormalization(input_shape=(self.input_size, self.input_size, 3)))
        self.model_to_train.add(self.conv_layers)
        self.model_to_train.add(ker.layers.Flatten())
        self.model_to_train.add(ker.layers.BatchNormalization())

        # Add fully connected part with dropout
        for i in range(n_layers):
            if i > 0:
                self.model_to_train.add(ker.layers.Dropout(dropout))
            self.model_to_train.add(ker.layers.Dense(n_hidden_unit, activation=activation))
        self.model_to_train.add(ker.layers.Dropout(dropout))
        self.model_to_train.add(ker.layers.Dense(self.n_output, activation='softmax'))

    def train_model(self, X_train, X_test, y_train, y_test, test_name, nb_epochs):
        """Method to train the model, the scores (precision, recall, f1 score) are saved in the
        tensorboard format and in a pickle object. The models are also saved at each end of each epoch.
        
        :param X_train: Images used for training as numpy array
        :param X_test: Images used for testing as numpy array
        :param y_train: labels corresponding to the training dataset as numpy array
        :param y_test: labels corresponding to the test dataset as numpy array
        :param test_name: name of the test that we want to run (used for saving the result)
        :param nb_epochs: number of epoch to train the model
        :return: It saves the models and the metrics (tensorboard and pickle file)
        """
        if self.model_to_train is None:
            logging.error("No model to train")
            return

        # Definition of all the tools used for training : learning rate decay, metrics, logs
        lr_decay = DecayLR(0.2)
        metrics = MetricsDetailedScore(test_name)
        model_ckp = ker.callbacks.ModelCheckpoint(DIR_GEN_MODELS + test_name + ".{epoch:02d}-{val_loss:.2f}.hdf5")
        logs_tb = ker.callbacks.TensorBoard(log_dir=DIR_LOG_TB + test_name)
        callbacks = [lr_decay, metrics, model_ckp, logs_tb]

        self.model_to_train.compile(loss="categorical_crossentropy", optimizer=ker.optimizers.RMSprop(lr=1e-4),
                                    metrics=['accuracy', precision_male, recall_male, f1_score_male,
                                             precision_female, recall_female, f1_score_female,
                                             precision, recall, f1_score])

        self.model_to_train.fit(X_train, y_train, batch_size=100, epochs=nb_epochs,
                                validation_data=(X_test, y_test), class_weight={0: 0.8, 1: 0.2}, callbacks=callbacks)


class Dataset(object):
    """ Object to get the dataset to train the model"""

    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path):
        """Constructor
        
        :param X_train_path: Path to the numpy array containing all training images
        :param X_test_path: Path to the numpy array containing all test images
        :param y_train_path: Path to the numpy array containing all training labels
        :param y_test_path: Path to the numpy array containing all test labels
        """
        self.X_train_path = X_train_path
        self.X_test_path = X_test_path
        self.y_train_path = y_train_path
        self.y_test_path = y_test_path

    def get_data_set_to_train_model(self):
        """Method to get training and test dataset
        
        :return: images (test & train) as numpy array, and labels (train & test) as dummy numpy array
        """
        X_train = np.load(self.X_train_path)
        X_test = np.load(self.X_test_path)
        y_train = np.load(self.y_train_path)
        y_train_dumm = np.concatenate([1 - y_train, y_train], axis=1)
        y_test = np.load(self.y_test_path)
        y_test_dumm = np.concatenate([1 - y_test, y_test], axis=1)

        return X_train, X_test, y_train_dumm, y_test_dumm


if __name__ == '__main__':
    # Defining parameters used by the model
    param_values = {
        "test_name": "vgg_img_crop_5",
        "nb_epoch": 15,
        "activation": "relu",
        "dropout": 0.5,
        "n_conv_layer_trainable": 15,
        "n_layers": 1,
        "n_hidden_unit": 512
    }

    # Loading train and test dataset
    ds = Dataset(DIR_DATASET_NDARRAY + "X_train_crop.npy", DIR_DATASET_NDARRAY + "X_test_crop.npy",
                 DIR_DATASET_NDARRAY + "y_train_crop.npy", DIR_DATASET_NDARRAY + "y_test_crop.npy")
    X_train, X_test, y_train_dumm, y_test_dumm = ds.get_data_set_to_train_model()

    # Loading the model
    vgg_model = TFModel("vgg16", input_size=128, n_output=2)
    vgg_model.generate_model_with_n_FC_layers(param_values["n_conv_layer_trainable"], param_values["dropout"],
                                              param_values["n_layers"], param_values["n_hidden_unit"],
                                              param_values["activation"])
    # Training the model
    vgg_model.train_model(X_train, X_test, y_train_dumm, y_test_dumm,
                          param_values["test_name"], param_values["nb_epoch"])
