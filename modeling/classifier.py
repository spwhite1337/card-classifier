import os
import cv2
import pickle
from typing import Tuple

from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, VGG16, InceptionV3

from sklearn.metrics import auc, roc_curve

from config import ROOT_DIR, logger


class MagicCardClassifier(object):

    # All colors to train on
    card_colors = ['B', 'G', 'N', 'R', 'U', 'W']

    # Curated dir
    curated_dir = os.path.join(ROOT_DIR, 'data', 'curated')

    # Models
    models = {
        'VGG': VGG16,
        'ResNet': ResNet50,
        'inception': InceptionV3,
    }

    def __init__(self,
                 # Processing
                 zoom_range: bool = True,
                 horizontal_flip: bool = True,
                 brightness_range: bool = True,
                 color_change: bool = True,
                 target_size: Tuple = (256, 256),

                 # Training
                 model_type: str = 'VGG',
                 batch_size: int = 32,
                 epochs: int = 50,

                 # I/O
                 results_dir: str = os.path.join(ROOT_DIR, 'modeling', 'results')
                 ):

        # Processing
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
        self.color_change = color_change
        self.target_size = target_size

        # Training
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs

        # I/O
        self.results_dir: str = results_dir
        self.models = None

    def process(self, color: str) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Set up Keras generators for augmented data
        """
        # processing function to change color scheme of input data
        def color_change(image):
            image = np.array(image)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            return Image.fromarray(hsv_image)

        # Set up training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1/255.,
            zoom_range=0.15 if self.zoom_range else None,
            horizontal_flip=True if self.horizontal_flip else None,
            brightness_range=[0.9, 1.1] if self.brightness_range else None,
            data_format='channels_last',
            preprocessing_function=color_change if self.color_change else None
        )
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(self.curated_dir, color, 'train'),
            target_size=self.target_size,
            color_mode='rgb',
            classes=['negative', 'positive'],
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=187
        )

        # Set up test data generator without augmentation
        test_datagen = ImageDataGenerator(
            rescale=1/255.
        )
        test_generator = test_datagen.flow_from_directory(
            directory=os.path.join(self.curated_dir, color, 'test'),
            target_size=self.target_size,
            color_mode='rgb',
            classes=['negative', 'positive'],
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=187
        )

        return train_generator, test_generator

    def train(self):
        """
        Train a model for each color class
        """
        for color in self.card_colors:
            train, test = self.process(color)

            # Get model
            model = self.models.get(self.model_type)
            if model is None:
                raise ValueError('Incorrect model selection.')

            step_size_train = train.n // train.batch_size
            step_size_test = test.n // test.batch_size
            model.fit_generator(generator=train,
                                steps_per_epoch=step_size_train,
                                validation_data=test,
                                validation_steps=step_size_test,
                                epochs=self.epochs
                                )



