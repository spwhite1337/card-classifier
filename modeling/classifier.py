import os
import cv2
import time
import pickle
from typing import Tuple

from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16

from sklearn.metrics import auc, roc_curve

from config import ROOT_DIR, logger


class MagicCardClassifier(object):

    # All colors to train on
    card_colors = ['B', 'G', 'N', 'R', 'U', 'W']

    # Curated dir
    curated_dir = os.path.join(ROOT_DIR, 'data', 'curated')

    # Models
    model_options = {
        'VGG': VGG16,
        'ResNet': ResNet50,
        'Inception': InceptionV3,
    }

    def __init__(self,
                 # Processing
                 zoom_range: bool = True,
                 horizontal_flip: bool = True,
                 brightness_range: bool = True,
                 color_change: bool = True,
                 target_size: Tuple = (128, 128),

                 # Training
                 model_type: str = 'VGG',
                 batch_size: int = 12,
                 epochs: int = 5,

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
        self.models = {}
        self.timestamp = str(int(time.time()))

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
        test_datagen = ImageDataGenerator(rescale=1/255.)
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

    def _prediction_generator(self, color: str, cls: str):
        """
        Create a keras generator for predictions
        """
        predict_datagen = ImageDataGenerator(rescale=1/255.)
        predict_generator = predict_datagen.flow_from_directory(
            directory=os.path.join(self.curated_dir, color, 'test', cls),
            target_size=self.target_size,
            color_mode='rgb',
            classes=None,
            batch_size=1,
            class_mode='binary',
            shuffle=False,
            seed=187
        )
        return predict_generator

    def _class_weights(self, color: str) -> dict:
        """
        Output a dictionary to weight classes:
        The data generators list negative first, so its index is 0 and weight is set at 1.;
        The weight on the positive class (1) is the number of neg-cases / number of pos-cases
        """
        train_dir = os.path.join(self.curated_dir, color, 'train')
        num_pos_cases = len(os.listdir(os.path.join(train_dir, 'positive')))
        num_neg_cases = len(os.listdir(os.path.join(train_dir, 'negative')))

        return {0: 1., 1: num_neg_cases / num_pos_cases}

    def train(self):
        """
        Train a model for each color class
        """
        # Callbacks
        tensorboard = TensorBoard(log_dir=os.path.join(ROOT_DIR, 'logs'))
        csv_logger = CSVLogger(os.path.join(ROOT_DIR, 'logs', 'csvlogger.csv'), separator=',', append=False)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=0, cooldown=2)

        # Train a model for each color
        for color in self.card_colors:
            train, test = self.process(color)

            # Instantiate model
            model = self.model_options.get(self.model_type)(
                include_top=True,
                weights='imagenet',
            )
            if model is None:
                raise ValueError('Incorrect model selection.')

            # Fit the model
            model.fit(x=train,
                      steps_per_epoch=train.n // train.batch_size,
                      validation_data=test,
                      validation_steps=test.n // test.batch_size,
                      class_weight=self._class_weights(color),
                      epochs=self.epochs,
                      verbose=2,
                      callbacks=[tensorboard, csv_logger, early_stopping, reduce_lr]
                      )
            self.models[color] = model

    def diagnose(self):
        """
        Run diagnostics for a trained model
        """
        if len(self.models) == 0:
            raise ValueError('Train models first.')

        # Get predictions for each positive class with every model
        df_results = []
        for model_color, model in self.models.items():
            for card_color in self.card_colors:
                logger.info('Predicting {} with {}'.format(card_color, model_color))
                pred_generator = self._prediction_generator(card_color, 'positive')
                predictions = model.predict(
                    pred_generator,
                    batch_size=1,
                    verbose=1,
                )
                filenames = pred_generator.filenames
                df_result = pd.DataFrame({'preds': predictions, 'filenames': filenames}).\
                    assign(ModelColor=model_color, CardColor=card_color)
                df_results.append(df_result)
        df_results = pd.concat(df_results).reset_index(drop=True)

        # Plots
        with PdfPages(os.path.join(self.results_dir, 'diagnostics_{}.pdf'.format(self.timestamp))) as pdf:

            # ROC Curve
            logger.info('ROC Curves.')
            fig, ax = plt.subplots(nrows=3, ncols=int(len(self.card_colors) / 3), figsize=(12, 12))
            ax = ax.reshape(-1, 1)
            for idx, (color, df_model) in enumerate(df_results.groupby('ModelColor')):
                preds = df_model['preds'].values
                y = (df_model['CardColor'] == color).values
                fpr, tpr, th = roc_curve(y, preds)
                score = auc(y, preds)

                ax[idx].plot(fpr, tpr, label='AUC: {a:0.2f}'.format(a=score))
                ax[idx].plot([0, 1], [0, 1], color='black')
                ax[idx].set_ylabel('True Positive Rate')
                ax[idx].set_xlabel('False Positive Rate')
                ax[idx].set_title(color)
                ax[idx].legend()
                ax[idx].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Precision / Recall
            logger.info('Precision / Recall.')
            fig, ax = plt.subplots(nrows=3, ncols=int(len(self.card_colors) / 3), figsize=(12, 12))
            ax = ax.reshape(-1, 1)
            for idx, (color, df_model) in enumerate(df_results.groupby('ModelColor')):
                preds = df_model['preds'].values
                y = (df_model['CardColor'] == color).values
                fpr, tpr, th = roc_curve(y, preds)

                ax[idx].plot(th, fpr, label='False-Positive')
                ax[idx].plot(th, tpr, label='True-Positive')
                ax[idx].set_xlabel('Threshold')
                ax[idx].set_title(color)
                ax[idx].legend()
                ax[idx].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Histograms
            logger.info('Histograms.')
            fig, ax = plt.subplots(nrows=3, ncols=int(len(self.card_colors) / 3), figsize=(12, 12))
            ax = ax.reshape(-1, 1)
            for idx, (color, df_model) in enumerate(df_results.groupby('ModelColor')):
                preds = df_model['preds'].values
                y = (df_model['CardColor'] == color).values

                bins = np.linspace(0., 1., 40)
                for y_val in set(y):
                    ax[idx].hist(preds[y == y_val], label=y_val, density=True, alpha=0.5, bins=bins)
                ax[idx].set_xlabel('Preds')
                ax[idx].title(color)
                ax[idx].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Card samples
        with PdfPages(os.path.join(self.results_dir, 'samples_{}.pdf'.format(self.timestamp))) as pdf:

            # Sample images for each model
            logger.info('Getting Select Examples.')
            df_samples = []
            for color in set(df_results['ModelColor']):
                # True Positive, high scoring cards of this color
                tps = df_results[(df_results['CardColor'] == color) & (df_results['ModelColor'] == color)].\
                    sort_values('preds', ascending=False)[['filename', 'preds']]. \
                    head(4). \
                    assign(ModelColor=color, CardColor=color, SampleType='TruePositive')
                # False negatives, low scoring cards of this color
                fns = df_results[(df_results['CardColor'] == color) & (df_results['ModelColor'] == color)]. \
                    sort_values('preds', ascending=True)[['filename', 'preds']].\
                    head(4).\
                    assign(ModelColor=color, CardColor=color, SampleType='FalseNegative')

                # Gather
                df_samples.append(pd.concat([tps, fns]))

                # Loop for false positives
                for other_color in set(df_results['CardColor']):
                    if other_color == color:
                        continue
                    fps = df_results[(df_results['CardColor'] == other_color) & (df_results['ModelColor'] == color)].\
                        sort_values('preds', ascending=False)[['filename', 'preds']]. \
                        head(4). \
                        assign(ModelColor=color, CardColor=other_color, SampleType='FalsePositive')
                    df_samples.append(fps)
            df_samples = pd.concat(df_samples).reset_index(drop=True)

            # Plot cards
            logger.info('Saving card samples.')
            for (model_color, card_color, sample_type), df_plot in df_samples. \
                    groupby(['ModelColor', 'CardColor', 'SampleType']):
                card_dir = os.path.join(ROOT_DIR, 'data', 'curated', card_color, 'test', 'positive')

                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
                ax = ax.reshape(-1, 1)
                for idx, (filename, df_score) in enumerate(df_plot.groupby('filename')):
                    img = cv2.imread(os.path.join(card_dir, filename))
                    ax[idx].imshow(img)
                    ax[idx].set_title('{fn}, Score: {a:0.3f}'.format(fn=filename, a=df_score['preds'].iloc[0]))
                plt.axis('off')
                plt.suptitle('Model: {}, Cards: {}, SampleType: {}'.format(model_color, card_color, sample_type))
                pdf.savefig()
                plt.close()

    def save(self):
        """
        Save model
        """
        logger.info('Saving Classifier.')
        save_file = 'magic_card_classifier_{}.pkl'.format(self.timestamp)
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', save_file), 'wb') as fp:
            pickle.dump(self, fp)
