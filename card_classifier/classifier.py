import os
import cv2
from typing import Tuple

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from sklearn.metrics import auc, roc_curve

from config import Config, logger


class MagicCardClassifier(object):
    """
    Object to train, diagnose, and predict magic card colors from images.
    """
    # All colors to train on
    card_colors = ['W', 'G', 'N', 'R', 'U', 'B']

    # Curated dir
    curated_dir = os.path.join(Config.DATA_DIR, 'card_classifier', 'curated')

    # Models
    model_options = {
        'VGG': VGG16,
        'ResNet': ResNet50,
        'Inception': InceptionV3,
    }

    target_size = (128, 128)

    def __init__(self,
                 model_type: str,
                 train_color: str = 'B',
                 # Processing
                 zoom_range: bool = False,
                 horizontal_flip: bool = False,
                 brightness_range: bool = False,

                 # Training
                 learning_rate: float = 10 ** -3,
                 batch_size: int = 32,
                 epochs: int = 20,

                 # I/O
                 debug: bool = False,
                 load: bool = False,
                 version: str = 'v0',
                 results_dir: str = os.path.join(Config.RESULTS_DIR, 'card_classifier')
                 ):

        # Processing
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range

        # Training
        self.train_color = train_color
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # I/O
        self.debug = debug
        self.version = version
        self.results_dir = os.path.join(results_dir, model_type, version, self.train_color)

        # Load
        self.models = {}
        if load:
            for color in self.card_colors:
                if os.path.exists(os.path.join(os.path.dirname(self.results_dir), color)):
                    self.models[color] = load_model(filepath=os.path.join(os.path.dirname(self.results_dir), color))

    def process(self, color: str) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Set up Keras generators for augmented data
        """
        # Set up training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1/255.,
            zoom_range=0.15 if self.zoom_range else 0.,
            horizontal_flip=True if self.horizontal_flip else False,
            brightness_range=[0.9, 1.1] if self.brightness_range else None,
            data_format='channels_last',
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

    def _prediction_generator(self, input_dir: str):
        """
        Create a Keras generator for predictions
        """
        predict_datagen = ImageDataGenerator(rescale=1/255.)
        predict_generator = predict_datagen.flow_from_directory(
            directory=input_dir,
            target_size=self.target_size,
            color_mode='rgb',
            classes=None,
            batch_size=1,
            class_mode=None,
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
        train, test = self.process(self.train_color)

        # Instantiate model
        if self.debug:
            # Quick model for debugging
            model = Sequential([
                Dense(3, activation='sigmoid'),
                Dense(3, activation='sigmoid'),
                Dense(3, activation='sigmoid')
            ])
        else:
            model = Sequential()
            model.add(self.model_options.get(self.model_type)(
                include_top=False,
                weights='imagenet',
                input_shape=(self.target_size[0], self.target_size[1], 3)
            ))
            # 10 Dense NN on "top"
            model.add(Dense(10))

        # Add activation layer for binary classification
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[AUC()]
        )

        # Callbacks
        tensorboard = TensorBoard(log_dir=os.path.join(Config.ROOT_DIR, 'logs'))
        csv_logger = CSVLogger(os.path.join(Config.ROOT_DIR, 'logs', 'csvlogger_{}_{}_{}.csv'.format(
            self.train_color, self.model_type, self.version)), separator=',', append=False)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=0,
                                      cooldown=2)
        callbacks = [
            tensorboard,
            csv_logger,
            # early_stopping,
            reduce_lr
        ] if not self.debug else [csv_logger]

        # Fit the model
        model.fit(x=train,
                  steps_per_epoch=(train.n / 10) // train.batch_size if not self.debug else 10,
                  validation_data=test,
                  validation_steps=test.n // test.batch_size if not self.debug else 10,
                  class_weight=self._class_weights(self.train_color),
                  epochs=self.epochs if not self.debug else 3,
                  verbose=1,
                  callbacks=callbacks
                  )

        # Save the model
        self.models[self.train_color] = model

    def diagnose(self):
        """
        Run diagnostics for the trained models
        """
        if len(self.models) == 0:
            raise ValueError('Train models first.')

        # Get predictions for each positive class with every model trained so far
        df_results = []
        for model_color, model in self.models.items():
            for card_color in self.card_colors:
                # Get generator
                pred_generator = self._prediction_generator(os.path.join(self.curated_dir, card_color, 'validation'))
                # Predict
                logger.info('Predicting {} images in {} with {}'.format(pred_generator.n, card_color, model_color))
                predictions = model.predict(
                    x=pred_generator,
                    batch_size=1,
                    verbose=1,
                    steps=pred_generator.n if not self.debug else 10
                )[:, 0]
                # Gather
                df_result = pd.DataFrame({
                    'preds': predictions,
                    'filename': pred_generator.filenames if not self.debug else pred_generator.filenames[:10]
                }).\
                    assign(ModelColor=model_color, CardColor=card_color)
                df_results.append(df_result)
        df_results = pd.concat(df_results).reset_index(drop=True)

        # Plots that build up as models are trained
        with PdfPages(os.path.join(self.results_dir, 'diagnostics.pdf')) as pdf:
            # Training Logs
            logger.info('Training Logs.')
            for metric in ['loss', 'auc']:
                fig, ax = plt.subplots(nrows=2, ncols=int(len(self.card_colors) / 2), figsize=(12, 12))
                for idx, color in enumerate(self.card_colors):
                    log_path = os.path.join(Config.ROOT_DIR, 'logs', 'csvlogger_{}_{}_{}.csv'.format(
                        color, self.model_type, self.version
                    ))
                    if not os.path.exists(log_path):
                        continue
                    df_logs = pd.read_csv(log_path)
                    # Force column names
                    df_logs.columns = ['epoch', 'auc', 'loss', 'val_auc', 'val_loss']

                    # Plot
                    ax[idx // 3, idx % 3].plot(df_logs['epoch'], df_logs[metric], label='training')
                    ax[idx // 3, idx % 3].plot(df_logs['epoch'], df_logs['val_' + metric], label='test')
                    ax[idx // 3, idx % 3].legend()
                    ax[idx // 3, idx % 3].set_xlabel('Epoch')
                    ax[idx // 3, idx % 3].set_ylabel(metric)
                    ax[idx // 3, idx % 3].set_title(color)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # ROC Curve
            logger.info('ROC Curves.')
            fig, ax = plt.subplots(nrows=2, ncols=int(len(self.card_colors) / 2), figsize=(12, 12))
            for idx, (color, df_model) in enumerate(df_results.groupby('ModelColor')):
                preds = df_model['preds']
                y = (df_model['CardColor'] == color).astype(int)
                fpr, tpr, th = roc_curve(y, preds)
                score = auc(fpr, tpr)

                ax[idx // 3, idx % 3].plot(fpr, tpr, label='AUC: {a:0.2f}'.format(a=score))
                ax[idx // 3, idx % 3].plot([0, 1], [0, 1], color='black')
                ax[idx // 3, idx % 3].set_ylabel('True Positive Rate')
                ax[idx // 3, idx % 3].set_xlabel('False Positive Rate')
                ax[idx // 3, idx % 3].set_title(color)
                ax[idx // 3, idx % 3].legend()
                ax[idx // 3, idx % 3].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Precision / Recall
            logger.info('Precision / Recall.')
            fig, ax = plt.subplots(nrows=2, ncols=int(len(self.card_colors) / 2), figsize=(12, 12))
            for idx, (color, df_model) in enumerate(df_results.groupby('ModelColor')):
                preds = df_model['preds']
                y = (df_model['CardColor'] == color).astype(int)
                fpr, tpr, th = roc_curve(y, preds)

                ax[idx // 3, idx % 3].plot(th, fpr, label='False-Positive')
                ax[idx // 3, idx % 3].plot(th, tpr, label='True-Positive')
                ax[idx // 3, idx % 3].set_xlabel('Threshold')
                ax[idx // 3, idx % 3].set_title(color)
                ax[idx // 3, idx % 3].legend()
                ax[idx // 3, idx % 3].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Histograms
            logger.info('Histograms.')
            fig, ax = plt.subplots(nrows=2, ncols=int(len(self.card_colors) / 2), figsize=(12, 12))
            for idx, (color, df_model) in enumerate(df_results.groupby('ModelColor')):
                bins = np.linspace(0., 1., 40)
                for card_color, df_plot in df_model.groupby('CardColor'):
                    ax[idx // 3, idx % 3].hist(df_model[df_model['CardColor'] == card_color]['preds'], label=card_color,
                                               density=True, alpha=0.5, bins=bins)
                ax[idx // 3, idx % 3].set_xlabel('Preds')
                ax[idx // 3, idx % 3].set_title(color)
                ax[idx // 3, idx % 3].grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Card samples
        with PdfPages(os.path.join(self.results_dir, 'samples.pdf')) as pdf:
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
                card_dir = os.path.join(os.path.join(Config.DATA_DIR, 'card_classifier', 'curated', card_color,
                                                     'validation'))

                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
                for idx, (filename, df_score) in enumerate(df_plot.groupby('filename')):
                    img = cv2.imread(os.path.join(card_dir, filename))
                    ax[idx // 2, idx % 2].imshow(img[:, :, [2, 1, 0]])
                    ax[idx // 2, idx % 2].set_title('{fn}, Score: {a:0.3f}'.format(
                        fn=filename, a=df_score['preds'].iloc[0]))
                plt.suptitle('Model: {}, Cards: {}, SampleType: {}'.format(model_color, card_color, sample_type))
                pdf.savefig()
                plt.close()

    def save(self):
        """
        Save models
        """
        logger.info('Saving Classifier.')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.models[self.train_color].save(self.results_dir)

    def load(self):
        """
        Load models
        """
        logger.info('Loading Classifiers.')
        self.models = {}
        for color in self.card_colors:
            if os.path.exists(os.path.join(os.path.dirname(self.results_dir), color)):
                self.models[color] = load_model(os.path.join(os.path.dirname(self.results_dir), color), compile=True)

    def predict(self, input_path: str) -> dict:
        """
        From a directory of images or single path generate likelihood of each color
        input_path should point to a directory containing a subdirectory of images to predict
        """
        if self.models is None:
            self.load()

        # Make generator
        pred_generator = self._prediction_generator(input_path)

        # Get predictions
        preds_by_color = {
            color: self.models[color].predict(
                    x=pred_generator,
                    batch_size=1,
                    verbose=1,
                    steps=pred_generator.n
                )[:, 0] for color in self.card_colors if color in self.models.keys()
        }

        # Wrangle outputs so that primary key is filename, values are dict with keys as colors and vals as preds
        output = {
            fn: {
                color: preds[idx] for color, preds in preds_by_color.items()
            } for idx, fn in enumerate(pred_generator.filenames)
        }

        return output
