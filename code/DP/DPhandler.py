import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
from enum import Enum
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import metrics

class DPhandler(object):

    def __init__(self, classes, num_epochs, early_stop_patience=10, batch_training=100, batch_valid=100, image_size=224, channels=3, finetuning=False, image_resize=224):
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.classes = classes
        self.num_classes = len(classes)
        self._channels = channels
        self.image_size = image_size
        self.batch_training = batch_training
        self.batch_valid = batch_valid
        self.checkpoint_path = r"E:\dataset\checkpoints"
        self.finetuning = finetuning


    def pretrain(self, model, weights, include_top=False):
        self.pretrain_model = model(include_top=include_top, pooling='avg', weights=weights)

    def create_model(self, classification_layers):
        self.model = Sequential()
        self.model.add(self.pretrain_model)
        for layer in classification_layers:
            self.model.add(layer)
        if self.finetuning:
            for layer in self.model.layers:
                layer.trainable = True
        else:
            self.model.layers[0].trainable = False
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        print('layers trainable:')
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name, layer.trainable)

    def loadDataset_preprocessing(self, dataset_dir, preprocess):

        data_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                            validation_split=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        # flow_From_directory generates batches of augmented data
        # Both train & valid folders must have NUM_CLASSES sub-folders
        self.train_generator = data_generator.flow_from_directory(
            dataset_dir,
            classes=self.classes,
            target_size=(self.image_size, self.image_size),
            batch_size= self.batch_training,
            subset='training',
            shuffle=True,
            seed=42,
            class_mode='categorical')

        self.validation_generator = data_generator.flow_from_directory(
            dataset_dir,
            classes=self.classes,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_valid,
            subset='validation',
            seed=42,
            shuffle=True,
            class_mode='categorical')
        self.step_per_epoch_train = len(self.train_generator)
        self.step_per_epoch_valid = len(self.validation_generator)
        print(self.batch_training, len(self.train_generator), self.batch_valid, len(self.validation_generator))


    def fit(self):
        cb_early_stopper = EarlyStopping(monitor='val_loss', patience=self.early_stop_patience)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        self.fit_history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.step_per_epoch_train,
            epochs=self.num_epochs,
            validation_data=self.validation_generator,
            callbacks=[cp_callback, cb_early_stopper],
            validation_steps=self.step_per_epoch_valid)
        #self.model.save_weights(checkpoint_path.format(epoch=0))

    def plot_acc_loss(self):
        plt.figure(1, figsize=(15, 8))
        plt.subplot(211)
        plt.plot(self.fit_history.history['accuracy'])
        plt.plot(self.fit_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'])
        plt.subplot(212)
        plt.plot(self.fit_history.history['loss'])
        plt.plot(self.fit_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'])
        plt.show()

    @staticmethod
    def load_dataset(dataset_dir, preprocess, classes, BATCH_SIZE=1, image_size=224):
        data_generator = ImageDataGenerator(preprocessing_function=preprocess)
        ds = data_generator.flow_from_directory(
            directory=dataset_dir,
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE,
            classes=classes,
            shuffle=False,
            seed=123
        )
        return ds


    def predict(self, ds):
        pred = self.model.predict_generator(ds, steps=len(ds), verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        return pred, predicted_class_indices


    @staticmethod
    def confuction_mat(y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        score = metrics.accuracy_score(y_true, y_pred)
        # Plot confusion matrix
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=10)
        ax.xaxis.tick_bottom()
        ax.set_ylabel('True', fontsize=20)
        ax.yaxis.set_ticklabels(class_names, fontsize=10)
        plt.yticks(rotation=0)
        plt.title(f'Refined Confusion Matrix score={score}', fontsize=20)
        plt.savefig('ConMat24.png')
        plt.show()


    def saved_model(self, path):
        self.model.save(path)


    def evaluate(self, ds):
        pred, predicted_class_indices = self.predict(ds)
        self.confuction_mat(ds.labels, predicted_class_indices, list(ds.class_indices.keys()))


    @staticmethod
    def dshistc(ds):
        fig, ax = plt.subplots(1)
        ax.set_ylabel('ds')
        h1 = ax.hist(np.array(list(ds.class_indices.keys()))[ds.labels])
        return h1

    def dshist(self, ds=''):
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].set_ylabel('train ds')
        ax[1].set_ylabel('valid ds')
        h1 = ax[0].hist(np.array(list(self.train_generator.class_indices.keys()))[self.train_generator.labels])
        h2 = ax[1].hist(np.array(list(self.validation_generator.class_indices.keys()))[self.validation_generator.labels])
        return h1, h2

    def load_model(self, path):
        # Recreate the exact same model, including its weights and the optimizer
        self.model = tf.keras.models.load_model(path)
        # Show the model architecture
        self.model.summary()

def load_saved_model(path):
    dp = DPhandler(classes=classes,
              num_epochs=20,
              early_stop_patience=3,
              batch_training=32,
              batch_valid=32,
              finetuning=True)
    dp.load_model(saved_models_path)
    return dp


def train_model():
    dp = DPhandler(classes=classes,
                   num_epochs=20,
                   early_stop_patience=3,
                   batch_training=32,
                   batch_valid=32,
                   finetuning=True)
    dp.pretrain(pretrain_model, weights=weights)
    dp.create_model([Flatten(),
                              Dense(512, activation='relu'),
                              Dense(len(classes), activation='softmax')])
    dp.loadDataset_preprocessing(dataset_dir, preprocess_input)
    dp.fit()
    dp.plot_acc_loss()
    return dp

if __name__ == '__main__':
    dataset_dir = r"E:\dataset\coco\train"
    dataset_test_dir = r"E:\dataset\coco\valid"
    dir = r"E:\dataset\ExDark"
    classes = ["bicycle", "boat", "bottle", "bus", "car",
               "cat", "chair", "cup", "dog", "motorcycle",
              "dining table"] #people
    weights = 'imagenet'
    checkpoint_path = r"E:\dataset\checkpoints"
    saved_models_path = r"E:\dataset\models_saved\resnet50.h"
    num_epochs = 20
    early_stop_patience = 3
    batch_training = 32
    batch_valid = 32
    pretrain_model = ResNet50
    ds_test = DPhandler.load_dataset(dataset_test_dir, preprocess_input, classes)
    ds_dark = DPhandler.load_dataset(dir, preprocess_input, classes)
    DPhandler.dshistc(ds_test)
    do_train = True

    if do_train:
        dp = train_model()
    else:
        dp = load_saved_model(saved_models_path)

    dp.saved_model(saved_models_path)
    dp.evaluate(ds_test)
    pass