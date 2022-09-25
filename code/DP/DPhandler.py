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

    def __init__(self, classes, num_epochs, early_stop_patience=10, batch_training=100, batch_valid=100, image_size=224, channels=3, image_resize=224):
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.classes = classes
        self.num_classes = len(classes)
        self._channels = channels
        self.image_size = image_size
        self.batch_training = batch_training
        self.batch_valid = batch_valid
        self.checkpoint_path = r"E:\dataset\checkpoints"


    def pretrain(self, model, weights, include_top=False):
        self.pretrain_model = model(include_top=include_top, pooling='avg', weights=weights)


    def create_model_transfer(self, classification_layers):
        self.model = Sequential()
        self.model.add(self.pretrain_model)
        for layer in classification_layers:
            self.model.add(layer)
        self.model.layers[0].trainable = False
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    def loadDataset_preprocessing(self, dataset_dir, preprocess):

        data_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                            validation_split=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        # flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
        # Both train & valid folders must have NUM_CLASSES sub-folders
        #dataset_dir = r"E:\dataset\coco\train"
        #dataset_valid_dir = r"E:\dataset\coco\valid"
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
            validation_steps=self.step_per_epoch_valid,
            callbacks=[cp_callback, cb_early_stopper])


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
    def load_dataset(dataset_dir, BATCH_SIZE_TESTING, image_size):
        data_generator = ImageDataGenerator()
        ds = data_generator.flow_from_directory(
            directory=dataset_dir,
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE_TESTING,
            classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorcycle", "person",
                     "dining table"],
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

    def dshist(self):
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].set_ylabel('train ds')
        ax[1].set_ylabel('valid ds')

        h1 = ax[0].hist(np.array(list(self.train_generator.class_indices.keys()))[self.train_generator.labels])
        h2 = ax[1].hist(np.array(list(self.validation_generator.class_indices.keys()))[self.validation_generator.labels])
        return h1, h2

if __name__ == '__main__':
    dataset_dir = r"E:\dataset\coco\train"
    dataset_test_dir = r"E:\dataset\coco\valid"
    dir = r"E:\dataset\ExDark"
    classes = ["bicycle", "boat", "bottle", "bus", "car",
               "cat", "chair", "cup", "dog", "motorcycle",
               "person", "dining table"]
    weights = 'imagenet'
    checkpoint_path = r"E:\dataset\checkpoints"
    saved_models_path = r"E:\dataset\models_saved"
    pretrain_model = ResNet50
    dp = DPhandler(classes=classes,
              num_epochs=50,
              early_stop_patience=5,
              batch_training=32,
              batch_valid=32)
    dp.pretrain(pretrain_model, weights=weights)
    dp.create_model_transfer([Flatten(),
                              Dense(512, activation='relu'),
                              Dense(len(classes), activation='softmax')])
    dp.loadDataset_preprocessing(dataset_dir, preprocess_input)
    dp.fit()
    dp.plot_acc_loss()
    # Save the weights using the `checkpoint_path` format
    dp.model.save_weights(checkpoint_path.format(epoch=0))
    loss, acc = dp.model.evaluate(dp.validation_generator, verbose=2)
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    #model.save('my_model.h5')
    #dp.model.evaluate()
    # model.load_weights('./checkpoints/my_checkpoint')
    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    # # Recreate the exact same model, including its weights and the optimizer
    # new_model = tf.keras.models.load_model('my_model.h5')
    #
    # # Show the model architecture
    # new_model.summary()
    pass