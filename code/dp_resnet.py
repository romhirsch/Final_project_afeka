import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from functools import partial
from sklearn.model_selection import train_test_split
print("Tensorflow version " + tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
#GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 16
IMAGE_SIZE = [512, 512]
CLASSES = ['0', '1', '2', '3', '4']
EPOCHS = 25

def decode_image(image):
    #normalize data and reshape to
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU.
    # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    return image, label


lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9)
img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
                                          input_shape=[*IMAGE_SIZE, 3])
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False
model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(renorm=True),
    img_adjust_layer,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(8, activation='relu'),
    # tf.keras.layers.BatchNormalization(renorm=True),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler, epsilon=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

    # load data
    # train_dataset = get_training_dataset()
    # valid_dataset = get_validation_dataset()

    # STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
    # VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE

    # history = model.fit(train_dataset,
    #                     steps_per_epoch=STEPS_PER_EPOCH,
    #                     epochs=EPOCHS,
    #                     validation_data=valid_dataset,
    #                     validation_steps=VALID_STEPS)
model.summary()
"""
Evaluating our model
"""
# print out variables available to us
#print(history.history.keys())
# create learning curves to evaluate model performance
# history_frame = pd.DataFrame(history.history)
# history_frame.loc[:, ['loss', 'val_loss']].plot()
# history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot();

"""
Making predictions
"""

# this code will convert our test image data to a float32
def to_float32(image, label):
    return tf.cast(image, tf.float32), label