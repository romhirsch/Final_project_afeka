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

# Fixed for our Cats & Dogs classes
NUM_CLASSES = 12
# Fixed for Cats & Dogs color images
CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 100

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 696
STEPS_PER_EPOCH_VALIDATION = 174

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 100

resnet_weights_path = 'imagenet'#'../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#Still not talking about our train/test data or any pre-processing.
model = Sequential()
# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights=resnet_weights_path))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
#   Dense for class classification, i.e using SoftMax activation
# model.add(Dense(512, activation='relu'))
#model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False
model.summary()
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)
image_size = IMAGE_RESIZE
# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
# Batch Normalization helps in faster convergence

def preprocess_extract_patch(p=0.2, size_list=[32, 64, 96, 128, 156]):
    def _preprocess_extract_patch(x):
        if np.random.random() < p:
        #     min_dim = min(x.shape[:2])
        #     _patch_size = sample(size_list, k=1)[0]
        #     patch_size = (_patch_size, _patch_size)
        #     x = extract_patches_2d(x, patch_size, max_patches=1, random_state=None)[0]
        #     x = np.array(Image.fromarray(x.astype('uint8'), 'RGB').resize((224, 224)))
        # x = x/255.0
            return preprocess_input(x)

        return preprocess_input(x)
    return _preprocess_extract_patch
preprocess_extract_patch = preprocess_extract_patch(p=0.9)

data_generator = ImageDataGenerator(preprocessing_function=preprocess_extract_patch,
                                    validation_split=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True)
# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
dataset_dir = r"E:\dataset\coco\train"
dataset_valid_dir =r"E:\dataset\coco\valid"
train_generator = data_generator.flow_from_directory(
        dataset_dir,
        classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorcycle", "person", "dining table"],
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        subset='training',
        shuffle=True,
        seed=42,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        dataset_dir,
        classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorcycle", "person", "dining table"],
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        subset='validation',
        seed=42,
        shuffle=True,
        class_mode='categorical')

# Max number of steps that these generator will have opportunity to process their source content
# len(train_generator) should be 'no. of available train images / BATCH_SIZE_TRAINING'
# len(valid_generator) should be 'no. of available train images / BATCH_SIZE_VALIDATION'
(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(validation_generator))
# Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
#cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
# Grid Search is an ideal candidate for distributed machine learning
# Pseudo code for hyperparameters Grid Search

'''
from sklearn.grid_search import ParameterGrid
param_grid = {'epochs': [5, 10, 15], 'steps_per_epoch' : [10, 20, 50]}

grid = ParameterGrid(param_grid)

# Accumulate history of all permutations (may be for viewing trend) and keep watching for lowest val_loss as final model
for params in grid:
    print(params)
'''
fit_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_early_stopper]

)
#model.load_weights(r"E:\dataset\best.hdf5")

print(fit_history.history.keys())

plt.figure(1, figsize=(15, 8))

plt.subplot(211)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(212)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.show()

# NOTE that flow_from_directory treats each sub-folder as a class which works fine for training data
# Actually class_mode=None is a kind of workaround for test data which too must be kept in a subfolder

# batch_size can be 1 or any factor of test dataset size to ensure that test dataset is samples just once, i.e., no data is left out

dataset_test_dir = r"E:\dataset\coco\valid" #r"E:\dataset\ExDark"
test_generator = data_generator.flow_from_directory(
    directory = dataset_valid_dir,
    target_size = (image_size, image_size),
    batch_size = BATCH_SIZE_TESTING,
    classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorcycle", "person",
             "dining table"],
    shuffle=False,
    seed=123
)
pred = model.predict_generator(test_generator, steps = len(test_generator), verbose=1)
from sklearn import metrics

predicted_class_indices = np.argmax(pred, axis = 1)
np.where(predicted_class_indices==test_generator.labels)
cm = confusion_matrix(test_generator.classes, predicted_class_indices)
score = metrics.accuracy_score(test_generator.classes, predicted_class_indices)

## Get Class Labels
labels = list(test_generator.class_indices.keys())
class_names = labels

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 10)
ax.xaxis.tick_bottom()
ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize = 10)
plt.yticks(rotation=0)
plt.title('Refined Confusion Matrix', fontsize=20)
plt.savefig('ConMat24.png')
plt.show()

TEST_DIR = dataset_valid_dir + '\\'
f, ax = plt.subplots(5, 5, figsize=(15, 15))
for i in range(0, 25):
        imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        # a if condition else b
        predicted_class = list(test_generator.class_indices.keys())[predicted_class_indices[i]]

        ax[i // 5, i % 5].imshow(imgRGB)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title("Predicted:{}".format(predicted_class))

plt.show()
