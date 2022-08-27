import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
print("Tensorflow version " + tf.__version__)

def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize(i, [224, 224])
    #i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
    i = tf.image.resize(i, 1, 224, 224, 3)
    return (i, label)

#ds, ds_info\
ds, ds_info  = tfds.load('imagenet2012_real',
                          split='validation',
                          as_supervised=True,
                          #batch_size=-1,
                          with_info=True,
                          data_dir=r"C:\Users\rom21\OneDrive\Desktop\git_project\code\dataset")

NUM_CLASSES = ds_info.features["label"].num_classes
batch_size = 64
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)
ds = ds.map(lambda image, label: (tf.image.resize(image, size), label))
#fig = tfds.show_examples(ds)


# image, label = tfds.as_numpy(tfds.load(
#     'imagenet_v2',
#     split='test[:300]',
#     batch_size=-1,
#     as_supervised=True,
# ))

#fig = tfds.show_examples(ds, row=4, cols=4)
# print(image.shape)
BATCH_SIZE = 16
#IMAGE_SIZE = [224, 224]
CLASSES = 5

# include_top=False - exclude the last layer of the ResNet model that makes predictions

pretrained_model = tf.keras.applications.resnet50.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=[224, 224, 3])

pretrained_model.trainable = False
model = tf.keras.Sequential([
    # To a base pretrained on ImageNet to extract features from images...
    pretrained_model,
    # attach a new head to act as a classifier.
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1000, activation='softmax')
])
model.summary()

model.compile(optimizer='sgd', # stochastic gradient descent (SGD)
              loss='categorical_crossentropy', #  log-loss - this is another term for the same thing
              metrics=['accuracy']) # report the accuracy metric
model.fit(
        ds,
        steps_per_epoch=6,
        validation_steps=1)

# display_training_curves(
#     history.history['loss'],
#     history.history['val_loss'],
#     'loss',
#     211,
# )
# display_training_curves(
#     history.history['sparse_categorical_accuracy'],
#     history.history['val_sparse_categorical_accuracy'],
#     'accuracy',
#     212,
# )

# EPOCHS = 12
# STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
#
# history = model.fit(
#     ds_train,
#     validation_data=ds_valid,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     callbacks=[lr_callback],
# )

# num_classes = 2
# resnet_weights_path = r"C:\Users\rom21\OneDrive\Desktop\git_project\code\DP\input\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
#
#
# my_new_model = Sequential()
# my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
# my_new_model.add(Dense(num_classes, activation='softmax'))
#
# # Say not to train first layer (ResNet) model. It is already trained
# my_new_model.layers[0].trainable = False
#
# my_new_model.summary()
# my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# image_size = 224
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
#
#
# my_new_model.fit(
#         train_generator,
#         steps_per_epoch=6,
#         validation_data=validation_generator,
#         validation_steps=1)