from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

print(tf.__version__)
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

IMG_SIZE = 224
(ds_train, ds_test), ds_info = tfds.load('imagenet_v2',
                          split=['test[25%:]', 'test[0:25%]'],
                          as_supervised=True,
                          #batch_size=-1,
                          with_info=True,
                          data_dir=r"C:\Users\rom21\OneDrive\Desktop\git_project\code\dataset")

print("Train set size: ", len(ds_train))
print("Test set size: ", len(ds_test))

NUM_CLASSES = ds_info.features["label"].num_classes
batch_size = 50
IMG_SIZE = 224
epochs = 20
size = (IMG_SIZE, IMG_SIZE)

def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)

ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

with strategy.scope():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = img_augmentation(inputs)
    outputs = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', classes=NUM_CLASSES)(inputs)
    #outputs = EfficientNetB0(include_top=True, weights='imagenet', classes=NUM_CLASSES)(inputs)
    model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer='sgd', # stochastic gradient descent (SGD)
              loss='categorical_crossentropy', #  log-loss - this is another term for the same thing
              metrics=['accuracy']) # report the accuracy metric


import numpy as np
evaluation = model.evaluate(ds_test, return_dict=True)
# print('Computing predictions...')
# test_images_ds = ds_test.map(lambda image, idnum: image)
# predictions = model.predict(test_images_ds)
# precege = tf.nn.softmax(predictions)
# predictions = np.argmax(predictions, axis=-1)
# print(predictions)
#
# data=[]
# for images, labels in ds_test.unbatch():
#     data.append(np.where(labels>0))
#
#
#
# # Get image ids from test set and convert to unicode
# test_ids_ds = ds_test.map(lambda image, idnum: idnum).unbatch()
# test_ids = next(iter(test_ids_ds).numpy().astype('U'))
# np.savetxt(
#     'submission.csv',
#     np.rec.fromarrays([test_ids, predictions]),
#     fmt=['%s', '%d'],
#     delimiter=',',
#     header='id,label',
#     comments='',
# )


hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)


#fig = tfds.show_examples(ds)



def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train"])#, "validation"], loc="upper left")
    plt.show()


plot_hist(hist)


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
# with strategy.scope():
#     model = build_model(num_classes=NUM_CLASSES)
pass