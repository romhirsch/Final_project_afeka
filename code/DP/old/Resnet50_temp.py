import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from augmentation.Exdark_analyze import *
import fiftyone.zoo as foz
import fiftyone
import matplotlib.pyplot as plt
tf.config.list_physical_devices()
# List available zoo datasets
print(foz.list_zoo_datasets())
#dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

#dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")
# dataset = fiftyone.zoo.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#     label_types=["detections", "segmentations"],
#     classes=["person", "car"],
#     max_samples=50,
# )

# session = fo.launch_app(dataset)

"""
Import your dataset:
"""
img_height, img_width = 224, 224
batch_size = 32
exdark_dir = r"E:\dataset\ExDark"
coco_val_dir = r"E:\dataset\coco_val"
dataset_dir = coco_val_dir
(train_ds, val_ds) = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_dir,
validation_split =0.2,
  subset="both",
  seed=123,
  batch_size=batch_size,
  image_size=(img_height, img_width),
  crop_to_aspect_ratio=True)



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")


"""
 Import Pre-trained Model
"""
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
#strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  resnet_model = Sequential()
  pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                     input_shape=(img_height, img_width, 3),
                     pooling='avg',
                     classes=len(train_ds.class_names),
                     weights='imagenet')
  for layer in pretrained_model.layers:
          layer.trainable = False

  resnet_model.add(pretrained_model)
  resnet_model.add(Flatten())
  resnet_model.add(Dense(512, activation='relu'))
  resnet_model.add(Dense(len(train_ds.class_names), activation='softmax'))
  resnet_model.summary()

  resnet_model.compile(optimizer=Adam(),
                       loss='scategorical_crossentropy',
                       metrics=['accuracy'])

history = resnet_model.fit(train_ds,
                           validation_data=val_ds,
                           epochs=20)
plt.figure()
fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)