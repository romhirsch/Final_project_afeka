import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import layers,Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from augmentation.Exdark_analyze import *
import fiftyone.zoo as foz
import fiftyone

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



img_height, img_width = 180, 180
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)