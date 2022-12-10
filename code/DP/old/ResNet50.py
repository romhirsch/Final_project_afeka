import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from augmentation.ds_analyze import *
import fiftyone.zoo as foz
import fiftyone
import fiftyone.utils.random as four

# List available zoo datasets
print(foz.list_zoo_datasets())
#dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

#dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="train",
    classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorcycle", "person", "dining table"]
)

dataset.export(
    r"E:\dataset\coco\train3",
    fiftyone.types.ImageClassificationDirectoryTree,
label_field="ground_truth")


four.random_split(dataset, {"train": 0.8, "val": 0.2}) # split the dataset to val and train datasets
d = 'val'
val_ds = dataset.match_tags('val')
train_ds = dataset.match_tags('train')
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")
# session = fo.launch_app(dataset)

"""
Import your dataset:
"""



img_height, img_width = 180, 180
batch_size = 32
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")