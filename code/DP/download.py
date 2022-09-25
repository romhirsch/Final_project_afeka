import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from augmentation.Exdark_analyze import *
import fiftyone.zoo as foz
import fiftyone
import fiftyone.utils.random as four

# List available zoo datasets
print(foz.list_zoo_datasets())
#dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

#dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    #split="validation",
    label_types=["detections"],
    classes=["bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorcycle", "person", "dining table"],
    #max_samples=50,
)