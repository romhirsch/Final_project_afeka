import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2
import pandas as pd
import os
import re
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import csv
import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2
import pandas as pd
import os
import re
from common.common import *
import seaborn as sns

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[..., 0].flatten(), 256, [0, 256], color='b')
    ax[1].hist(im2[..., 1].flatten(), 256, [0, 256], color='g')
    ax[1].hist(im2[..., 2].flatten(), 256, [0, 256], color='r')

def calc_img_params(images, image_name):
    df = pd.DataFrame(columns=['image_name', 'brightness', 'contrast', 's_hsv', 'light', 'blur', 'Y'], dtype=np.float64)

    for i, (img, name) in enumerate(zip(images, image_name)):
        try:
            df.loc[i, 'image_name'] = name
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            df.loc[i, 's_hsv'] = hsv[:, :, 1].mean()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            df.loc[i, 'brightness'] = np.median(img_gray)
            df.loc[i, 'contrast'] = np.std(img)
            df.loc[i, 'blur'] = variance_of_laplacian(img)
            df.loc[i, 'Y'] = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0].mean()
            df.loc[i, 'light'] = ds.df[ds.df['Name'] == name]['Light'].to_numpy()
        except:
            pass
    return df

def plot_diff_ds(df_all):
    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    f.suptitle('Brightness', fontsize=14)
    sns.boxplot(x="ds", y="brightness", data=df_all, ax=ax)
    ax.set_xlabel("datasets", size=12, alpha=0.8)
    ax.set_ylabel("brightness", size=12, alpha=0.8)
    plt.savefig('Brightness_datasets.png')

    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    f.suptitle('Contrast', fontsize=14)
    sns.boxplot(x="ds", y="contrast", data=df_all, ax=ax)
    ax.set_xlabel("datasets", size=12, alpha=0.8)
    ax.set_ylabel("contrast", size=12, alpha=0.8)
    plt.savefig('Contrast_datasets.png')

    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    f.suptitle('s_hsv', fontsize=14)
    sns.boxplot(x="ds", y="s_hsv", data=df_all, ax=ax)
    ax.set_xlabel("datasets", size=12, alpha=0.8)
    ax.set_ylabel("s_hsv", size=12, alpha=0.8)
    plt.savefig('s_hsv_datasets.png')

    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    f.suptitle('blur', fontsize=14)
    sns.boxplot(x="ds", y="blur", data=df_all, ax=ax)
    ax.set_xlabel("datasets", size=12, alpha=0.8)
    ax.set_ylabel("blur", size=12, alpha=0.8)
    plt.savefig('blur_datasets.png')

    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    f.suptitle('Y luminance', fontsize=14)
    sns.boxplot(x="ds", y="Y", data=df_all, ax=ax)
    ax.set_xlabel("datasets", size=12, alpha=0.8)
    ax.set_ylabel("blur", size=12, alpha=0.8)
    plt.savefig('blur_datasets.png')
class Dataset_creator():

    def __init__(self, dataset_folder, exdark=False):
        self.dataset_folder = dataset_folder
        self.df = pd.DataFrame()
        self.exdark = exdark
        if self.exdark:
            """
            Class column: Bicycle(1), Boat(2), Bottle(3), Bus(4), Car(5), Cat(6), Chair(7), Cup(8), Dog(9), Motorbike(10), People(11), Table(12)
            Light column: Low(1), Ambient(2), Object(3), Single(4), Weak(5), Strong(6), Screen(7), Window(8), Shadow(9), Twilight(10)
            In/Out column: Indoor(1), Outdoor(2)
            Train/Val/Test: Training(1), Validation(2), Testing(3)
            """
            self.df = pd.read_csv((r"E:\dataset\metadata.txt"), delimiter=' ')
            self.df.loc[:, 'valid'] = False
        self.formats = [".jpg", ".png", ".jpeg"]


    def create_dataset(self):
        img_data_array = []
        class_name = []
        img_name = []
        for dir1 in os.listdir(img_folder):
            print(os.path.join(img_folder, dir1))
            for file in os.listdir(os.path.join(img_folder, dir1)):
                if file.endswith(tuple(self.formats)):
                    image_path = os.path.join(img_folder, dir1, file)
                    image = cv2.imread(image_path)#, cv2.COLOR_BGR2RGB)
                    #image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                    #image = np.array(image)
                    #image = image.astype('float32')
                    #image /= 255
                    img_data_array.append(image)
                    class_name.append(dir1)
                    img_name.append(file)
        self.img_data_array = np.array(img_data_array, dtype=object)
        self.class_name = np.array(class_name)
        self.img_name = np.array(img_name)
        return img_data_array, class_name




if __name__ == '__main__':

    ds_names = [PathDatasets.EXDARK.name, PathDatasets.COCO_TEST_SMALL.name, PathDatasets.COCO_TRAIN_SMALL.name]
    img_folders = [PathDatasets.EXDARK.value, PathDatasets.COCO_TEST_SMALL.value, PathDatasets.COCO_TRAIN_SMALL.value ]
    df_all = pd.DataFrame()
    exdark_flag = False
    for ds_name, img_folder in zip(ds_names, img_folders):
        print(ds_name)
        ds = Dataset_creator(img_folder, exdark=exdark_flag)
        ds.create_dataset()
        if exdark_flag:
            Names = ds.df[ds.df['Name'].isin(ds.img_name)].to_numpy()#& (ExDark.df['Light']==1)]['Name'].to_numpy()
            ind = np.where(np.isin(ds.img_name, np.array(Names)))
        else:
            ind = np.arange(ds.img_data_array.shape[0])
        df = calc_img_params(ds.img_data_array[ind], ds.img_name[ind])
        df.loc[:, 'ds'] = ds_name
        df_all = df_all.append(df)

    # Box Plots
    plot_diff_ds(df_all)


    # if exdark_flag:
    #     ax = df.boxplot(column=['brightness', 'contrast', 's_hsv'], by='light')
    #     light_label = ['Low', 'Ambient', 'Object', 'Single', 'Weak', 'Strong', 'Screen', 'Window', 'Shadow', 'Twilight']
    #     ax = df.boxplot(column=['blur'], by='light')
    #     ax.set_xticklabels(light_label)
    # else:
    #     fig, ax = plt.subplots(1)
    #     ax = df.boxplot(column=['brightness', 'contrast', 's_hsv'], ax=ax)
    #     ax.set_title(ds_name)
    #     plt.savefig()
    #     fig, ax = plt.subplots(1)
    #     ax = df.boxplot(column=['blur'], ax=ax)
    #     ax.set_title(ds_name)


    plt.show()
