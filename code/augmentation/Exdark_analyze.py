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
    df = pd.DataFrame(columns=['image_name', 'brightness', 'contrast', 's_hsv', 'light', 'blur'], dtype=np.float64)
    for i, (img, name) in enumerate(zip(images, image_name)):
        try:
            df.loc[i, 'image_name'] = name
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            df.loc[i, 's_hsv'] = hsv[:, :, 1].mean()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            df.loc[i, 'brightness'] = np.median(img_gray)
            df.loc[i, 'contrast'] = np.std(img)
            df.loc[i, 'blur'] = variance_of_laplacian(img)
            df.loc[i, 'light'] = ExDark.df[ExDark.df['Name'] == name]['Light'].to_numpy()
        except:
            pass
    return df

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
        self.img_data_array = np.array(img_data_array)
        self.class_name = np.array(class_name)
        self.img_name = np.array(img_name)
        return img_data_array, class_name




if __name__ == '__main__':
    img_folder = r"E:\dataset\ExDark"
    ExDark = Dataset_creator(img_folder, exdark=True)
    ExDark.create_dataset()
    Names = ExDark.df[ExDark.df['Name'].isin(ExDark.img_name)].to_numpy()#& (ExDark.df['Light']==1)]['Name'].to_numpy()
    ind = np.where(np.isin(ExDark.img_name, np.array(Names)))
    df = calc_img_params(ExDark.img_data_array[ind], ExDark.img_name[ind])
    ax = df.boxplot(column=['brightness', 'contrast', 's_hsv'], by='light')
    light_label = ['Low', 'Ambient', 'Object', 'Single', 'Weak', 'Strong', 'Screen', 'Window', 'Shadow', 'Twilight']
    ax = df.boxplot(column=['blur'], by='light')
    ax.set_xticklabels(light_label)
    plt.show()
