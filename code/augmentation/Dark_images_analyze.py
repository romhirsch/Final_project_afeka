import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2
import pandas as pd
import os
import re
import cv2
from skimage.restoration import estimate_sigma
import seaborn as sns

def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[..., 0].flatten(), 256, [0, 256])
    ax[1].hist(im2[..., 1].flatten(), 256, [0, 256])
    ax[1].hist(im2[..., 2].flatten(), 256, [0, 256])

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def calc_img_params(files, data_path=r'dataset\lol\our485\\'):
    df = pd.DataFrame(columns=['image_name', 'Brightness_low', 'Brightness_high', 'diff', 'Contrast_low', 'Contrast_high', 'blur_low', 'blur_high', 'noise_low', 'noise_high', 's_hsv_low', 's_hsv_high'], dtype=np.float64)
    for i, file in enumerate(files):
        df.loc[i, 'image_name'] = np.int32(re.findall('^\d+', file)[0])
        img_low = cv2.imread(os.path.join(data_path + 'low', file))
        hsv_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2HSV)
        df.loc[i, 's_hsv_low'] = hsv_low[:, :, 1].mean()
        df.loc[i, 'blur_low'] = variance_of_laplacian(img_low)
        df.loc[i, 'noise_low'] = estimate_noise(img_low)

        img_gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
        df.loc[i, 'Brightness_low'] = np.median(img_gray_low)
        img_high = cv2.imread(os.path.join(data_path + 'high', file))
        hsv_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2HSV)
        df.loc[i, 's_hsv_high'] = hsv_high[:, :, 1].mean()
        img_gray_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
        df.loc[i, 'Brightness_high'] = np.mean(img_gray_high)
        df.loc[i, 'Contrast_low'] = np.std(img_low)
        df.loc[i, 'Contrast_high'] = np.std(img_high)
        df.loc[i, 'blur_high'] = variance_of_laplacian(img_high)
        df.loc[i, 'noise_high'] = estimate_noise(img_high)

    df.loc[:, 'diff_Saturation'] = abs(df['s_hsv_high'] - df['s_hsv_low'])
    df.loc[:, 'diff_Brightness'] = df['Brightness_high'] - df['Brightness_low']
    df.loc[:, 'diff_Contrast'] = df['Contrast_high'] - df['Contrast_low']
    df.loc[:, 'Brightness'] = df['diff_Brightness'] / df['Brightness_high']
    df.loc[:, 'Contrast'] = df['diff_Contrast'] / df['Contrast_high']
    df.loc[:, 'Saturation'] = df['s_hsv_low'] / df['diff_Saturation']

    return df


if __name__ == '__main__':
    dataset_path = r"E:\dataset\lol\our485\\"
    files = os.listdir(os.path.join(dataset_path, 'low'))
    #high_files = os.listdir(os.path.join(dataset_path, 'high'))
    df = calc_img_params(files, dataset_path)

    #img = cv2.imread(rf"dataset\lol\our485\high\{img_name}.png")

   # df.describe().to_excel(r"C:\Users\rom21\OneDrive\Desktop\git_project\code\img_darken_desc2.xlsx")
   # df.loc[:, 'Brightness_percentage'].boxplot()
    bp = df.boxplot(column=['Brightness_low', 'Brightness_high', 'Contrast_low',  'Contrast_high'], showmeans=True)
    plt.title('Dateset Lol Contrast and Brightness')
    plt.figure()
    bp = df.boxplot(column=['blur_low', 'blur_high'], showmeans=True)
    plt.title('Dateset Lol blur')

    bp = df.boxplot(column=['s_hsv_low', 's_hsv_high'], showmeans=True)
    plt.title('Dateset Lol saturation')

    plt.title('Changes precentage')
    img_name = '468'
    img = cv2.imread(rf"dataset\lol\our485\high\{img_name}.png")
    plotim(img)
    img = cv2.imread(rf"dataset\lol\our485\low\{img_name}.png")
    plotim(img)
    pass