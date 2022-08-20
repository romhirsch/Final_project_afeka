import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2
import pandas as pd
import os
import re

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[..., 0].flatten(), 256, [0, 256])
    ax[1].hist(im2[..., 1].flatten(), 256, [0, 256])
    ax[1].hist(im2[..., 2].flatten(), 256, [0, 256])

def calc_img_params(files, data_path=r'dataset\lol\our485\\'):
    df = pd.DataFrame(columns=['image_name', 'low_mean', 'high_mean', 'diff', 'contrast_low', 'contrast_high'], dtype=np.float64)
    for i, file in enumerate(files):
        df.loc[i, 'image_name'] = np.int32(re.findall('^\d+', file)[0])
        img_low = cv2.imread(os.path.join(data_path + 'low', file))
        hsv_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2HSV)
        df.loc[:, 's_hsv_low'] = hsv_low[:, :, 1].mean()
        img_gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
        df.loc[i, 'low_mean'] = np.median(img_gray_low)
        img_high = cv2.imread(os.path.join(data_path + 'high', file))
        hsv_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2HSV)
        df.loc[:, 's_hsv_high'] = hsv_high[:, :, 1].mean()
        img_gray_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
        df.loc[i, 'high_mean'] = np.mean(img_gray_high)
        df.loc[i, 'contrast_low'] = np.std(img_low)
        df.loc[i, 'contrast_high'] = np.std(img_high)
    df.loc[:, 'diff_Saturation'] = abs(df['s_hsv_high'] - df['s_hsv_low'])
    df.loc[:, 'diff_Brightness'] = df['high_mean'] - df['low_mean']
    df.loc[:, 'diff_Contrast'] = df['contrast_high'] - df['contrast_low']
    df.loc[:, 'Brightness'] = df['diff_Brightness'] / df['high_mean']
    df.loc[:, 'Contrast'] = df['diff_Contrast'] / df['contrast_high']
    df.loc[:, 'Saturation'] = df['s_hsv_low'] / df['diff_Saturation']

    return df


if __name__ == '__main__':
    dataset_path = r"C:\Users\rom21\OneDrive\Desktop\git_project\code\dataset\lol\our485\\"
    files = os.listdir(os.path.join(dataset_path, 'low'))
    #high_files = os.listdir(os.path.join(dataset_path, 'high'))
    df = calc_img_params(files, dataset_path)

    #img = cv2.imread(rf"dataset\lol\our485\high\{img_name}.png")

   # df.describe().to_excel(r"C:\Users\rom21\OneDrive\Desktop\git_project\code\img_darken_desc2.xlsx")
   # df.loc[:, 'Brightness_percentage'].boxplot()
    bp = df.boxplot(column=['Brightness', 'Contrast'], showmeans=True)
    plt.title('Changes precentage')
    img_name = '468'
    img = cv2.imread(rf"dataset\lol\our485\high\{img_name}.png")
    plotim(img)
    img = cv2.imread(rf"dataset\lol\our485\low\{img_name}.png")
    plotim(img)
    pass