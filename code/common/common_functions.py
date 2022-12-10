import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2
import os

def plot_img(im2, title='image'):
    """
    :param im2: image  (load from cv2 format BGR)
    plot image + histogram colors
    """
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].set_title(title)
    ax[0].imshow(im2)
    ax[1].hist(im2[..., 0].flatten(), 256, [0, 256], color='r')
    ax[1].hist(im2[..., 1].flatten(), 256, [0, 256], color='g')
    ax[1].hist(im2[..., 2].flatten(), 256, [0, 256], color='b')

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()