import cv2
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.util.noise import random_noise


def bgr_to_bayer(img):
    (height, width) = img.shape[:2]
    (B, G, R) = cv2.split(img)
    bayer = np.empty((height, width), np.uint8)

    # strided slicing for this pattern:
    #   G R
    #   B G
    bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
    bayer[0::2, 1::2] = R[0::2, 1::2]  # top right
    bayer[1::2, 0::2] = B[1::2, 0::2]  # bottom left
    bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right
    return bayer

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256], color='r')
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256], color='g')
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256], color='b')
from sklearn.metrics import mean_squared_error

def uint8(img):
    return np.clip(img*255, 0, 255).astype(np.uint8)


img = cv2.imread(r"E:\dataset\lol\our485\high\714.png") # r"E:\dataset\lol\our485\high\482.png"
target_dark = cv2.imread(r"E:\dataset\lol\our485\low\715.png") #r"E:\dataset\lol\our485\low\481.png"
plotim(img) #
plotim(target_dark)
alphas = np.array([0.9]*15) #np.linspace(0.8, 1, 15)
gammas = np.linspace(1.5, 5, 15)
betas =np.linspace(0.5, 1, 15) #
img = np.float32(img/255)
target_dark = np.float32(target_dark/255)
img_dark_ycr = cv2.cvtColor(target_dark, cv2.COLOR_BGR2YCR_CB)
y_dark_hist, bin_edges = np.histogram(img_dark_ycr[..., 0], bins=256, range=(0, 1))
res = []
res_ind = []
xv, yv, zv = np.meshgrid(gammas, alphas, betas)
for i in range(len(xv)):
    for j in range(len(yv)):
        for s in range(len(zv)):
            dark_img = np.zeros(img.shape, dtype=np.float32)
            gamma = xv[j, i, s]
            alpha = yv[j, i, s]
            beta = zv[j, i, s]
            dark_img = beta*((alpha*img)**gamma)
            img_out = cv2.cvtColor(dark_img, cv2.COLOR_BGR2YCR_CB)
            aug_y, bin_edges = np.histogram(dark_img[...,0], bins=256, range=(0, 1))
            res.append(mse(dark_img, target_dark))
            #res.append(mean_squared_error(aug_y[1:], y_dark_hist[1:]))
            res_ind.append([gamma, alpha, beta])
res = np.array(res)
min = res.argmin()
gamma = res_ind[min][0]
alpha = res_ind[min][1]
beta = res_ind[min][2]

#img = cv2.imread(r"E:\dataset\lol\our485\high\591.png") # r"E:\dataset\lol\our485\high\482.png"
#img = np.float32(img/255)
#gamma = 2
#alpha = 0.9
#beta = 0.5

# beta = 1
# gamma = 1.5
dark_img = beta * ((alpha * img) ** gamma)
img_out = cv2.cvtColor(dark_img, cv2.COLOR_BGR2YCR_CB)
aug_y, bin_edges = np.histogram(img_out[..., 0], bins=256, range=(0, 1))
plt.figure()
plt.plot(aug_y)
#plt.plot(y_dark_hist)


blur = cv2.GaussianBlur(dark_img, (3, 3),0)
dark_img = np.float32(uint8(random_noise(dark_img, mode='gaussian', var=0.0001))/255)

plt.figure()
plt.plot(aug_y, color='r')
plt.plot(y_dark_hist)
plotim(uint8(blur))
plotim(uint8(dark_img))
plotim(uint8(img))

plotim(uint8(target_dark))
uinoise_img = target_dark - dark_img
noise_h, bin_edges = np.histogram(noise_img[..., 0], bins=256, range=(0, 1))


plotim(uint8(target_dark) - uint8(dark_img))
[5.0, 0.8, 0.5]
[1.75, 0.8285714285714286, 0.5]
r"E:\dataset\lol\our485\low\481.png"