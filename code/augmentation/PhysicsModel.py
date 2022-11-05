import numpy as np
import cv2
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256], color='r')
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256], color='g')
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256], color='b')
# Np photon shot noise

img = cv2.imread(r"E:\dataset\lol\our485\high\482.png")
Io = cv2.imread(r"E:\dataset\lol\our485\high\482.png")
plotim(img)
target_dark = cv2.imread(r"E:\dataset\lol\our485\low\481.png")
#Io = cv2.imread(r"E:\dataset\MyCoco\test\boat\004630.jpg")
(height, width) = Io.shape[:2]
(B,G,R) = cv2.split(Io)
bayer = np.empty((height, width), np.uint8)

# strided slicing for this pattern:
#   G R
#   B G
bayer[0::2, 0::2] = G[0::2, 0::2] # top left
bayer[0::2, 1::2] = R[0::2, 1::2] # top right
bayer[1::2, 0::2] = B[1::2, 0::2] # bottom left
bayer[1::2, 1::2] = G[1::2, 1::2] # bottom right

b = cv2.cvtColor(bayer, cv2.COLOR_BAYER_GRBG2BGR)
gnoise = random.poisson(lam=2, size=10)
mu, sigma = 0, 0.1
# creating a noise with the same dimension as the dataset (2,2)
noise = np.random.normal(mu, sigma, bayer.shape)
noisy = bayer/255 + noise
b = cv2.cvtColor(np.clip(noisy*255, 0, 255).astype(np.uint8), cv2.COLOR_BAYER_GRBG2BGR)

Io.shape[0]
alpha = np.random.uniform(0.9, 1)
beta = np.random.uniform(0.5, 1)
gamma = np.random.uniform(1.5, 5)
Iu = np.zeros(Io.shape)
Io = np.float32(Io/255)
for i in range(Io.shape[0]):
    for j in range(Io.shape[1]):

        Iu[i, j, :] = beta * ((alpha * Io[i, j, :]) ** gamma)



#Iu = beta * (alpha*Io)**gamma
Iu = np.clip(Iu*255, 0, 255).astype(np.uint8)
Io = np.clip(Io*255, 0, 255).astype(np.uint8)
plotim(Io)
plotim(Iu)
plotim(target_dark)
img_out = cv2.cvtColor(Io, cv2.COLOR_BGR2YUV)
fig, ax = plt.subplots(2, 1)
ax[0].imshow(img_out)
ax[1].hist(img_out[..., 0].flatten(), 256, [0, 256], color='y')

img_out = cv2.cvtColor(Iu, cv2.COLOR_BGR2YCR_CB)
fig, ax = plt.subplots(2, 1)
ax[0].imshow(img_out)
ax[1].hist(img_out[..., 0].flatten(), 256, [0, 256], color='y')

img_out = cv2.cvtColor(target_dark, cv2.COLOR_BGR2YCR_CB)
fig, ax = plt.subplots(2, 1)
ax[0].imshow(img_out)
ax[1].hist(img_out[..., 0].flatten(), 256, [0, 256], color='y')