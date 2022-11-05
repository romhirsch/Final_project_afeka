import cv2
import numpy as np

# load image as YUV (or YCbCR) and select Y (intensity)
# or convert to grayscale, which should be the same.
# Alternately, use L (luminance) from LAB.
img = cv2.imread(r"E:\dataset\MyCoco\test_small\bus\004836.jpg")
Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
L_mean = Y.mean()
# compute min and max of Y
min = np.min(Y)
max = np.max(Y)
dL = max - min
# compute contrast
contrast = (max-min)/(max+min)
print(min, max, contrast)

# contrast retargeting:
Lmin = np.min(Y)
Lmax = np.max(Y)
M = (Lmax-Lmin)/(Lmax+Lmin)

G = 0.5*np.log10(Lmax/Lmin)

G_M = 0.5*np.log10((M+1)/(M-1))
M_G = (np.power(10, 2*G) - 1) / ((np.power(10, 2*G) + 1))