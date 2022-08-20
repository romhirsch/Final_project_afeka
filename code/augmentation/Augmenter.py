import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256])

class Augmenter:

    def __init__(self, brightness=60, saturation=2,
                 blur=True, contrast=2, noise='gaussian'):
        self._brightness_factor = brightness
        self._saturation_factor = saturation
        self._blur = blur
        self._contrast_factor = contrast
        self._noise = noise

    def GammaCorr(self, img, gamma):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, lookUpTable)
        return res

    def illumination_augmenter(self, img):
        origImg = img.copy()
        img = np.squeeze(img)
        img = np.float32(img)
        img = self.saturation(img)
        img = self.blur_img(img)
        img = self.contrast(img)
        img = self.brightness(img)
        img = self.image_noise(img/255)
        img = self.uint8(img*255)
        return img

    def uint8(self, img):
        return np.clip(img, 0, 255).astype(np.uint8)

    def brightness(self, image):
        if not self._brightness_factor:
            return img
        return image + image * self._brightness_factor

    def saturation(self, img):
        if not self._saturation_factor:
            return img
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsvImg[:, :, 1] = hsvImg[:, :, 1] * self._saturation_factor
        return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    def blur_img(self, image):
        if not self._blur:
            return image
        return cv2.blur(image, (5, 5))

    def contrast(self, img):
        if not self._contrast_factor:
            return img
        return np.clip(128 + self._contrast_factor * img - self._contrast_factor * 128, 0, 255)

    def image_noise(self, img):
        if not self._noise:
            return img
        return random_noise(img, mode=self._noise)

class Augmenter_rand(Augmenter):
    def __init__(self, brightness=(79, 80), saturation=1.3,
                 blur=True, contrast=(56, 57), noise='gaussian'):
        brightness_factor, saturation_factor, contrast_factor = (False, False, False)
        if brightness:
           brightness_factor = (np.random.randint(brightness[0], brightness[1])/100) * -1
        if saturation:
            saturation_factor = np.random.random() * saturation
        if contrast:
            contrast_factor = max(contrast[0]/100, np.random.random() * contrast[1]/100)
        super().__init__(brightness=np.float32(brightness_factor),
                        saturation=np.float32(saturation_factor),
                        contrast=np.float32(contrast_factor),
                        blur=blur,
                        noise=noise)

    def print_factors(self):
        print('contrast', self._contrast_factor)
        print('brightness', self._brightness_factor)
        print('saturation', self._saturation_factor)


if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\rom21\OneDrive\Desktop\git_project\code\dataset\lol\eval15\high\778.png")
    ag = Augmenter(brightness=-110, contrast=0.1, noise=False, blur=False, saturation=False)
    x1 = ag.illumination_augmenter(img)
    aug = Augmenter_rand(img, noise=False, blur=False, saturation=False)
    img_dark = aug.illumination_augmenter(img)
    aug.print_factors()
    plotim(img_dark)
    plotim(img)
    plt.show()

pass
# im2 = illumination_augmenter(im,global_mask=(0, 10), local_mask=(100, 110))
# im2 = im2.reshape(720, 1280, 3)
# im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(im2)