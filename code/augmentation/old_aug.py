import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
import numpy as np
import cv2
import os
from DP.DPhandler import PathDatasets
def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256], color='r')
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256], color='g')
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256], color='b')

def read_noise(image, amount, gain=1):
        shape = image.shape
        noise = np.random.normal(scale=amount / gain, size=shape)
        return noise


class AugmenterHandler:

    def __init__(self, brightness=60, saturation=2,
                 blur=True, contrast=2, noise='gaussian', gamma=False):
        self._brightness_factor = brightness
        self._saturation_factor = saturation
        self._blur = blur
        self._contrast_factor = contrast
        self._noise = noise
        self.gamma = gamma
        if self.gamma:
            self.lugc = np.empty((1, 256), np.uint8)
            for i in range(256):
                self.lugc[0, i] = np.clip(pow(i / (255.0), gamma) * 255.0, 0, 255)
        else:
            self.lugc = False

    def GammaCorr(self, img, gamma):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, lookUpTable)
        return res

    def illumination_augmenter(self, img):
        origImg = img.copy()
        #img = np.float32(img)
        img = self.contrast(img, 0.8)
        img = self.brightness(img)
        img = self.Gamma_correction(img)
        img = self.image_noise(img / 255)
        img = self.uint8(img * 255)
        img = np.squeeze(img)
        img = np.float32(img)
        img = self.saturation(img)
        img = self.contrast(img)
        img = self.blur_img(img)
        img = self.uint8(img)
        return img

    def uint8(self, img):
        return np.clip(img, 0, 255).astype(np.uint8)

    def brightness(self, image):
        if not self._brightness_factor:
            return image
        #return image + np.ones_like(image) * self._brightness_factor
        return image + self._brightness_factor

    def saturation(self, img):
        if not self._saturation_factor:
            return img
        try:
            hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsvImg[:, :, 1] = hsvImg[:, :, 1] * self._saturation_factor
        except:
            print('image failed gray image')
            return img
        return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

    def blur_img(self, image):
        if not self._blur:
            return image
        #blur = cv2.blur(image, (5, 5))
        blur = cv2.bilateralFilter(image, 9, self._blur, self._blur)
        return blur

    def contrast(self, img, cont=False):
        if not self._contrast_factor:
            return img
        if cont:
            return self.uint8(cont * np.float32(img))
        return self.uint8(self._contrast_factor * np.float32(img)) #np.clip(128 + self._contrast_factor * img - self._contrast_factor * 128, 0, 255)



    def image_noise(self, img):
        if not self._noise:
            return img
        return random_noise(img, mode='gaussian', var=self._noise)

    def Gamma_correction(self, img):
        if not self.gamma:
            return img
        return cv2.LUT(img, self.lugc)


class Augmenter_rand(AugmenterHandler):
    def __init__(self, brightness=False, saturation=False,
                 blur=False, contrast=False, noise='gaussian', gamma=False):
        brightness_factor, saturation_factor, contrast_factor = (False, False, False)
        if gamma:
            gamma = np.random.randint(gamma[0]*100, gamma[1]*100) / 100
        if brightness:
           brightness_factor = np.random.randint(brightness[0],brightness[1])/100 * -1# (np.random.randint(brightness[0], brightness[1])/100) * -1
        if saturation:
            saturation_factor = np.random.random() * saturation
        if contrast:
            contrast_factor = max(contrast[0]/100, np.random.random() * contrast[1]/100)
        if blur:
            blur = np.random.randint(blur[0], blur[1])
        super().__init__(brightness=np.float32(brightness_factor),
                        saturation=np.float32(saturation_factor),
                        contrast=np.float32(contrast_factor),
                        blur=blur,
                        noise=noise,
                        gamma=gamma)

    def print_factors(self):
        print('contrast', self._contrast_factor)
        print('brightness', self._brightness_factor)
        print('saturation', self._saturation_factor)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, os.listdir(folder)

def create_dark_images(imgs):
    dark_imgs = []
    for img in imgs:

        aug = Augmenter_rand(noise='gaussian', blur=(0, 300), saturation=False, gamma=(1.5, 5), brightness=False, contrast=False)
        dark_imgs.append(aug.illumination_augmenter(img))
    return dark_imgs


def save_imgs(imgs, list_files, path):
    for img, name in zip(imgs, list_files):
        path_img = os.path.join(path, name)
        cv2.imwrite(path_img, img)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


if __name__ == '__main__':
    img = cv2.imread(r"E:\dataset\lol\our485\high\482.png")
    img_ycr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y_hist, bin_edges = np.histogram(img_ycr[..., 0], bins=256, range=(0, 255))

    target_dark = cv2.imread(r"E:\dataset\lol\our485\low\481.png")
    img_dark_ycr = cv2.cvtColor(target_dark, cv2.COLOR_BGR2YCR_CB)
    y_dark_hist, bin_edges = np.histogram(img_dark_ycr[..., 0], bins=256, range=(0, 255))
    plotim(img)
    gammas = np.linspace(1.1, 5, 100)
    from sklearn.metrics import mean_squared_error
    res = []
    histogram_target, bin_edges = np.histogram(target_dark, bins=256, range=(0, 255))
    for gamma in gammas:
        aug = AugmenterHandler(noise=False, blur=False, saturation=False, gamma=gamma, brightness=False,
                               contrast=False)
        dark_img = aug.illumination_augmenter(img)
        img_out = cv2.cvtColor(dark_img, cv2.COLOR_BGR2YCR_CB)
        aug_y, bin_edges = np.histogram(img_out[...,0], bins=256, range=(0, 255))
        res.append(mean_squared_error(aug_y[0:], y_dark_hist[0:]))


    res = np.array(res)
    min = res.argmin()
    gamma = gammas[min]
    aug = AugmenterHandler(noise=False, blur=False, saturation=False, gamma=gamma, brightness=False,
                           contrast=False)
    dark_img = aug.illumination_augmenter(img)
    img_out = cv2.cvtColor(dark_img, cv2.COLOR_BGR2YCR_CB)
    aug_y, bin_edges = np.histogram(img_out[..., 0], bins=256, range=(0, 255))

    plt.figure()
    plt.plot(aug_y)
    #plt.plot(y_hist)
    plt.plot(y_dark_hist, color='green')
    plotim(target_dark)
    plotim(dark_img)
    # configure and draw the histogram figure
    plt.figure();
    plt.plot(aug.lugc[0, :])
    plt.figure()


    gammas = np.linspace(1.1, 5, 50)
    noises = np.linspace(0, 0.1, 100)
    corr_shift = np.linspace(10, 100, 50)
    blurs = []#np.linspace(0, 300, 100)
    xv, yv = np.meshgrid(gammas, corr_shift)
    res = []
    for i in range(len(gammas)):
        for j in range(len(corr_shift)):
            #for s in range(len(blurs)):
                noise = xv[j, i]
                corr_shift = yv[j, i]
                #blur = yv[j, i]
                aug = AugmenterHandler(noise=noise, blur=50, saturation=False, gamma=gamma, brightness=False,
                                   contrast=0.8)
                dark_img = aug.illumination_augmenter(img)
                res.append(mse(target_dark, dark_img))
    # treat xv[j,i], yv[j,i]
    aug = AugmenterHandler(noise=0.01, blur=50, saturation=False, gamma=2.5, brightness=False,
                         contrast=0.8)

    dark_img = aug.illumination_augmenter(img)
    plotim(dark_img)


    # img = random_noise(img/255, mode='gaussian')
    # img = np.clip(img*255, 0, 255).astype(np.uint8)
    # #img = cv2.blur(img, (5, 5))
    # img = cv2.bilateralFilter(img, 9, 300, 300)
    # plotim(img)
    #
    print(variance_of_laplacian(dark_img))

    print(variance_of_laplacian(target_dark))
    # plotim(blur)
    # blur = cv2.blur(img, (5, 5))
    # variance_of_laplacian(blur)

    save_folder = r"E:\dataset\augmentation_images_small"
    #img = cv2.imread(r"E:\dataset\lol\eval15\high\778.png")
    #ag = Augmenter(brightness=-110, contrast=0.1, noise=False, blur=False, saturation=False)
    #x1 = ag.illumination_augmenter(img)
    ds_folder = PathDatasets.COCO_TEST_SMALL.value
    for i, folder_name in enumerate(os.listdir(ds_folder)):
        print(folder_name)
        folder = os.path.join(ds_folder, folder_name)
        imgs, list_files = load_images_from_folder(folder)
        dark_imgs = create_dark_images(imgs)
        cur_save_folder = os.path.join(save_folder, os.path.basename(folder))
        if not os.path.exists(cur_save_folder):
            os.mkdir(cur_save_folder)
        save_imgs(dark_imgs, list_files, cur_save_folder)


    #
    # aug = Augmenter_rand(noise='gaussian', blur=False, saturation=False)
    # img_dark = aug.illumination_augmenter(img)
    # aug.print_factors()
    # plotim(img_dark)
    # plotim(img)
    # plt.show()

# im2 = illumination_augmenter(im,global_mask=(0, 10), local_mask=(100, 110))
# im2 = im2.reshape(720, 1280, 3)
# im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(im2)