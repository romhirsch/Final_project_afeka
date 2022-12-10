import cv2
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.util.noise import random_noise
from common.common import PathDatasets
from common.common_functions import plot_img, mse, uint8, variance_of_laplacian


def GammaCorr(img, gamma):
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return np.float32(res)/255

def low_light_transform(img, alpha, beta, gamma):
    return beta * GammaCorr(alpha * img, gamma)

def read_noise(img, var):
    return random_noise(img, mode='gaussian', var=var)

def blur(img, size=3):
    return cv2.GaussianBlur(img, (size, size), 0)

def get_Y_hist(img):
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, bin_edges = np.histogram(img_out[..., 0], bins=256, range=(0, 255))
    return y

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

def noise_bayer(img, var=0.0001):
    bayer = bgr_to_bayer(img)
    bayer_noised = random_noise(bayer, mode='gaussian', var=var)
    img_noisy = cv2.cvtColor(uint8(bayer_noised), cv2.COLOR_BAYER_GRBG2BGR)
    return img_noisy

def pair_compare(img, target_dark, gamma, alpha, beta):
    img = np.float32(img)/255
    img_dark = low_light_transform(img, alpha, beta, gamma)
    img_dark = blur(img_dark)
    #img_dark = read_noise(img_dark, 0.0001)
    #img_dark = random_noise(img_dark, mode='gaussian', var=0.0001)
    img_dark = uint8(img_dark*255)
    img = uint8(img*255)
    plot_img(img, 'normal light image')
    plot_img(img_dark, 'Synthesis low-light image')
    plot_img(target_dark, 'references low-light image')
    print('blur estimate target: ', variance_of_laplacian(target_dark))
    print('blur estimate synthesis: ', variance_of_laplacian(img_dark))
    yorig = get_Y_hist(img)
    ydark = get_Y_hist(img_dark)
    yreff = get_Y_hist(target_dark)
    fig, ax = plt.subplots(1)
    ax.set_title('histogram Y channel in YCbCr')
    ax.plot(ydark, color='r', label='Y synthesis')
    ax.plot(yorig, color='g', label='Y orig')
    ax.plot(yreff, color='b', label='Y references')
    ax.set_xlabel('Pixel values')
    ax.set_ylabel('No. of Pixels')
    ax.legend()


def find_params_from_target(img, target_dark, alpha=0.9):
    gammas = np.linspace(1.2, 5, 15)
    betas = np.linspace(0.5, 1, 15)  #
    img = np.float32(img / 255)
    target_dark = np.float32(target_dark / 255)
    res = []
    res_ind = []
    xv, yv = np.meshgrid(gammas, betas)
    for i in range(len(xv)):
        for j in range(len(yv)):
                gamma = xv[j, i]
                beta = yv[j, i]
                dark_img = beta * ((alpha * img) ** gamma)
                res.append(mse(dark_img, target_dark))
                res_ind.append([gamma, alpha, beta])
    res = np.array(res)
    min = res.argmin()
    gamma = res_ind[min][0]
    alpha = res_ind[min][1]
    beta = res_ind[min][2]
    return gamma, alpha, beta


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, os.listdir(folder)


def create_dark_images(imgs, alphas=(0.8, 0.9), betas=(1.5, 7), gammas=(1.5, 7), var_noises=(0.0001, 0.01)):
    dark_imgs = []
    for img in imgs:
        alpha = np.random.uniform(alphas[0], alphas[0])
        beta = np.random.uniform(betas[0], betas[1])
        gamma = np.random.uniform(gammas[0], gammas[1])
        var_noise = np.random.uniform(var_noises[0], var_noises[1])
        img_dark = low_light_transform(np.float32(img)/255, alpha, beta, gamma)
        img_dark = blur(img_dark)
        img_dark = read_noise(img_dark, var_noise)
        dark_imgs.append(np.clip(img_dark*255, 0, 255).astype(np.uint8))
    return dark_imgs


def save_imgs(imgs, list_files, path):
    for img, name in zip(imgs, list_files):
        path_img = os.path.join(path, name)
        cv2.imwrite(path_img, img)

def plot_differ_noise(path):
    img = cv2.imread(path)
    noise_var = np.linspace(0.0001, 0.01, 9)
    fig, ax = plt.subplots(3, 3)
    count=0
    for i in range(3):
        for j in range(3):
            img_dark = read_noise(np.float32(img)/255, noise_var[count])
            img_dark = np.clip(img_dark*255, 0, 255).astype(np.uint8)
            im2 = cv2.cvtColor(np.uint8(img_dark), cv2.COLOR_BGR2RGB)
            ax[i, j].set_title(f'noise var:{round(noise_var[count],7)}')
            ax[i, j].imshow(im2)
            ax[i, j].axis
            ax[i, j].axis('off')
            count+=1


def plot_differ_param(path):
    img = cv2.imread(path)
    plot_img(img)
    alpha = 0.8
    betas = np.linspace(0.5, 1, 3)
    gammas = np.linspace(2, 7, 3)
    xv, yv = np.meshgrid(betas, gammas)
    fig, ax = plt.subplots(xv.shape[0], yv.shape[0])
    for i in range(len(xv)):
        for j in range(len(yv)):
            gamma = yv[j, i]
            beta = xv[j, i]
            img_dark = low_light_transform(np.float32(img) / 255, alpha, beta, gamma)
            img_dark = blur(img_dark)
            img_dark = read_noise(img_dark, 0.0001)
            img_dark = np.clip(img_dark*255, 0, 255).astype(np.uint8)
            im2 = cv2.cvtColor(np.uint8(img_dark), cv2.COLOR_BGR2RGB)
            ax[i, j].set_title(f'gamma: {round(gamma,2)}, beta: {round(beta,2)}')
            ax[i, j].imshow(im2)
            ax[i, j].axis
            ax[i, j].axis('off')


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


if __name__ == '__main__':
    save_folder = r"E:\Imagenet_aug_4"
    # img = cv2.imread(r"E:\dataset\lol\our485\high\481.png")
    # target_dark = cv2.imread(r"E:\dataset\lol\our485\low\481.png")
    # plot_img(target_dark)
    # gamma, alpha, beta = find_params_from_target(img, target_dark)
    # print(gamma, alpha, beta)
    # pair_compare(img, target_dark, gamma, alpha, beta)
    # img_dark = low_light_transform(img, 1, 1, 5)
    #
    # PSNR(img_dark, target_dark)
    # plot_differ_param(r"E:\imagenet_test\bicycle\n02835271_870.JPEG")
    # plot_differ_noise(r"E:\imagenet_test\bicycle\n02835271_870.JPEG")
    ds_folder = PathDatasets.Imagenet_test.value
    alphas = (0.8, 0.8)
    betas = (0.5, 0.6)
    gammas = (7, 9)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for i, folder_name in enumerate(os.listdir(ds_folder)):
        print(folder_name)
        folder = os.path.join(ds_folder, folder_name)
        imgs, list_files = load_images_from_folder(folder)
        dark_imgs = create_dark_images(imgs, alphas, betas, gammas)
        cur_save_folder = os.path.join(save_folder, os.path.basename(folder))
        if not os.path.exists(cur_save_folder):
            os.mkdir(cur_save_folder)
        save_imgs(dark_imgs, list_files, cur_save_folder)





