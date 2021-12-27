# -*- coding: utf-8 -*-
from PIL import Image, ImageFilter
import random
import cv2
import skimage
from skimage import exposure
import numpy as np

class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def random_rotation(image, angle_range=5):
    height, width = image.shape[:2]
    random_angle = np.random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), random_angle, 1)
    image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return image


def random_shift(image, wrg, hrg):
    height, width = image.shape[:2]
    tx = np.random.uniform(-wrg, wrg) * width
    ty = np.random.uniform(-hrg, hrg) * height
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (width, height))
    return image


def random_zoom(image, zoom_range):
    height, width = image.shape[:2]
    zx, zy = 1 + np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0]])
    image = cv2.warpAffine(image, zoom_matrix, (int(width * zy), int(height * zx)))
    new_h, new_w = image.shape[:2]

    x_range = new_h - height
    y_range = new_w - width

    x_start = np.random.randint(x_range) if x_range > 0 else 0
    y_start = np.random.randint(y_range) if y_range > 0 else 0

    image = image[x_start:x_start + height, y_start:y_start + width]

    return image


def random_shear(image, intensity):
    height, width = image.shape[:2]

    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0]])

    image = cv2.warpAffine(image, shear_matrix, (width, height))
    return image


def random_channel_shift(image, intensity=0.1, channel_axis=2):
    x = np.rollaxis(image, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x.astype(np.uint8)


def adjust_gamma(image, gamma=0.05, gain=1):
    gamma = np.random.uniform(1 - gamma, 1 + gamma, 1)[0]
    image = exposure.adjust_gamma(image, gamma)
    #     image = ((image / 256) ** gamma) * 256 * gain
    return image


def imageaug(image):
    # 生成随机概率
    choice = np.random.choice(a=[0,1], size=6, replace=True, p=None)
    if choice[0]:
        image = adjust_gamma(image, gamma=0.03)
    if choice[1]:
        image = random_shear(image, intensity=0.02)
    if choice[2]:
        image = random_shift(image, wrg=0.04, hrg=0.08)
    if choice[3]:
        image = random_rotation(image, angle_range=1)
    if choice[4]:
        image = random_zoom(image, (0, 0.05))
    if choice[5]:
        image = random_channel_shift(image, 0.5)

    return image

if __name__ == "__main__":
    image = cv2.imread()
    imageaug(image)