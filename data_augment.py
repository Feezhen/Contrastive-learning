# -*- coding: utf-8 -*-
from PIL import Image, ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x