from cv2 import data
import torch
import torchvision.transforms as T
import os
import pandas as pd
import cv2
import logging  # 引入logging模块
from PIL import Image
import numpy as np


logging.basicConfig(level=logging.NOTSET)  # 设置日志级别

# from params import Args


def normalization(X):
    '''
    归一化函数，将图像归一化到[-1, 1]
    :param X:
    :return:
    '''
    # print(X)
    return X / 127.5 - 1

'''
class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn=None):
        self.dataset = dataset
        self.csv_path = os.path.join(Args.root_dir, 'csv',  'train_{}.csv'.format(self.dataset))
        self.df = pd.read_csv(self.csv_path) # df是读入的csv文件里的内容
        self.transform_fn = transform_fn

    def __len__(self):
        return self.df.shape[0] # 得到训练样本的数量

    def __getitem__(self, index):
        img = cv2.imread(self.df['img_path'][index]) # 感觉像从相应纵坐标里面索引出对应的行（index）
        img = cv2.resize(img, (128, 128))
        # img = normalization(img)

        PIL_image = Image.fromarray(img)
        img = self.transform_fn(PIL_image)
        # img = torch.tensor(img)
        label = int(self.df['label'][index])
        return img, label # 返回相应的图片以及label
'''

class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn=None):
        self.dataset = dataset  #Tongji_mixsession_0.3
        self.csv_path = os.path.join('csv', 'test_{}.csv'.format(dataset))
        self.df = pd.read_csv(self.csv_path)
        self.transform_fn = transform_fn

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img1 = cv2.imread(self.df['img1_path'][index], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.df['img2_path'][index], cv2.IMREAD_GRAYSCALE) 

        img1 = cv2.resize(img1, (224, 224), cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (224, 224), cv2.INTER_LINEAR)
        PIL_image1 = Image.fromarray(img1)
        PIL_image2 = Image.fromarray(img2)
        # img1 = normalization(img1)
        # img2 = normalization(img2)
        if self.transform_fn is not None:
            img1 = self.transform_fn(PIL_image1)
            img2 = self.transform_fn(PIL_image2)
        label = int(self.df['flag'][index])
        return img1, img2, label

'''
class sne_Dataset(torch.utils.data.Dataset):
    def __init__(self, dir='sne_set.csv', transform_fn=None):
        self.dir = dir
        csv_path = os.path.join(Args.root_dir,'csv',self.dir)
        self.df = pd.read_csv(csv_path)
        self.transform_fn = transform_fn

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img = cv2.imread(self.df['img_path'][index])
        img = cv2.resize(img, (128, 128))
        PIL_image = Image.fromarray(img)
        img = self.transform_fn(PIL_image)
        label = int(self.df['label'][index])
        return img, label
'''

if __name__ == '__main__':
    import torchvision.transforms as transforms
    test_augmentation = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], 
                                 std=[0.5])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            #                         std=[0.5, 0.5, 0.5]) 
        ]
    dataset = Test_Dataset('Tongji_mixsession_0.3', transform_fn=transforms.Compose(test_augmentation))
    print('dataset_len', len(dataset))
    print('img_size', dataset[1][0].shape)