#!/usr/bin/env python
# coding:utf-8
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision import utils as vutils
import os
import numpy as np
import pandas as pd
import cv2
import logging  # 引入logging模块
import params
from PIL import Image
from data_process import imageaug
from utils import mkdir

logging.basicConfig(level=logging.NOTSET)  # 设置日志级别

def normalization(X):
    '''
    归一化函数，将图像归一化到[-1, 1]
    :param X:
    :return:
    '''
    # print(X)
    return X / 127.5 - 1


class FVDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, mode='classified', args=params.get_args(), transform_fcn=None):
        self.train = train
        self.mode = mode
        self.args = args
        self.transform_fcn = transform_fcn

        pwd = os.getcwd()
        csv_dir = os.path.join(pwd, 'csv')
        #train
        data_csv_path = os.path.join(csv_dir, '{}_{}.csv'.format(self.args.dataset, self.mode))
        df = pd.read_csv(data_csv_path)
        self.train_df = df
        # valid
        data_csv_path = os.path.join(csv_dir, '{}_{}.csv'.format(self.args.dataset, 'valid'))
        df = pd.read_csv(data_csv_path)
        self.valid_df = df
        #test
        data_csv_path = os.path.join(csv_dir, '{}_{}.csv'.format(self.args.dataset, 'test'))
        df = pd.read_csv(data_csv_path)
        self.test_df = df
        # self.test_df.index = self.test_df.index - self.train_df.shape[0]
        # if mode == 'classified':
        #     self.num_class = self.train_df['label'].max()
        # else:
        self.num_class = self.train_df['label'].max() + 1
        # self.saveImgpath = './ramdomcrop'
        # mkdir(self.saveImgpath)

    def __len__(self):
        if self.mode != 'valid' and self.mode != 'test':
            return self.train_df.shape[0]
        elif self.mode == 'valid':
            return self.valid_df.shape[0]
        else:
            return self.test_df.shape[0]

    def __getitem__(self, index):
        if self.train:
            # random_num = np.random.random(1)
            if self.mode != 'contrastive':
                # 不使用对比方法训练
                sample1 = self.train_df['sample1'][index]
                label = int(self.train_df['label'][index])
                # if random_num < 0.5:
                sample1_path = os.path.join(self.args.data_dir, sample1)
                # else:
                #     sample1_path = os.path.join(self.args.data_dir+'_weaken', sample1)
                img = Image.open(sample1_path)
                # img.show()
                # if args.model != 'efficientnet-b0':
                #     img = img.convert("RGB")
                # img = img.resize(self.args.img_size, Image.ANTIALIAS)
                # img.show()
                if self.transform_fcn is not None:
                    img = self.transform_fcn(img)
                return img, label
            else:
                # 使用对比方法训练
                sample1 = self.train_df['sample1'][index]
                sample2 = self.train_df['sample2'][index]
                label = int(self.train_df['label'][index])
                # print('random_num   ', random_num)
                # if random_num < 0.5: #增强图片
                sample1_path = os.path.join(self.args.data_dir, sample1)
                sample2_path = os.path.join(self.args.data_dir, sample2)
                # else: #弱化图片
                #     sample1_path = os.path.join(self.args.data_dir+'_weaken', sample1)
                #     sample2_path = os.path.join(self.args.data_dir+'_weaken', sample2)
                sample1_img = Image.open(sample1_path)
                sample2_img = Image.open(sample2_path)
                # if args.model != 'efficientnet-b0':
                # sample1_img = sample1_img.convert("RGB")
                # sample2_img = sample2_img.convert("RGB")
                
                # sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img = sample2_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img.show()
                if self.transform_fcn is not None:
                    sample1_img = self.transform_fcn(sample1_img)
                    sample2_img = self.transform_fcn(sample2_img)
                    # sample_path = '/home/hhz/Pytorch_prj/Contrastive2/sample_image/'
                    # PILsample1_img = ToPILImage()(sample1_img)
                    # PILsample2_img = ToPILImage()(sample2_img)
                    # PILsample1_img.save(sample_path+'{}_1.bmp'.format(index))
                    # PILsample2_img.save(sample_path+'{}_2.bmp'.format(index))
                    # vutils.save_image(sample1_img, os.path.join(self.saveImgpath, sample1), normalize=False)
                return sample1_img, sample2_img, label
        else:
            if self.mode == 'valid':
                sample1 = self.valid_df['sample1'][index]
                sample2 = self.valid_df['sample2'][index]
                label = int(self.valid_df['label'][index])
                sample1_path = os.path.join(self.args.data_dir, sample1)
                sample2_path = os.path.join(self.args.data_dir, sample2)
                sample1_img = Image.open(sample1_path)
                sample2_img = Image.open(sample2_path)

                # sample1_img = sample1_img.convert("RGB")
                # sample2_img = sample2_img.convert("RGB")
                # sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img = sample2_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img.show()
                if self.transform_fcn is not None:
                    sample1_img = self.transform_fcn(sample1_img)
                    sample2_img = self.transform_fcn(sample2_img)
                return sample1_img, sample2_img, label

            elif self.mode == 'test':
                sample1 = self.test_df['sample1'][index]
                sample2 = self.test_df['sample2'][index]
                label = int(self.test_df['label'][index])
                sample1_path = os.path.join(self.args.data_dir, sample1)
                sample2_path = os.path.join(self.args.data_dir, sample2)
                sample1_img = Image.open(sample1_path)
                sample2_img = Image.open(sample2_path)

                # sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img = sample2_img.resize(self.args.img_size, Image.ANTIALIAS)
                if self.transform_fcn is not None:
                    sample1_img = self.transform_fcn(sample1_img)
                    sample2_img = self.transform_fcn(sample2_img)
                return sample1_img, sample2_img, label

            else :
                sample1 = self.test_df['sample1'][index]
                label = int(self.test_df['label'][index])
                sample1_path = os.path.join(self.args.data_dir, sample1)
                # sample1_path = os.path.join(self.args.data_dir, sample1)
                sample1_img = Image.open(sample1_path)

                # sample1_img = sample1_img.convert("RGB")
                # sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img.show()
                if self.transform_fcn is not None:
                    sample1_img = self.transform_fcn(sample1_img)
                return sample1_img, label #可视化的话，只返回一张图


if __name__ == '__main__':
    # pwd = os.getcwd()
    # logging.info(pwd)
    args = params.get_args()
    train_dataset = FVDataset(train=True, args=args, mode='classified')
    test_dataset = FVDataset(train=False, args=args, mode='valid')

    logging.info('shape of train_dataset: {}'.format(len(train_dataset)))
    logging.info('shape of test dataset: {}'.format(len(test_dataset)))
    # for i in range(20):
    #     print(test_dataset[i])
    #     print(train_dataset[i])
    #     cv2.waitKey()
    # print(test_dataset[100])
    # data1, data2, label = train_dataset[0]
    print(train_dataset[0][0].size)
    print(test_dataset[0][0].size)
    print(train_dataset.num_class)

