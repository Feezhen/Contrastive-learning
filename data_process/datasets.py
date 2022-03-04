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
        train_csv_dir = os.path.join(pwd, 'csv')
        test_csv_dir = os.path.join(args.data_dir, '523csv') #测试验证集的路径
        #train
        if self.train:
            data_csv_path = os.path.join(train_csv_dir, '{}_contrastive523.csv'.format(self.args.dataset))
            df = pd.read_csv(data_csv_path)
            self.train_df = df
            self.num_class = self.train_df['label'].max() + 1
        # valid
        elif self.mode == 'valid':
            colname = ['sample1', 'sample2', 'label']
            data_csv_path = os.path.join(test_csv_dir, '{}_pair_valid.csv'.format(self.args.dataset))
            df = pd.read_csv(data_csv_path, names=colname)
            self.valid_df = df
        #test
        elif self.mode == 'test':
            colname = ['sample1', 'sample2', 'label']
            data_csv_path = os.path.join(test_csv_dir, '{}_pair_test.csv'.format(self.args.dataset))
            df = pd.read_csv(data_csv_path, names=colname)
            self.test_df = df
        else:
            colname = ['sample1', 'label']
            data_csv_path = os.path.join(test_csv_dir, '{}_sample_test.csv'.format(self.args.dataset))
            df = pd.read_csv(data_csv_path, names=colname)
            self.test_df = df
        

    def __len__(self):
        if self.train:
            return self.train_df.shape[0]
        elif self.mode == 'valid':
            return self.valid_df.shape[0]
        else:
            return self.test_df.shape[0]

    def __getitem__(self, index):
        if self.train:
            # random_num = np.random.random(1)
            if self.mode != 'contrastive':
                # 使用分类损失
                sample1 = self.train_df['sample1'][index]
                label = int(self.train_df['label'][index])
                # if random_num < 0.5:
                sample1_path = os.path.join(self.args.data_dir, sample1)
                # else:
                #     sample1_path = os.path.join(self.args.data_dir+'_weaken', sample1)
                sample1_img = Image.open(sample1_path).convert('L')
                sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # img.show()
                if self.transform_fcn is not None:
                    img = self.transform_fcn(sample1_img)
                    hist = torch.histc(img, bins=4, min=-1., max=1.)
                    while hist[3] > self.args.img_size[0]*self.args.img_size[1] / 8:
                        img = self.transform_fcn(sample1_img)
                        hist = torch.histc(img, bins=4, min=-1., max=1.)
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
                sample1_img = Image.open(sample1_path).convert('L')
                sample2_img = Image.open(sample2_path).convert('L')
                # orisample1_img = Image.open(sample1_path).convert('L')
                
                sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # orisample1_img = orisample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                sample2_img = sample2_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img.show()
                if self.transform_fcn is not None:
                    img1 = self.transform_fcn(sample1_img)
                    img2 = self.transform_fcn(sample2_img)
                    hist = torch.histc(img1, bins=4, min=-1., max=1.)
                    while hist[3] > self.args.img_size[0]*self.args.img_size[1] / 8:
                        img1 = self.transform_fcn(sample1_img)
                        hist = torch.histc(img1, bins=4, min=-1., max=1.)
                    hist = torch.histc(img2, bins=4, min=-1., max=1.)
                    while hist[3] > self.args.img_size[0]*self.args.img_size[1] / 8:
                        img2 = self.transform_fcn(sample2_img)
                        hist = torch.histc(img2, bins=4, min=-1., max=1.)
                    # sample_path = 'pic/'
                    # self.saveImg(img1, sample_path+'/'+os.path.split(sample1)[1], Gray=True)
                    # self.saveImg(img2, sample_path+'/'+os.path.split(sample2)[1], Gray=True)

                return img1, img2, label
        else:
            if self.mode == 'valid':
                sample1 = self.valid_df['sample1'][index]
                sample2 = self.valid_df['sample2'][index]
                label = int(self.valid_df['label'][index])
                sample1_path = os.path.join(self.args.data_dir, sample1)
                sample2_path = os.path.join(self.args.data_dir, sample2)
                sample1_img = Image.open(sample1_path).convert('L')
                sample2_img = Image.open(sample2_path).convert('L')

                # sample1_img = sample1_img.convert("RGB")
                # sample2_img = sample2_img.convert("RGB")
                sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                sample2_img = sample2_img.resize(self.args.img_size, Image.ANTIALIAS)
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
                sample1_img = Image.open(sample1_path).convert('L')
                sample2_img = Image.open(sample2_path).convert('L')

                sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                sample2_img = sample2_img.resize(self.args.img_size, Image.ANTIALIAS)
                if self.transform_fcn is not None:
                    sample1_img = self.transform_fcn(sample1_img)
                    sample2_img = self.transform_fcn(sample2_img)
                return sample1_img, sample2_img, label

            else :
                #可视化
                sample1 = self.test_df['sample1'][index]
                label = int(self.test_df['label'][index])
                sample1_path = os.path.join(self.args.data_dir, sample1)
                # sample1_path = os.path.join(self.args.data_dir, sample1)
                sample1_img = Image.open(sample1_path).convert('L')

                # sample1_img = sample1_img.convert("RGB")
                sample1_img = sample1_img.resize(self.args.img_size, Image.ANTIALIAS)
                # sample2_img.show()
                if self.transform_fcn is not None:
                    sample1_img = self.transform_fcn(sample1_img)
                return sample1_img, label

    def saveImg(self, image_tensor, save_dir, Gray=False):
        img_array = self.tensor2array(image_tensor)
        image_pil = Image.fromarray(img_array)
        if Gray:
            image_pil.convert('L').save(save_dir)
        else:
            image_pil.save(save_dir)
    
    def tensor2array(self, image_tensor, imtype=np.uint8, normalize=True):
        if isinstance(image_tensor, list):
            image_numpy = []
            for i in range(len(image_tensor)):
                image_numpy.append(self.tensor2array(image_tensor[i], imtype, normalize))
            return image_numpy
        image_numpy = image_tensor.cpu().float().numpy()
        if normalize:
            image_numpy = (np.transpose(image_numpy, (1,2,0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1,2,0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:,:,0]
        return image_numpy.astype(imtype)




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

