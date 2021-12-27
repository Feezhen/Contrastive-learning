#!/usr/bin/env python
# coding:utf-8
import numpy as np
import torch
import os
import shutil
import params
import matplotlib.pyplot as plt
import params
import torchvision.transforms as transforms
import logging
import random

from sklearn import manifold
from model.mobilenetV2 import MobileNet_v2
from PIL import Image
from data_process.datasets import FVDataset
from data_process.online_datasets import Test_Dataset
from utils import cos_calc_eer, l2_calc_eer, batch_l2_distance
from tqdm import tqdm


def test():
    '''
    模型测试
    '''
    # 保存参数
    args = params.get_args()
    # 指定使用的gpu
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('-----Using GPU: {}-----'.format(args.gpu))
    test_augmentation = [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            #                         std=[0.5, 0.5, 0.5]) 
            transforms.Normalize(mean=[0.5], 
                                    std=[0.5]) 
        ]
    # train_augmentation = [
    #         transforms.RandomApply([
    #             transforms.RandomChoice([
    #                 transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.3), # 随机亮度、对比度、饱和度、色调抖动
    #                 transforms.RandomPerspective(distortion_scale=0.05, p=0.5),   # 随机透射变换     ,这里只有透射变换是有概率发生的
    #                 transforms.RandomAffine(degrees=5, translate=(0.001,0.005)),  # 随机仿射变换
    #                 transforms.RandomRotation(3),     # 随机旋转
    #             ])
    #         ],p=0.5),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.5], 
    #                             #  std=[0.5]) 
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], 
    #                              std=[0.5, 0.5, 0.5]) 
    #     ]
    transform_fcn = transforms.Compose(test_augmentation)
    
    # 加载模型
    load_mode = 'ckpt'
    model_name = 'mobilenetV2'
    dataset_name = 'PUT_PV'
    MeasurementOptions = 'cos'
    log_path = './log/mobilenetV2_PUT_PV_20211217-2059_contrastive_lr0.0025_batchsize32/'

    if load_mode == 'model': #加载model
        model_path = 'log/mobilenetV2_target_padding_20210309-1536_classified/{}_{}_model.pth'.format(model_name, dataset_name)
        model = torch.load(model_path).cuda()
    else: #加载ckpt
        model = MobileNet_v2(num_classes=300, img_size=args.img_size, in_channel=1).cuda()
        ckpt_path = log_path+'checkpoint_{}_{}_{}_contrastive.pth.tar'\
                    .format(model_name, dataset_name, MeasurementOptions)
        if os.path.isfile(ckpt_path):
            logging.info("=> loading checkpoint '{}'".format(ckpt_path))
            if args.gpu is None:
                checkpoint = torch.load(ckpt_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(0)
                checkpoint = torch.load(ckpt_path, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("=> no checkpoint found at '{}'".format(ckpt_path))
    test_dataset = FVDataset(train=False, mode='divide', args=args, transform_fcn=transform_fcn)
    # 创建测试数据迭代器
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                    batch_size=1, shuffle=False,
                                    num_workers=4, pin_memory=True)
    # 加载模型
    
    model.eval()
    featureList = []
    labels = []
    with torch.no_grad():
        for batch_idx, (image1, label) in enumerate(tqdm(test_loader)):
            image1 = image1.cuda().float()
            # image2 = image2.cuda().float()
            # # 如果使用opencv扩增，加上如下维度变换
            # image1 = image1.permute(0, 3, 1, 2)
            # image2 = image2.permute(0, 3, 1, 2)

            label = label.cuda()
            _, feature1 = model(image1, label)
            # _, feature2 = model(image2)

            # 保存所有特征和label
            featureList.append(feature1)
            labels.append(label)
        # 转numpy

        feature = torch.cat(featureList).cpu().numpy()
        label = torch.cat(labels).cpu().numpy()
        # labels_mask = (label < 10)
        # feature = feature[labels_mask]
        # label = label[labels_mask]

        print('Computing t-SNE embedding')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(feature)

        fig = plot_embedding(result, label, 'mobilenetV2_PUT_PV_contrastive')
        dir = os.path.join(log_path, 'tsne.jpg')
        plt.savefig(dir)

def plot_embedding(data, label, title):
    plt.switch_backend('agg')
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    label = np.append(label, 0)
    cl = '#'+''.join([random.choice("0123456789ABCDEF") for i in range(6)])  # 随机选色
    # cl_list = ['#FF0000','#FFFF00','#0000FF']
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], '.',
                    color=cl)  # markersize=3  color=plt.cm.Set1(label[i])
        if label[i+1] != label[i]:
            cl = '#'+''.join([random.choice("0123456789ABCDEF") for j in range(6)])

        # plt.plot(data[i, 0], data[i, 1], '.',
        #          color=cl_list[int(label[i])-1])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def test_Tongji():
    args = params.get_args()
    # 指定使用的gpu
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('-----Using GPU: {}-----'.format(args.gpu))
    test_augmentation = [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            #                         std=[0.5, 0.5, 0.5]) 
            transforms.Normalize(mean=[0.5], 
                                    std=[0.5]) 
        ]
    # 加载模型
    load_mode = 'ckpt'
    model_name = args.model
    dataset_name = args.dataset
    MeasurementOptions = 'cos'
    log_path = './log/mobilenetV2_Tongji_20210419-2254_contrastive_lr0.0025_batchsize64/'

    if load_mode == 'model': #加载model
        model_path = 'log/mobilenetV2_target_padding_20210309-1536_classified/{}_{}_model.pth'.format(model_name, dataset_name)
        model = torch.load(model_path).cuda()
    else: #加载ckpt
        model = MobileNet_v2(num_classes=360, img_size=args.img_size, in_channel=1).cuda()
        ckpt_path = log_path+'checkpoint_{}_{}_{}_contrastive.pth.tar'\
                    .format('mobilenetV2', 'Tongji', MeasurementOptions)
        if os.path.isfile(ckpt_path):
            logging.info("=> loading checkpoint '{}'".format(ckpt_path))
            if args.gpu is None:
                checkpoint = torch.load(ckpt_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(0)
                checkpoint = torch.load(ckpt_path, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("=> no checkpoint found at '{}'".format(ckpt_path))
    test_dataset = Test_Dataset(dataset='Tongji_mixsession_0.3', transform_fn=transforms.Compose(test_augmentation))
    # 创建测试数据迭代器
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                    batch_size=16, shuffle=False,
                                    num_workers=4, pin_memory=True)

    model.eval()
    distances_cos = []
    distances_l2 = []
    labels = []
    eer = 1
    tqdm_batch = tqdm(test_loader)
    with torch.no_grad():
        for i, (image1, image2, label) in enumerate(tqdm_batch):
            image1 = image1.cuda().float()
            image2 = image2.cuda().float()
            # # 如果使用opencv扩增，加上如下维度变换
            # image1 = image1.permute(0, 3, 1, 2)
            # image2 = image2.permute(0, 3, 1, 2)

            label = label.cuda()
            _, feature1 = model(image1)
            _, feature2 = model(image2)
            # feature1 = model(image1)
            # feature2 = model(image2)
            # feature1 = feature1.view(-1, 1280)
            # feature2 = feature2.view(-1, 1280)

            # 计算余弦距离
            if MeasurementOptions == 'cos':
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                distances_cos.append(cos(feature1, feature2))
            # 计算欧氏距离
            else:
                distances_l2.append(batch_l2_distance(feature1, feature2))
            labels.append(label)

        # 将所有batch的distance矩阵拼在一起
        if MeasurementOptions == 'cos':
            distances_cos = torch.cat(distances_cos)
        else:
            distances_l2 = torch.cat(distances_l2)
        # last_eer = eer
        # 将所有batch的label也拼在一起
        label = torch.cat(labels)
        # 计算等误率
        if MeasurementOptions == 'cos':
            eer, bestThresh, minV = cos_calc_eer(distances_cos, label, log_path, 0, None)
        else:
            eer, bestThresh, minV = l2_calc_eer(distances_l2, label, log_path, 0)
        print('eer {:.6f}, thres {:.6f}'.format(eer, bestThresh))
    tqdm_batch.close()

if __name__ == '__main__':
    test()
    # test_Tongji()
    