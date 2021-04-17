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
from datasets import FVDataset
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
    train_augmentation = [
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.3), # 随机亮度、对比度、饱和度、色调抖动
                    transforms.RandomPerspective(distortion_scale=0.05, p=0.5),   # 随机透射变换     ,这里只有透射变换是有概率发生的
                    transforms.RandomAffine(degrees=5, translate=(0.001,0.005)),  # 随机仿射变换
                    transforms.RandomRotation(3),     # 随机旋转
                ])
            ],p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], 
                                #  std=[0.5]) 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                 std=[0.5, 0.5, 0.5]) 
        ]
    transform_fcn = transforms.Compose(test_augmentation)
    
    # 加载模型
    load_mode = 'ckpt'
    model_name = args.model
    dataset_name = args.dataset
    MeasurementOptions = 'cos'
    log_path = './log/mobilenetV2_dataBase_roi_2_padding1_20210330-1801_contrastive_lr0.025_batchsize128/'

    if load_mode == 'model': #加载model
        model_path = 'log/mobilenetV2_target_padding_20210309-1536_classified/{}_{}_model.pth'.format(model_name, dataset_name)
        model = torch.load(model_path).cuda()
    else: #加载ckpt
        model = MobileNet_v2(num_classes=240, img_size=args.img_size, in_channel=1).cuda()
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
            _, feature1 = model(image1)
            # _, feature2 = model(image2)

            # 保存所有特征和label
            featureList.append(feature1)
            labels.append(label)
        # 转numpy
        feature = torch.cat(featureList).cpu().numpy()
        label = torch.cat(labels).cpu().numpy()

        print('Computing t-SNE embedding')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(feature)

        fig = plot_embedding(result, label, 'dataBase_roi_2_padding1_20210330-1801_contrastive')
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
    

if __name__ == '__main__':
    test()
    