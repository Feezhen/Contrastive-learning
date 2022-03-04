#!/usr/bin/env python
# coding:utf-8
from unittest import result
from ast import arg
from cv2 import log
import numpy as np
import sklearn
import torch
import os
import shutil
import params
import matplotlib.pyplot as plt
import params
import torchvision.transforms as transforms
import logging
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

from model.mobilenetV2 import MobileNet_v2
from data_process.datasets import FVDataset
from data_process.online_datasets import Test_Dataset
from utils import cos_calc_eer, l2_calc_eer, batch_l2_distance

clist = ['#BB31F9', '#61BC77', '#F08AFF', '#C7C172', '#EBF2BF', '#2577BA', '#100BAB',
         '#EC0EEE', '#9ACB94', '#756B1F', '#E87911', '#7352E0', '#24898C', '#326580', 
         '#225107', '#91996F', '#529B07', '#2E7321', '#CA8881', '#B1259E', '#4A1271', 
         '#C7678A', '#A059BF', '#EE75C5', '#8BF45D', '#BC0436', '#D6D579', '#9BA5A3', 
         '#9DD6D2', '#5E1C30', '#BEBE50', '#9E643C', '#13E72A', '#5ECA13', '#1249A6', 
         '#D8072F', '#738D1C', '#46D3C0', '#E17923', '#560A0D', '#6FCEFC', '#0F3F3B', 
         '#3F1B08', '#037494', '#DE713B', '#EE15C0', '#6CD475', '#05EF3B', '#8B90C5', '#F8BC65']

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
    model_name = args.model
    dataset_name = args.dataset
    MeasurementOptions = 'cos'
    # method = 'contrastive'
    method = 'classified'
    log_path = './log/mobilenetV2_SCUT_PPPV_V2_PV_20220223-100712_classified_lr0.04_batchsize64/'

    if load_mode == 'model': #加载model
        model_path = 'log/mobilenetV2_target_padding_20210309-1536_classified/{}_{}_model.pth'.format(model_name, dataset_name)
        model = torch.load(model_path).cuda()
    else: #加载ckpt
        model = MobileNet_v2(args, num_classes=50, img_size=args.img_size, in_channel=1).cuda()
        ckpt_path = log_path+'checkpoint_{}_{}_{}_{}.pth.tar'\
                    .format(model_name, dataset_name, MeasurementOptions, method)
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
            if(label[0] > 50):
                break
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
        # print("feature: "+'-'*30)
        # print(feature[0][0], " ", feature[0][1], feature[0][2])
        # print(feature[1][0], " ", feature[1][1], feature[1][2])
        # labels_mask = (label < 10)
        # feature = feature[labels_mask]
        # label = label[labels_mask]

        dis_metric = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean', 'mahalanobis', 'seuclidean', 
                    'sqeuclidean', 'minkowski']

        print('Computing t-SNE embedding')
        pca = PCA(n_components=50)
        result = pca.fit_transform(feature)
        for dis in dis_metric:
        # dis = 'cosine'
            tsne = manifold.TSNE(n_components=2, random_state=2022, metric=dis)
            result_tsne = tsne.fit_transform(result)

        
            fig = plot_embedding(result_tsne, label, dis)
            # # pic name
            # dir = os.path.join(log_path, f'{dataset_name}_contrastive_{dis}.jpg')
            dir = os.path.join(log_path, f'{dataset_name}_classify{args.ModelMetric}_{dis}.jpg')
            plt.savefig(dir, bbox_inches='tight')

def plot_embedding(data, label, title=None):
    plt.switch_backend('agg')
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    x_min, x_max = data.min(0), data.max(0)
    # 二维
    data = (data - x_min) / (x_max - x_min)
    # 三维
    # data = data / (x_max - x_min)

    # 二维
    fig = plt.figure()
    label = np.append(label, 0)
    
    # cl = '#'+''.join([random.choice("0123456789ABCDEF") for i in range(6)])  # 随机选色
    # cl_list = ['#FF0000','#FFFF00','#0000FF']
    x = []
    y = []
    l = 1
    # 不错的设置 2022-2-28
    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], '.', color=plt.cm.Set1(label[i]),
        #             fontdict={'size': 9}, label=f'c{label[i]}')
        x.append(data[i, 0])
        y.append(data[i, 1])
        if label[i+1] != label[i]:
            plt.plot(x, y, '.',
                    color=clist[(l-1)%50], label=f'c{l}')  # markersize=3  color=plt.cm.Set1(label[i])
            # cl = '#'+(''.join([random.choice("0123456789ABCDEF") for j in range(3)]))*2
            l += 1
            x.clear()
            y.clear()

        # plt.plot(data[i, 0], data[i, 1], '.',
        #          color=cl_list[int(label[i])-1])
    plt.xticks([])
    plt.yticks([])
    # 三维
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2],
    #             c=plt.cm.Set1(label))
    # plt.axis('off')
    # 不错的设置 2022-2-28
    # plt.legend(bbox_to_anchor=(1.04, 0), ncol=6)
    # plt.show()
    plt.title(title)
    return fig



if __name__ == '__main__':
    test()
    # cllist = []
    # for i in range(50):
    #     cl = "#" + "".join([random.choice("0123456789ABCDEF") for i in range(6)])
    #     cllist.append(cl)
    # print(cllist)
    