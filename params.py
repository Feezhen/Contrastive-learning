#!/usr/bin/env python
# coding:utf-8
import math
import os
from sys import flags

from parso import parse
import torch
from torch.backends import cudnn
import logging  # 引入logging模块
import argparse
logging.basicConfig(level=logging.NOTSET)  # 设置日志级别


model_zoo = {
    'Mobile2': 'mobilenetV2',
    'Effnet': 'efficientnet-b0',
    'Res18': 'Resnet18',
    'Res50': 'Resnet50',
    'MobileNetV2m_arccenter': 'MobileNetV2m_arccenter',
}
#数据集名称
dataset_zoo = {
    'SCUTPVV1': 'SCUT_PV_V1',
    'SCUTV2PV': 'SCUT_PPPV_V2_PV',
    'SCUTV2PP': 'SCUT_PPPV_V2_PP',
    # 'TP': 'target_padding',
    'TJ': 'Tongji',
    'PolyMS': 'PolyU_MS',
    'CA_460': 'CASIA_460',
    'CA_850': 'CASIA_850',
    'PUT': 'PUT_PV',
    'vera': 'VERA_PV',
    'c10': 'cifar10',
    'c100': 'cifar100',
}
#数据集路径
datadir_zoo = {
    'TJPV': "/home/data/palm/Tongji_Contactless_Palmvein/ROI_outer",
    'TJPP': "/home/data/palm/Tongji_Contactless_Palmprint/ROI_outer",
    'PolyPP': "/home/data/palm/PolyU2011/PolyU_2011_palmprints_Database/Blue",
    # 'PolyPV': "/home/data/palm/PolyU2011/PolyU_2011_palmprints_Database/NIR",
    'PolyPV': '/home/qyt/new/PolyU/PolyU/NIR',
    'CASIA': "/home/data/palm/CASIA-Multi-Spectral-PalmprintV1/outer_roi",
    'SCUTPVV1': "/home/data/palm/SCUT_PV_V1/ROI_outer",
    'SCUTV2': '/home/data/palm/SCUT_PPPV_V2/roi_outer',
    'PUT': '/home/data/palm/PUT_Vein/VPBase_corrected/Palm',
    'vera': '/home/data/palm/idiap/idiap/vera-palmvein/VERA-Palmvein/roi',
}

def get_args():
    """
    获取传入参数与默认参数
    """
    parser = argparse.ArgumentParser(description='PyTorch Training Params')
    # 传入训练epochs数、metavar只用于help打印帮助信息时起占位符作用
    parser.add_argument('--epochs', default=180, type=int, metavar='N',
                        help='number of total epochs to run')
    # 设置输入图片尺寸
    parser.add_argument('--img_size', default=(224,224),metavar='(a,b)',
                        help='the size of input image')
    # 指定使用的gpu编号
    parser.add_argument('--gpu', default='1', metavar='0',
                        help='the id of gpu')
    # 是否使用多卡训练
    parser.add_argument('--distributed', default=False, type=bool, metavar="True or False",
                        help='distributed')
    # 重启训练时从哪个epoch开始训练
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    # 测试频率
    parser.add_argument('--test_freq', default=10, type=int, metavar='N',
                        help='test freq')
    # 平衡采样
    parser.add_argument('--balance_sample', default=True, type=bool, metavar='N',
                        help='test freq')
    # ckpt保存频率
    parser.add_argument('--save_freq', default=50, type=int, metavar='N',
                        help='test freq')
    # batch-size大小
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    # 设置学习率
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, #.01 .02
                        metavar='LR', help='initial learning rate', dest='lr')
    # 梯度截断
    parser.add_argument('--clip', default=10, type=float,
                        help='gradient clip')
    # 设置warmup
    parser.add_argument('--warm', default=False, type=bool,
                        help='warm-up for large batch training')
    parser.add_argument('--base_lr', '--base_learning_rate', default=0.0001, type=float,
                        metavar='baseLR', help='initial learning rate')
    parser.add_argument('--warm_epochs', default=20, type=int,
                        metavar='warmup_epoch', help='Number of Warmup Epochs During Contrastive Training.')
    # 数据集路径
    parser.add_argument('--data_dir', default=datadir_zoo['TJPV'], 
                        type=str, metavar='dir', help='the path of src files')
    # 数据集名称
    parser.add_argument('--dataset', default=dataset_zoo['TJ'], type=str,
                        metavar='name', help='the name of dataset')
    
    # cos学习率衰减策略或指数衰减
    parser.add_argument('--cos', action="store_true", help='use cos learning rate')
    
    # 是否使用数据扩增
    parser.add_argument('--aug', default=True, type=bool,
                        metavar='True/False', help='Use data augment or not.True or False')
    # 日志保存路径
    parser.add_argument('--log_dir', default='./log', type=str,
                        metavar='dir', help='the path of csv files')
    # 断点续训
    parser.add_argument('--resume', default=False, type=bool, metavar='BOOL',
                        help='Use ckpt or not')
    # 训练模型
    parser.add_argument('--model', default=model_zoo['Mobile2'], type=str,
                        metavar='model', help='choose a model')
    parser.add_argument('--ModelMetric', default='', type=str,
                        help='model fc layer')
    # SGD动量
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD权值衰减
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    #高斯模糊参数
    parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 1.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
    # NCE loss的困难样本权重
    parser.add_argument('--focal', type=str, default='',
                        help='choose to use which loss function')
    parser.add_argument('--gamma', default=0.5, type=float, metavar='gamma',
                        help='focal gamma')
    parser.add_argument('--keep_w', default=.1, type=float, metavar='keep_weight',
                        help='focal keep weight')
    # 随机种子
    parser.add_argument('--seed', default=44, type=int, metavar='seed',
                        help='random seed')

    args = parser.parse_args()
    # if args.batch_size > 256:
    args.warm = True
    if args.warm:
        args.warmup_from = .0005
        # args.warm_epochs = 30
        # if args.cos:
        #     eta_min = args.lr * (args.lr_decay_rate ** 3)
        #     args.warmup_to = eta_min + (args.lr - eta_min) * (
        #         1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        # else:
        args.warmup_to = args.lr
    
    if args.dataset == 'cifar10':
        args.n_cls = 10
    elif args.dataset == 'cifar100':
        args.n_cls = 100
    return args


if __name__ == "__main__":
    # 指定使用的gpu
    args = get_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 检测是否可以使用gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("using gpu: " + args.gpu)
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print(xx)
