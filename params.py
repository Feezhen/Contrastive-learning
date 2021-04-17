#!/usr/bin/env python
# coding:utf-8
import math
import os
from sys import flags
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
}
dataset_zoo = {
    'DR2P': 'dataBase_roi_2_padding1',
    'TP': 'target_padding',
    'c10': 'cifar10',
    'c100': 'cifar100',
}

def get_args():
    """
    获取传入参数与默认参数
    """
    parser = argparse.ArgumentParser(description='PyTorch Training Params')
    # 传入训练epochs数、metavar只用于help打印帮助信息时起占位符作用
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    # 设置输入图片尺寸
    parser.add_argument('--img_size', default=(224,224),metavar='(a,b)',
                        help='the size of input image')
    # 指定使用的gpu编号
    parser.add_argument('--gpu', default='2', metavar='0',
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
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    # 设置学习率
    parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    # 设置warmup
    parser.add_argument('--warm', default=False, type=bool,
                        help='warm-up for large batch training')
    parser.add_argument('--base_lr', '--base_learning_rate', default=0.0001, type=float,
                        metavar='baseLR', help='initial learning rate')
    parser.add_argument('--warm_epochs', default=0, type=int,
                        metavar='warmup_epoch', help='Number of Warmup Epochs During Contrastive Training.')
    # 数据集路径
    parser.add_argument('--data_dir', default="/home/data/palm/ScutPalm_padding/"+dataset_zoo['DR2P'], 
                        type=str, metavar='dir', help='the path of src files')
    parser.add_argument('--data_folder', default="/home/data/"+dataset_zoo['c10'], 
                        type=str, metavar='dir', help='the path of src files')
    # cos学习率衰减策略或指数衰减
    parser.add_argument('--cos', default=True, type=bool,
                        metavar='Ture', help='use cos learning rate')
    # 数据集名称
    parser.add_argument('--dataset', default=dataset_zoo['DR2P'], type=str,
                        metavar='name', help='the name of dataset') 
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
    parser.add_argument('--beta', default=2., type=float, metavar='beta',
                        help='focal beta')

    args =  parser.parse_args()
    # if args.batch_size > 256:
    args.warm = True
    if args.warm:
        args.warmup_from = .001
        args.warm_epochs = 10
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
