#!/usr/bin/env python
# coding:utf-8
import os
import argparse
import logging
import math
import time
from tqdm import tqdm
import shutil
import params

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model.mobilenetV2 import MobileNet_v2
from datasets import FVDataset
from loss.center_loss import CenterLoss
from utils import AverageMeter, LOG
from utils import ProgressMeter
from utils import mkdir
from utils import cos_calc_eer, batch_l2_distance, l2_calc_eer
from utils import adjust_learning_rate
from data_augment import GaussianBlur


logging.getLogger('tensorflow').disabled = True
logging.info('-----Finish Import Module-----')


# 主流程
def main():
    args = params.get_args()
    # 检查相应文件夹是否已经创建
    mkdir('./ckpt')
    mkdir(args.log_dir)
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    log_path =args.log_dir + '/'+ args.model + '_' + args.dataset + '_' + time_now + "_classified_lr" + str(args.lr) + '_batchsize' + str(args.batch_size)
    mkdir(log_path)
    # logging.basicConfig(filename=log_path+'/log.log')
    logger = LOG(log_path)
    # 指定使用的gpu
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 检测是否可以使用gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('-----Using GPU:{}.-----'.format(args.gpu))

    #是否需要进行随机数据扩增
    train_augmentation = None
    test_augmentation = None
    if args.aug:
        # 数据扩增类型
        # guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=0.5) #高斯模糊
        # train_augmentation = [
        #     guassian_blur,
        #     transforms.RandomApply([
        #         transforms.RandomChoice([
        #             transforms.ColorJitter(brightness=0.3), # 随机亮度、对比度、饱和度、色调抖动 transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.3)
        #             transforms.RandomPerspective(distortion_scale=0.03, p=0.5),   # 随机透射变换     ,这里只有透射变换是有概率发生的
        #             transforms.RandomAffine(degrees=3, translate=(0.001,0.005), scale=(0.97, 1), shear=(-1, 1), fillcolor=0)  # 随机仿射变换
        #             # transforms.RandomRotation(3),     # 随机旋转
        #         ])
        #     ],p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], 
        #                          std=[0.5, 0.5, 0.5]) 
        # ]
        random_resizecrop = transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.), ratio=(1., 1.))
        train_augmentation = [
            # guassian_blur,
            random_resizecrop,
            # transforms.RandomApply([
            #     transforms.RandomChoice([
            #         transforms.ColorJitter(brightness=0.3), # 随机亮度、对比度、饱和度、色调抖动 transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3, hue=0.3)
            #         transforms.RandomPerspective(distortion_scale=0.03, p=0.5),   # 随机透射变换     ,这里只有透射变换是有概率发生的
            #         transforms.RandomAffine(degrees=3, translate=(0.001,0.005), scale=(0.97, 1), shear=(-1, 1), fillcolor=0)  # 随机仿射变换
            #         # transforms.RandomRotation(3),     # 随机旋转
            #     ])
            # ],p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
                transforms.RandomPerspective(distortion_scale=0.05, p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.001, 0.005)),
            ], p=0.7),
            transforms.RandomRotation(5),     # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], 
                                 std=[0.5]) 
        ]

        test_augmentation = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], 
                                 std=[0.5])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            #                         std=[0.5, 0.5, 0.5]) 
        ]

    #加载数据集
    logger.info('-----Loading dataset-----')
    train_dataset = FVDataset(train=True, mode='classified', args=args, transform_fcn=transforms.Compose(train_augmentation))
    valid_dataset = FVDataset(train=False, mode='valid', args=args, transform_fcn=transforms.Compose(test_augmentation))
    test_dataset = FVDataset(train=False, mode='test', args=args, transform_fcn=transforms.Compose(test_augmentation))
    
    # 如果使用分布式训练，进行分布式采样
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    # 创建训练数据迭代器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)

    # 创建测试数据迭代器
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # 训练使用的模型
    model = MobileNet_v2(num_classes=train_dataset.num_class, img_size=args.img_size, in_channel=1)

# FixMe分布式训练没写好***************************************************************

    # 是否使用分布式训练
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            model = model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu)
        else:
            model = model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = model.cuda()

# 以上部分待修改*********************************************************************

    # 如果使用GPU训练，模型固定的情况下使用cudnn加速
    torch.backends.cudnn.benchmark = True
    # 定义损失函数
    criterion_CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
    criterion_CenterLoss = CenterLoss(num_classes=int(train_dataset.num_class), feat_dim=512).cuda()
    # 定义优化器
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), 
                                  lr=args.lr, betas=(0.9, 0.99))

    best_eer = 1
    Thres_record = 0.0
    # 是否进行断点续训
    if args.resume:
        ckpt_path = './ckpt/checkpoint_{}_classified.pth.tar'.format(args.dataset)
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            if args.gpu is None:
                checkpoint = torch.load(ckpt_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(ckpt_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_eer = checkpoint['eer']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))

    # 构建tensorboard writer
    writer = SummaryWriter(log_dir=log_path)
    # 开始迭代训练
    for epoch in range(args.start_epoch, args.epochs + 1):
        # 使sampler变换在多个epoch之间正常工作。否则，将始终使用相同的顺序。
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # 根据epoch调整学习率
        adjust_learning_rate(optimizer, epoch, args)
        logger.info('-------Lr:{}------'.format(optimizer.param_groups[0]['lr']))
        # 使用迭代器创建tqdm进度条
        tqdm_batch = tqdm(train_loader, desc='INFO:root:-----Epoch-{} training-----'.format(epoch))
        
        cross_entropy_loss_tracker = AverageMeter('crossEntropy', ':.4e')
        center_loss_tracker = AverageMeter('centerLoss', ':.4e')
        loss_tracker = AverageMeter('loss', ':.4e')

        # switch to train mode
        model.train()
        # 训练batch
        for i, (image, label) in enumerate(tqdm_batch):
            # measure data loading time
            image = image.cuda()
            # print(image.shape)
            label = label.cuda()
            optimizer.zero_grad()
            # compute output
            output, feature = model(image)
            cross_entropy_loss = criterion_CrossEntropyLoss(output, label)
            # center_loss = criterion_CenterLoss(feature, label)
            loss = cross_entropy_loss #+ center_loss

            # compute gradient and do SGD or ADAM step
            loss.backward()
            optimizer.step()
            # optimizer_centerloss.step()
            # loss_tracker.update(loss.item(), label.size(0))
            # center_loss_tracker.update(center_loss.item(), label.size(0))
            cross_entropy_loss_tracker.update(cross_entropy_loss.item(), label.size(0))
        tqdm_batch.close()
        # logger.info("-----Loss: {:.6f} ({:.6f}) -------CenterLoss: {:.6f} ({:.6f}) ---CrossEntropy:{:.6f} ({:.6f}) "\
        #              .format(loss_tracker.val, loss_tracker.avg, center_loss_tracker.val, center_loss_tracker.avg,\
        #                  cross_entropy_loss_tracker.val, cross_entropy_loss_tracker.avg))
        logger.info("---CrossEntropy:{:.6f} ({:.6f}) "\
                     .format(cross_entropy_loss_tracker.val, cross_entropy_loss_tracker.avg))
        # 写进tensorboard
        # writer.add_scalar('Loss', loss_tracker.avg, epoch)
        # writer.add_scalar('CenterLoss', center_loss_tracker.avg, epoch)
        writer.add_scalar('SoftmaxLoss', cross_entropy_loss_tracker.avg, epoch)

        # 使用测试集测试
        if epoch % args.test_freq == 0: 
            # 使用迭代器创建tqdm进度条
            # 使用迭代器创建tqdm进度条
            tqdm_batch = tqdm(valid_loader, desc='-----Epoch-{} validating-----'.format(epoch))
            model.eval()
            distances_cos = []
            distances_l2 = []
            MeasurementOptions = 'cos'
            labels = []
            # best_performance = False
            eer = 1
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
                    eer, bestThresh, minV = cos_calc_eer(distances_cos, label, log_path, epoch)
                else:
                    eer, bestThresh, minV = l2_calc_eer(distances_l2, label, log_path, epoch)
                # 得到验证集的阈值
                logger.info('----valid_eer: {:.6f}, bestThreshold:{:.6f}'.format(eer, bestThresh))
            tqdm_batch.close()

            # 测试集
            tqdm_batch = tqdm(test_loader, desc='-----Epoch-{} testing-----'.format(epoch))
            model.eval()
            distances_cos = []
            distances_l2 = []
            best_performance = False
            labels = []
            eer = 1
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
                    eer, bestThresh, minV = cos_calc_eer(distances_cos, label, log_path, epoch, bestThresh)
                else:
                    eer, bestThresh, minV = l2_calc_eer(distances_l2, label, log_path, epoch)
            
                best_performance = True if eer < best_eer else False
                # eer, bestThresh, minV = l2_calc_eer(distances_l2, label, log_path)
                logger.info('-----test_eer: {:.6f}, bestThreshold: {:.6f}, minV: {:.6f}'.format(eer, bestThresh, minV))
                writer.add_scalar('eer', eer, epoch)
            # 保存checkpoints
            if best_performance:
                best_eer = eer
                Thres_record = bestThresh
                save_checkpoint({
                    'epoch': epoch + 1,
                    'eer': eer,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, is_best=best_performance, filename=log_path+'/checkpoint_{}_{}_{}_classified.pth.tar'\
                                                            .format(args.model, args.dataset, MeasurementOptions))
            tqdm_batch.close()
            logger.info('---best_eer:{:.6f}, Thres:{:.6f}-----'.format(best_eer, Thres_record))

    writer.close()
    torch.save(model, log_path + '/{}_{}_model.pth'.format(args.model, args.dataset))
    torch.save(model.state_dict(), log_path + '/{}_{}_params.pth'.format(args.model, args.dataset))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()


