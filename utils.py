#!/usr/bin/env python
# coding:utf-8
import os
import sys
import logging
from PIL.Image import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# import cv2
import matplotlib.pyplot as plt

class LOG():
    def __init__(self, dir):
        # 创建一个logger
        self.logger = logging.getLogger('MyLogger')
        self.logger.setLevel(logging.INFO)
        # logging.getLogger('tensorflow').disabled = True
        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(
            os.path.join(dir, 'log.log'), encoding='utf-8')
        fh.setLevel(logging.INFO)

        # 定义handler的输出格式
        # formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] # %(message)s')
        formatter = logging.Formatter('[%(asctime)s] # %(message)s')
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)

        # 记录一条日志
        self.logger.info('-----------------------------------------------------------')
        # self.logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        self.logger.info(sys.argv[0])  # 第0个就是这个python文件本身的路径（全路径）
        # 记录所有参数
        self.logger.info(self.__dict__)
    '''
    LOGGING
    '''
    def info(self, *input):
        final = ''
        for item in input:
            final = final + str(item)
        self.logger.info(final)

class TwoCropTransform:
    '''create two crops of the same image'''
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# class Distances_Metric(nn.Module):
#     """
#     A distance measurement network
#     """
#     def __init__(self, input_dim):
#         '''
#         构造函数
#         input_dim: 输入向量的维度
#         '''
#         super(Distances_Metric, self).__init__()
#         self.cal_layer = nn.Linear(input_dim, 2)
        
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), 0)
#         x = self.cal_layer(x1)
#         score = F.log_softmax(x, dim=-1)
#         return score

def accuracy(output, target, topk=(1,)):
    '''Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# 学习率调整
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        lr = lr * (0.8 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# warmup Lr
def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)# 线性增长

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# 判断路径path是否存在，不存在就创建路径
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print("Create Folder: " + path)
    else :
        print(path + " is existed")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def cos_distance(feature1, feature2):
    '''
    计算两个特征向量之间的余弦距离
    :param feature1: [1, feat_dim]
    :param feature2: [1, feat_dim]
    :return: distance: 余弦距离
    '''
    # reshape成[1, feat_dim]
    feature1 = feature1.reshape(-1, feature1.size(-1))
    feature2 = feature2.reshape(-1, feature2.size(-1))
    return torch.sum(feature1 * feature2, -1) / (torch.norm(feature1) * torch.norm(feature2))


def batch_cos_distance(feature1, feature2):
    '''
    计算两组特征向量之间的余弦距离
    :param feature1:  [batch_size, feat_dim]
    :param feature2:  [batch_size, feat_dim]
    :return: distances: [batch_size]
    '''
    batch_size = feature1.size(0)
    # feat_dim = feature1.size(1)
    distances = []
    for i in range(batch_size):
        distances.append(cos_distance(feature1[i], feature2[i]))
    return torch.Tensor(distances)


def l2_distance(feature1, feature2):
    '''
    计算两个特征向量之间的余弦距离
    :param feature1: [1, feat_dim]
    :param feature2: [1, feat_dim]
    :return: distance: 余弦距离
    '''
    distance = torch.norm((feature1 - feature2))
    return distance


def batch_l2_distance(feature1, feature2):
    '''
    计算两组特征向量之间的余弦距离
    :param feature1:  [batch_size, feat_dim]
    :param feature2:  [batch_size, feat_dim]
    :return: distances: [batch_size]
    '''
    batch_size = feature1.size(0)
    # feat_dim = feature1.size(1)
    distances = []
    for i in range(batch_size):
        distances.append(l2_distance(feature1[i], feature2[i]))
    return torch.Tensor(distances)

'''
# 1表示同类，0表示异类
def cos_calc_eer(distances, label, log_path, epoch):
    
    计算等误率
    :param distances:  余弦距离矩阵，[batch_size]
    :param label:  标签，[batch_size]；1，表示同类；0，表示异类
    :return:
    
    batch_size = label.size(0)
    minV = 100
    bestThresh = 0
    eer = 1
    file_path = log_path + '/' + 'cos_roc.txt'
    fd = open(file_path, 'w+')
    # max_dist = torch.max(distances)
    # min_dist = torch.min(distances)
    threshold_list = np.linspace(0, 1, num=100)

    for threshold in threshold_list:
        intra_cnt = 0
        intra_len = 0
        inter_cnt = 0
        inter_len = 0
        errorList = []
        for i in range(batch_size):
        # 注意是余弦距离，越大越相近，所以这里若小于0则错误了
            if label[i] == 1:
                intra_len += 1
                if distances[i] < threshold:
                    intra_cnt += 1
            elif label[i] == 0:
                inter_len += 1
                if distances[i] > threshold:
                    inter_cnt += 1

        fr = intra_cnt / intra_len
        fa = inter_cnt / inter_len

        print('fr {:.6f}, fa {:.6f}, thr {:.6f}'.format(fr, fa, threshold))
        if epoch % 50 == 0:
            fd.write(str(fr) + ',' + str(fa) + ',' + str(threshold) + '\n')

        if abs(fr - fa) < minV:
            minV = abs(fr - fa)
            eer = (fr + fa) / 2
            bestThresh = threshold
    fd.close()

    return eer, bestThresh, minV
'''
def cos_calc_eer(distances, label, log_path, epoch, best_thres = None):
    minV = 100
    bestThresh = 0
    eer = 1
    ones = torch.ones(label.shape).type(torch.LongTensor).cuda() #全1变量
    zeros = torch.zeros(label.shape).type(torch.LongTensor).cuda()  # 全0变量
    is_test = False
    if best_thres == None:
        threshold_list = np.linspace(0, 1, num=100)
    else:
        threshold_list = []
        threshold_list.append(best_thres)
        is_test = True
    if not is_test:
        file_path = log_path + '/' + 'cos_roc{}.txt'.format(epoch)
        fd = open(file_path, 'w+')
    for threshold in threshold_list:
        pred = torch.gt(distances, threshold).cuda()
        # 获取TP,TN,FP,FN
        tp = ((label == ones) & (pred==ones)).sum().type(torch.FloatTensor).cuda()     # 实际标签为1(正例)且预测标签为1的数量    预测正确
        fp = ((label == zeros) & (pred == ones)).sum().type(torch.FloatTensor).cuda()  # 实际标签为0(负例)且预测标签为1的数量
        fn = ((label == ones) & (pred == zeros)).sum().type(torch.FloatTensor).cuda()  # 实际标签为1(正例)且预测标签为0的数量
        tn = ((label == zeros) & (pred == zeros)).sum().type(torch.FloatTensor).cuda() # 实际标签为0(负例)且预测标签为0的数量    预测正确
        fr = fn / (fn + tp)
        fa = fp / (fp + tn)

        print('fr {:.6f}, fa {:.6f}, thr {:.6f}'.format(fr, fa, threshold))
        if not is_test:#验证集才写匹配情况
            fd.write('{:.6f}, {:.6f}, {:.6f}\n'.format(fr, fa, threshold))
            fd.write('tp{}, fp{}, fn{}, tn{}\n'.format(tp, fp, fn, tn))
        if abs(fr - fa) < minV:
            minV = abs(fr - fa)
            eer = (fr + fa) / 2
            bestThresh = threshold
    if not is_test:
        fd.close()
    return eer, bestThresh, minV

# L2距离计算1表示同类，0表示异类
def l2_calc_eer(distances, label, log_path, epoch):
    '''
    计算等误率
    :param distances:  余弦距离矩阵，[batch_size]
    :param label:  标签，[batch_size]；1，表示同类；0，表示异类
    :return:
    '''
    batch_size = label.size(0)
    minV = 100
    bestThresh = 0
    eer = 1

    file_path = log_path + '/' + 'l2_roc.txt'
    fd = open(file_path, 'w+')
    max_dist = torch.max(distances).cpu()
    min_dist = torch.min(distances).cpu()
    threshold_list = np.linspace(min_dist, max_dist, num=100)

    for threshold in threshold_list:
        intra_cnt = 0
        intra_len = 0
        inter_cnt = 0
        inter_len = 0

        for i in range(batch_size):
        # intra
        # 注意是余弦距离，越大越相近，所以这里若小于则错误了
            if label[i] == 1:
                intra_len += 1
                if distances[i] > threshold:
                    intra_cnt += 1
            elif label[i] == 0:
                inter_len += 1
                if distances[i] < threshold:
                    inter_cnt += 1

        fr = intra_cnt / intra_len
        fa = inter_cnt / inter_len
        if epoch % 50 == 0:
            print('fr {:.6f}, fa {:.6f}, thr {:.6f}'.format(fr, fa, threshold))
            fd.write(str(fr) + ',' + str(fa) + ',' + str(threshold) + '\n')

        if abs(fr - fa) < minV:
            minV = abs(fr - fa)
            eer = (fr + fa) / 2
            bestThresh = threshold
    fd.close()

    return eer, bestThresh, minV

#重命名图片，使得数据集构建按照顺序
def Pic_rename(data_path):
    data_list = os.listdir(data_path)
    data_list.sort()
    for i in range(len(data_list)):
        sub_name = data_list[i].split('_')[0]
        sub_name_len = len(sub_name)
        next_name = data_list[i][sub_name_len:]
        if sub_name_len == 1:
            sub_name = '000' + sub_name
        elif sub_name_len == 2:
            sub_name = '00' + sub_name
        elif sub_name_len == 3:
            sub_name = '0' + sub_name
        image_name = sub_name + next_name
        Image = cv2.imread(os.path.join(data_path, data_list[i]), -1)
        cv2.imwrite(os.path.join(data_path+'new', image_name), Image)
    # return 0

#将图片从各自的文件夹拷贝到一起
def gather_pic():
    Pic_root = '/home/data/palm/ScutPalm_padding/dataBase_roi_2_padding1/train'
    folder_list = os.listdir(Pic_root)
    for folder in folder_list:
        Pic_path = os.path.join(Pic_root, folder)
        Pic_list = os.listdir(Pic_path)
        for Pic in Pic_list:
            path1 = os.path.join(Pic_path, Pic)
            Image = cv2.imread(path1, 0)
            path2 = os.path.join(Pic_root, Pic)
            cv2.imwrite(path2, Image)

#根据txt来画eer曲线
def draw_roc(file_path): 
    scores = []
    with open(file_path, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            fr = float(i[:-1].split(',')[0])
            fa = float(i[:-1].split(',')[1])
            t = float(i[:-1].split(',')[2])
            scores.append([t, fa, fr])
    scores = np.array(scores)
    plt.plot(scores[:, 0], scores[:, 1], label='fa', color='#ff0000')
    plt.plot(scores[:, 0], scores[:, 2], label='fr', color='#00ff00')
    plt.legend(loc='center right')
    # plt.show()
    plt.savefig(file_path[:-14]+'roc')
    return


if __name__ == '__main__':
    # f1 = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    # f2 = f1 * 2
    # dists = batch_cos_distance(f1, f2)
    # print(dists)
    # x1 = torch.tensor([[1,2,3,4,5]])
    # x2 = torch.tensor([[6,7,8,9,10]])
    # x = torch.cat((x1, x2), 0)
    # for i in range(30):
    #     x = np.random.choice(a=[0,1], size=6, replace=True, p=None)
        # print(x[0])
    draw_roc('./log/mobilenetV2_Undergraduate_2020_20210511-1341_contrastive_lr0.00125_batchsize32/cos_roc200.txt')
    # gather_pic()