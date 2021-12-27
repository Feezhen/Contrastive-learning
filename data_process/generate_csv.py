#!/usr/bin/env python
# coding:utf-8
import os
import csv
import math
from typing import get_args
from cv2 import data
import numpy as np
from torch.nn.modules.module import T
from torch.serialization import save
from torch.utils.tensorboard import writer
import params
import pandas as pd
from tqdm import tqdm

# path = "../data/finger_roi/MMCBNU_6000_FVDataSet_ROI"
# path = "/home/data/finger_vein/SCUT_LFMB/SCUT_LFMB_A_device/A_Session1_Picture"

# 多光谱/多视角数据采集特定光照/视角  
def create_contrastive_data_csv(data_path, t=6, test_mode='open', test_ratio=0.3, dataset_name='A1', view_total=1, view=1):
    data_list = os.listdir(data_path)
    data_list.sort()
    # 根据每类数据的数量算出数据集中类别数量
    sub = int(len(data_list) / t)
    
    file_name = dataset_name + '_' + test_mode + '_contrastive' + '.csv'
    pwd = os.getcwd()
    save_path = os.path.join(os.path.join(pwd, 'csv'), file_name)
    fd = open(save_path, 'w', encoding='utf-8')
    headers = ['group', 'sample1', 'sample2', 'label']
    writer = csv.DictWriter(fd,headers)
    writer.writeheader()
    # 每类样本可以构成正样本对的组合数
    pairs_num = math.factorial(t)//(math.factorial(2)*math.factorial(t-2))
    print(pairs_num)
    train_sub = sub * (1-test_ratio)
    for i in range(sub) :
        for j in range(int(t/view_total)-1):
            sample1 = i*t + j*view_total + view-1
            if i <= train_sub:
                for k in range(j+1, int(t/view_total)):
                    sample2 = i*t + k*view_total + view-1
                    data = { 'group':'train', 'sample1': data_list[sample1], 
                            'sample2':data_list[sample2], 'label':i }
                    writer.writerow(data)

            else :
                for k in range(j+1, int(t/view_total)):
                    # 类内
                    sample2 = i*t + k*view_total + view-1
                    data = { 'group':'test', 'sample1': data_list[sample1], 
                            'sample2':data_list[sample2], 'label':1 }
                    writer.writerow(data)
                    # 类间
                    sub_neg = np.random.randint(train_sub + 1, sub)
                    while sub_neg  == i:
                        sub_neg = np.random.randint(train_sub + 1, sub)
                    t_neg = np.random.randint(1, t/view_total)
                    sample2_neg = sub_neg*t + t_neg*view_total + view-1
                    data = { 'group':'test', 'sample1': data_list[sample1], 
                            'sample2':data_list[sample2_neg], 'label':0 }
                    writer.writerow(data)
    fd.close()
    
    df = pd.read_csv(save_path)
    sample1 = df['sample1']
    trainSample = df[df['group'] == 'train']
    testSample = df[df['group'] == 'test']
    print('总样本对数：' + str(len(sample1)))
    print('训练样本对数：' + str(len(trainSample)))
    print('测试样本对数：' + str(len(testSample)))
    

# 非样本对
def create_classified_data_csv(data_path, t=60, test_mode='open', test_ratio=0.3, dataset_name='A', view_total=1, view=1):
    """
    划分测试集和训练集
    参数：
    data_path：数据集路径
    t：每个类别样本数量
    test_mode：开集测试或闭集测试
    test_ratio：测试集占比
    dataset_name：测试集名称
    view_total：数据集中每个类别样本的视角/模态等不同拍摄条件的数量(比如三视角数据就是3，六视角数据就是6，多光谱数据里有六种光照就是6，有红外光和可见光两种模态数据就是2)
    view：要筛选的视角(比如要单独提取三视角数据中第一个视角的数据，就是1，要单独提取多光谱数据中第三种光照的数据，就是3)
    （注：view_total和view仅在多种视图下需筛选单一视图时使用，否则就都是1）
    """
    data_list = os.listdir(data_path)
    data_list.sort()
    # 根据每类数据的数量算出数据集中类别数量
    sub = int(len(data_list) / t)
    
    file_name = dataset_name + '_' + test_mode + '_classified{}'.format(view) + '.csv'
    pwd = os.getcwd()
    save_path = os.path.join(os.path.join(pwd, 'csv'), file_name)
    fd = open(save_path, 'w', encoding='utf-8')
    headers = ['group', 'sample1', 'sample2', 'label']
    writer = csv.DictWriter(fd,headers)
    writer.writeheader()
    if test_mode == 'open' :
        train_sub = sub * (1-test_ratio)
        test_sub = sub * test_ratio
        for i in range(sub) :
            for j in range(int(t/view_total)):
                sample1 = i*t + j*view_total + view-1
                if i <= train_sub:
                    data = { 'group':'train', 'sample1': data_list[sample1], 
                                'sample2':None, 'label':i }
                    writer.writerow(data)

                elif j <= t/view_total - 1:
                    for k in range(j+1, int(t/view_total)):
                        # 类内
                        sample2 = i*t + k*view_total + view-1
                        data = { 'group':'test', 'sample1': data_list[sample1], 
                                'sample2':data_list[sample2], 'label':1 }
                        writer.writerow(data)
                        # 类间
                        sub_neg = np.random.randint(train_sub + 1, sub)
                        while sub_neg  == i:
                            sub_neg = np.random.randint(train_sub + 1, sub)
                        t_neg = np.random.randint(1, t/view_total)
                        sample2_neg = sub_neg*t + t_neg*view_total + view-1
                        data = { 'group':'test', 'sample1': data_list[sample1], 
                                'sample2':data_list[sample2_neg], 'label':0 }
                        writer.writerow(data)
    else :
        # close mode
        mode = 0

    # close file
    fd.close()
    # print totalNum
    df = pd.read_csv(save_path)
    sample1 = df['sample1']
    trainSample = df[df['group'] == 'train']
    testSample = df[df['group'] == 'test']
    print('总样本数：' + str(len(sample1)))
    print('训练样本数：' + str(len(trainSample)))
    print('测试样本数：' + str(len(testSample)))



def divide_data(data_path, t=60, test_mode='open', test_ratio=0.2, dataset_name='A', view_total=1, view=1):
    """
    功能：
    划分测试集和训练集
    参数：
    data_path：数据集路径
    t：每个类别样本数量
    test_mode：开集测试或闭集测试
    test_ratio：测试集占比
    dataset_name：测试集名称
    view_total：数据集中每个类别样本的视角/模态等不同拍摄条件的数量(比如三视角数据就是3，六视角数据就是6，多光谱数据里有六种光照就是6，有红外光和可见光两种模态数据就是2)
    view：要筛选的视角(比如要单独提取三视角数据中第一个视角的数据，就是1，要单独提取多光谱数据中第三种光照的数据，就是3)
    （注：view_total和view仅在多种视图下需筛选单一视图时使用，否则就都是1）
    """
    data_list = os.listdir(data_path)
    data_list.sort()
    # 根据每类数据的数量算出数据集中类别数量
    sub = int(len(data_list) / t)
    file_name = dataset_name + '_' + test_mode + '_divide' + '.csv'
    pwd = os.getcwd()
    save_path = os.path.join(os.path.join(pwd, 'csv'), file_name)
    fd = open(save_path, 'w', encoding='utf-8')
    headers = ['group', 'sample1', 'sample2', 'label']
    writer = csv.DictWriter(fd,headers)
    writer.writeheader()
    if test_mode == 'open' :
        train_sub = sub * (1-test_ratio)
        test_sub = sub * test_ratio
        for i in tqdm(range(sub)):
            for j in range(int(t/view_total)):
                sample1 = i*t + j*view_total + view-1
                if i <= train_sub:
                    data = { 'group':'train', 'sample1': data_list[sample1], 
                                'sample2':None, 'label':i }
                    writer.writerow(data)

                else:
                    data = { 'group':'test', 'sample1': data_list[sample1], 
                                'sample2':None, 'label':i }
                    writer.writerow(data)
    else :
        # close mode
        mode = 0

    # close file
    fd.close()
    # print totalNum
    df = pd.read_csv(save_path)
    sample1 = df['sample1']
    trainSample = df[df['group'] == 'train']
    testSample = df[df['group'] == 'test']
    print('总样本数：' + str(len(sample1)))
    print('训练样本数：' + str(len(trainSample)))
    print('测试样本数：' + str(len(testSample)))

def divide_data2(data_path, t=60, test_mode='open', test_ratio=0.2, dataset_name='A', view_total=1, view=1):
    """
    功能：
    划分测试集和训练集
    参数：
    data_path：数据集路径
    t：每个类别样本数量
    test_mode：开集测试或闭集测试
    test_ratio：测试集占比
    dataset_name：测试集名称
    view_total：数据集中每个类别样本的视角/模态等不同拍摄条件的数量(比如三视角数据就是3，六视角数据就是6，多光谱数据里有六种光照就是6，有红外光和可见光两种模态数据就是2)
    view：要筛选的视角(比如要单独提取三视角数据中第一个视角的数据，就是1，要单独提取多光谱数据中第三种光照的数据，就是3)
    （注：view_total和view仅在多种视图下需筛选单一视图时使用，否则就都是1）
    """
    test_path = os.path.join(data_path, 'test')
    data_list = os.listdir(test_path)
    data_list.sort()
    # 根据每类数据的数量算出数据集中类别数量
    sub = int(len(data_list) / t)
    file_name = dataset_name + '_' + test_mode + '_divide' + '.csv'
    pwd = os.getcwd()
    save_path = os.path.join(os.path.join(pwd, 'csv'), file_name)
    fd = open(save_path, 'w', encoding='utf-8')
    headers = ['group', 'sample1', 'sample2', 'label']
    writer = csv.DictWriter(fd,headers)
    writer.writeheader()
    if test_mode == 'open' :
        for i in tqdm(range(sub)):
            for j in range(int(t/view_total)):
                sample1 = i*t + j*view_total + view-1
                data = { 'group':'test', 'sample1': data_list[sample1], 
                                'sample2':None, 'label':i }
                writer.writerow(data)
    else :
        # close mode
        mode = 0

    # close file
    fd.close()
    # print totalNum
    df = pd.read_csv(save_path)
    sample1 = df['sample1']
    trainSample = df[df['group'] == 'train']
    testSample = df[df['group'] == 'test']
    print('总样本数：' + str(len(sample1)))
    print('训练样本数：' + str(len(trainSample)))
    print('测试样本数：' + str(len(testSample)))


# 非完全配对——一个样本只跟一个正例配对
def create_contrastive_data_csv2(data_path, t=6, test_mode='open', test_ratio=0.3, dataset_name='A1', view_total=1, view=1):
    data_list = os.listdir(data_path)
    data_list.sort()
    # 根据每类数据的数量算出数据集中类别数量
    sub = int(len(data_list) / t)
    
    file_name = dataset_name + '_' + test_mode + '_contrastive' + '.csv'
    pwd = os.getcwd()
    save_path = os.path.join(os.path.join(pwd, 'csv'), file_name)
    fd = open(save_path, 'w', encoding='utf-8')
    headers = ['group', 'sample1', 'sample2', 'label']
    writer = csv.DictWriter(fd,headers)
    writer.writeheader()
    # 每类样本可以构成正样本对的组合数
    train_sub = sub * (1-test_ratio)
    test_sub = sub * test_ratio
    for i in tqdm(range(sub)) :
        for j in range(int(t/view_total)):
            if i <= train_sub:
                sample1 = i*t + j * view_total + view-1
                if j < t / view_total - 1:
                    sample2 = i*t + (j+1) * view_total + view-1
                else:
                    sample2 = i*t + view-1
                data = { 'group':'train', 'sample1': data_list[sample1], 
                        'sample2':data_list[sample2], 'label':i }
                writer.writerow(data)

            else :
                for k in range(j+1, int(t/view_total)):
                    sample1 = i*t + j*view_total + view-1
                    # 类内
                    sample2 = i*t + k*view_total + view-1
                    data = { 'group':'test', 'sample1': data_list[sample1], 
                            'sample2':data_list[sample2], 'label':1 }
                    writer.writerow(data)
                    # 类间
                    sub_neg = np.random.randint(train_sub + 1, sub)
                    while sub_neg  == i:
                        sub_neg = np.random.randint(train_sub + 1, sub)
                    t_neg = np.random.randint(0, t/view_total)
                    sample2_neg = sub_neg*t + t_neg*view_total + view-1
                    data = { 'group':'test', 'sample1': data_list[sample1], 
                            'sample2':data_list[sample2_neg], 'label':0 }
                    writer.writerow(data)
    fd.close()
    
    # fd = open(save_path, 'r')
    # reader = csv.DictReader(fd)
    # sample1 = [row['sample1'] for row in reader]
    # fd.close()
    df = pd.read_csv(save_path)
    sample1 = df['sample1']
    trainSample = df[df['group'] == 'train']
    testSample = df[df['group'] == 'test']
    print('总样本类别数：' + str(train_sub))
    print('总样本对数：' + str(len(sample1)))
    print('训练样本对数：' + str(len(trainSample)))
    print('测试样本对数：' + str(len(testSample)))

# 用于test和train分开文件夹的情况
def create_contrastive_data_csv3(data_path, t=6, test_mode='open', test_ratio=0.3, dataset_name='A1', view_total=1, view=1):
    '''
        用于test和train分开文件夹的情况下的csv生成
    '''
    # train_path = os.path.join(data_path, 'train')
    # test_path = os.path.join(data_path, 'test')
    # train data
    # data_list = os.listdir(train_path)
    # data_list.sort()
    csv_path = os.path.join(os.path.join(data_path, '523csv'), dataset_name+'_train.csv')
    data_list = get_trainset(csv_path)
    sub = int(len(data_list) / t)
    file_name = dataset_name + '_contrastive523.csv'
    pwd = os.getcwd()
    save_path = os.path.join(os.path.join(pwd, 'csv'), file_name)
    fd = open(save_path, 'w', encoding='utf-8')
    headers = ['group', 'sample1', 'sample2', 'label']
    writer = csv.DictWriter(fd, headers)
    writer.writeheader()
    for i in tqdm(range(sub)):
        for j in range(int(t/view_total)):
            sample1 = i*t + j*view_total + view-1
            if j < t / view_total -1:
                sample2 = i*t + (j+1)*view_total + view-1
            else:
                sample2 = i*t + view-1
            data = {'group': 'train', 'sample1': data_list[sample1],
                    'sample2': data_list[sample2], 'label': i}
            writer.writerow(data)
    # test data
    # data_list = os.listdir(test_path)
    # data_list.sort()
    # sub = int(len(data_list) / t)
    # for i in tqdm(range(sub)):
    #     for j in range(int(t/view_total)):
    #         sample1 = i*t + j*view_total + view-1
    #         for k in range(int(t/view_total)):
    #             if k == j:
    #                 continue
    #             #类内
    #             sample2 = i*t + k*view_total + view-1
    #             data = {'group': 'test', 'sample1': data_list[sample1],
    #                 'sample2': data_list[sample2], 'label': 1}
    #             writer.writerow(data)
    #             #类间
    #             sub_neg = np.random.randint(0, sub)
    #             while sub_neg == i:
    #                 sub_neg = np.random.randint(0, sub)
    #             t_neg = np.random.randint(0, int(t/view_total))
    #             sample2_neg = sub_neg*t + t_neg*view_total + view-1
    #             data = {'group': 'test', 'sample1': data_list[sample1],
    #                 'sample2': data_list[sample2_neg], 'label': 0}
    #             writer.writerow(data)
    fd.close()
    df = pd.read_csv(save_path)
    sample1 = df['sample1']
    # print(type(df['group'] == 'train'))
    trainSample = df[df['group'] == 'train']
    # testSample = df[df['group'] == 'test']
    print('总样本对数：' + str(len(sample1)))
    print('训练样本对数：' + str(len(trainSample)))
    # print('测试样本对数：' + str(len(testSample)))

#根据csv确定训练的样本范围
def get_trainset(path):
    '''
        根据csv确定训练的样本范围
    '''
    colname = ['data', 'label']
    df = pd.read_csv(path, header=None, names=colname)
    data = df['data']
    # data_set = set(data)
    # data_set = sorted(data_set)

    return list(data)


    
if __name__ == "__main__":
    args = params.get_args()
    # divide_data2(args.data_dir, t=30, test_mode='open', test_ratio=0.3, dataset_name=args.dataset, view_total=1, view=1)
    create_contrastive_data_csv3(args.data_dir, t=10, test_mode='open', test_ratio=0.3, dataset_name=args.dataset, view_total=1, view=1)
    # create_classified_data_csv(args.data_dir, t=6, test_mode='open', test_ratio=0.2, dataset_name=args.dataset, view_total=1, view=1)
    # data_list = os.listdir("/home/data/dataBase_temp_gqc")
    # print(data_list)
    # data = get_trainset('/home/data/palm/csv/Undergraduate_2020_train.csv')
    # print(data)
    # print(len(data))
