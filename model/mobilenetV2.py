#!/usr/bin/env python
# coding:utf-8
import os, sys
from math import floor
path = os.path.dirname(os.path.dirname(__file__))
# print(path)
sys.path.append(path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
# from ResNet import *
# from .ResNet import ResNet18


class Bottleneck(nn.Module):
    '''
    Inverted Residual Block
    '''
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(Bottleneck, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError('Stride should be 1 or 2')

        # Inverted Residual Block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, # DW卷积
                      kernel_size, stride, padding=int((kernel_size-1)/2),
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1, bias=False),# PW卷积
            nn.BatchNorm2d(out_channels),
            # Linear Bottleneck，这里不接ReLU6
            # nn.ReLU6(inplace=True)
        )

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        self.if_match_bypss = True if in_channels != out_channels else False
        if self.if_match_bypss:
            self.bypass_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x): #自定义结构要写forward
        output = self.block(x)
        if self.if_match_bypss:
            return output + self.bypass_conv(x)
        else:
            return output + x


def conv_bn(input, output, stride):
    '''
    普通卷积模块（conv + bn + relu）
    :param input: 输入
    :param output: 输出
    :param stride: 步长
    :return: 普通卷积block
    '''

    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(output),
        # inplace，默认设置为False，表示新创建一个对象对其修改，也可以设置为True，表示直接对这个对象进行修改
        nn.ReLU(inplace=True)
    )


def conv_bottleneck(input, output, stride):
    return Bottleneck(in_channels=input, out_channels=output, stride=stride)


class TestNet(nn.Module):
    '''
    实验用网络网络
    使用mobilenet bottleneck
    '''
    def __init__(self, img_size=(256, 128), in_channel=3):
        '''
        构造函数
        img_size: 输入尺寸
        '''
        super(TestNet, self).__init__()
        self.in_height = img_size[0]
        self.in_widht = img_size[1]
        self.feature_map1 = None
        self.feature_map2 = None
        self.feature_map3 = None
        self.in_channel = in_channel

        self.conv_bn_3_32 = conv_bn(in_channel, 32, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_bottleneck_32_64 = conv_bottleneck(32, 64, 1)
        # self.conv_bottleneck_64_64 = conv_bottleneck(64, 64, 1)
        self.conv_bottleneck_64_128 = conv_bottleneck(64, 128, 1)
        # self.conv_bottleneck_128_128 = conv_bottleneck(128, 128, 1)
        self.drop_out = nn.Dropout(0.5)
        self.conv_bottleneck_128_256 = conv_bottleneck(128, 256, 1)
        self.conv_bottleneck_256_512 = conv_bottleneck(256, 512, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        forward function
        '''
        x = self.conv_bn_3_32(x)
        x = self.max_pool(x)
        self.feature1 = x
        x = self.conv_bottleneck_32_64(x)
        x = self.max_pool(x)
        self.feature2 = x
        x = self.conv_bottleneck_64_128(x)
        x = self.max_pool(x)
        self.feature3 = x
        x = self.conv_bottleneck_128_256(x)
        # print(x.size())
        x = self.avg_pool(x)
        # print(x.size())
        x = x.view(-1, 256)
        
        return 0, x




class MobileNet_v2(nn.Module):
    '''
    MobileNet v2网络
    '''
    def __init__(self, num_classes=507, img_size=(256, 128), in_channel=3):
        '''
        构造函数
        num_classes: 总类别数
        img_size: 输入尺寸
        '''
        super(MobileNet_v2, self).__init__()
        self.in_height = img_size[0]
        self.in_widht = img_size[1]
        self.num_classes = num_classes
        self.feature_map1 = None
        self.feature_map2 = None
        self.feature_map3 = None
        self.feature = None
        self.in_channel = in_channel

        self.conv_bn_3_32 = conv_bn(in_channel, 32, 2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv_bottleneck_32_64 = conv_bottleneck(32, 64, 1)
        # self.conv_bottleneck_64_64 = conv_bottleneck(64, 64, 1)
        self.conv_bottleneck_64_128 = conv_bottleneck(64, 128, 1)
        # self.conv_bottleneck_128_128 = conv_bottleneck(128, 128, 1)
        self.drop_out = nn.Dropout(0.5)
        self.conv_bottleneck_128_256 = conv_bottleneck(128, 256, 1)
        # self.conv_bottleneck_256_256 = conv_bottleneck(256, 256, 1)
        self.conv_bottleneck_256_512 = conv_bottleneck(256, 512, 1)
        # self.conv_bottleneck_512_512 = conv_bottleneck(512, 512, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.drop_out = nn.Dropout(0.5)

        self.fc_256_classes = nn.Linear(256, self.num_classes)
        self.fc_512_classes = nn.Linear(512, self.num_classes)

        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        forward function
        '''
        x = self.conv_bn_3_32(x)
        x = self.max_pool(x)
        # self.feature1 = x
        x = self.conv_bottleneck_32_64(x)
        x = self.max_pool(x)
        # self.feature2 = x
        # x = self.conv_bottleneck_64_64(x)
        x = self.conv_bottleneck_64_128(x)
        x = self.max_pool(x)
        # x = self.conv_bottleneck_64_128(x)
        # x = self.conv_bottleneck_128_128(x)
        x = self.conv_bottleneck_128_256(x)
        x = self.max_pool(x)
        # self.feature3 = x
        # x = self.conv_bottleneck_128_256(x)
        # x = self.max_pool(x)
        # x = self.conv_bottleneck_256_256(x)
        # x = self.max_pool(x)
        x = self.conv_bottleneck_256_512(x)
        # print(x.size())
        x = self.avg_pool(x)
        # print(x.size())
        x = x.view(-1, 512)
        self.feature = x
        x = self.drop_out(x)
        x = self.fc_512_classes(x)

        # x = F.softmax(x, dim=0)
        return x, self.feature


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    net = MobileNet_v2(num_classes=107, in_channel=1).cuda()
    # import os
    # net = EfficientNet.from_name('efficientnet-b7', include_top=False, in_channels=1).cuda()
    # net = ResNet18().cuda()
    # print(net)
    summary(net, (1, 224, 224))

    # data = torch.rand((8, 3, 360, 360))
    # output, embed = net(data)
    # print('input: {}'.format(data.shape))
    # print('output: {}'.format(output.shape))
    # # print(output)
    #
    # # embed = net.get_embedding(data)
    # print('embedding: {}'.format(embed.shape))
    #
    # loss = CenterLoss(num_classes=107, feat_dim=256)
    # labels = torch.Tensor(np.random.randint(low=0, high=107, size=8)).long()
    # print(labels.shape)
    # loss_out = loss(embed, labels)
    # print(loss_out)

