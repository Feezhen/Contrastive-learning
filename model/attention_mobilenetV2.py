#!/usr/bin/env python
# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor
from model.attention_module import SELayer

class SEBottleneck(nn.Module):
    '''
    Inverted Residual Block
    '''
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2, reduction = 4):
        super(SEBottleneck, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError('Stride should be 1 or 2')

        # Inverted Residual Block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, padding=int((kernel_size-1)/2),
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Linear Bottleneck，这里不接ReLU6
            # nn.ReLU6(inplace=True)
        )
        self.se = SELayer(out_channels, reduction)
        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        self.if_match_bypss = True if in_channels != out_channels else False
        if self.if_match_bypss:
            self.bypass_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        

    def forward(self, x):
        output = self.block(x)
        output = self.se(output)
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
    return SEBottleneck(in_channels=input, out_channels=output, stride=stride)


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
        self.conv_bottleneck_64_64 = conv_bottleneck(64, 64, 1)
        self.conv_bottleneck_64_128 = conv_bottleneck(64, 128, 1)
        self.conv_bottleneck_128_128 = conv_bottleneck(128, 128, 1)
        self.drop_out = nn.Dropout(0.5)
        self.conv_bottleneck_128_256 = conv_bottleneck(128, 256, 1)
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



if __name__ == '__main__':
    net = TestNet()
    print(net)
    # summary(net, (3, 360, 360))

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

