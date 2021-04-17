from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SALayer(nn.Module):
    """docstring for SALayer"""
    def __init__(self, kernel_size = 3):
        super(SALayer, self).__init__()

        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = x
        avg_out = torch.mean(y,dim=1,keepdim=True)
        max_out = torch.max(y,dim=1,keepdim=True)
        y = torch.cat([avg_out,max_out],dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x
        

class jointAtten(nn.Module):
    def __init__(self, planes,reduction=8,kernel_size=3):
        super(jointAtten, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=True),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=1,bias=False)
        self.conv2 = nn.Conv2d(2*planes,planes,1,1,False)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True)
        sa = torch.cat([avg_out,max_out],dim=1)
        sa = self.conv1(sa)
        sa = self.sigmoid(sa) * x

        b, c, _, _ = x.size()
        ca = self.avg_pool(x).view(b, c)
        ca = self.fc(ca).view(b, c, 1, 1)
        ca = x*ca.expand_as(x)

        joint = torch.cat([ca,sa],1)
        ## print(joint.shape)
        return self.conv2(joint)
