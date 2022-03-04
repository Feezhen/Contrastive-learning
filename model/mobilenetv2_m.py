import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
# from .utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List

import torchvision

# 导入arcface
from loss.arcface import ArcMarginProduct,AddMarginProduct,SphereProduct
# 导入center loss
from loss.center_loss import CenterLoss


FEATURE_DIM2 = 512   # MobileNetv2m的特征输出维度


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_oup: int,
        out_oup: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_oup, out_oup, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_oup),
            activation_layer(inplace=True)
        )
        self.out_channels = out_oup


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
        

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            x = self.conv(x)
            return x


#################   MobileNetV2m; Edited by disay 20211228      ##########
class MobileNetV2m(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2m, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = FEATURE_DIM2 
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        pic_channel = 1
        self.feature = nn.Sequential(
            ConvBNReLU(pic_channel, input_channel, stride=2, norm_layer=norm_layer),
            nn.AvgPool2d(2),
            block(input_channel, 64, stride = 1, expand_ratio=6, norm_layer=norm_layer),
            nn.AvgPool2d(2),
            block(64, 128, stride = 1, expand_ratio=6, norm_layer=norm_layer),
            nn.AvgPool2d(2),
            block(128, 256, stride = 1, expand_ratio=6, norm_layer=norm_layer),
            nn.AvgPool2d(2),
            block(256, 512, stride = 1, expand_ratio=6, norm_layer=norm_layer),
            nn.AvgPool2d(2)
        )
        self.drop_out = nn.Dropout(0.2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        # feature = x
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.drop_out(x)
        return x
          
    def extract_features(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


####   MobileNetv2m with different loss functions  #######
class MobileNetV2m_Base(nn.Module):
    def __init__(self,num_classes=1000):
        super(MobileNetV2m_Base, self).__init__()
        self.encoder = MobileNetV2m()
    def forward(self,x):
        x = self.encoder.extract_features(x)
        return x

class MobileNetV2m_fc(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2m_fc, self).__init__()
        # backbone网络
        self.encoder = MobileNetV2m_Base(num_classes = num_classes)
        self.classifier = nn.Linear(FEATURE_DIM2,num_classes)
        
    def forward(self, x, label=None):
        loss_cent = 0
        loss_tri = 0
        feature = self.encoder(x)
        if not label == None:
            logit = self.classifier(feature)  
            return feature,logit,loss_cent,loss_tri
        else:
            logit = None
            return feature,None
        
class MobileNetV2m_addmargin(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2m_addmargin, self).__init__()
        # backbone网络
        self.encoder = MobileNetV2m_Base(num_classes = num_classes)
        self.classifier = AddMarginProduct(FEATURE_DIM2,num_classes)
        
    def forward(self, x, label=None):
        loss_cent = 0
        loss_tri = 0
        feature = self.encoder(x)
        if not label == None:
            logit = self.classifier(feature,label)
            return feature,logit,loss_cent,loss_tri
        else:
            logit = None
            return feature,None


class MobileNetV2m_arcmargin(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2m_arcmargin, self).__init__()
        # backbone网络
        self.encoder = MobileNetV2m_Base(num_classes = num_classes)
        self.classifier = ArcMarginProduct(FEATURE_DIM2,num_classes,easy_margin=True)
        
    def forward(self, x, label=None):
        loss_cent = 0
        loss_tri = 0
        feature = self.encoder(x)
        if not label == None:
            logit = self.classifier(feature,label)
                  
            return feature,logit,loss_cent,loss_tri
        else:
            logit = None
            return feature,None
        

class MobileNetV2m_fccenter(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2m_fccenter, self).__init__()
        # backbone网络
        self.encoder = MobileNetV2m_Base(num_classes = num_classes)
        self.classifier = nn.Linear(FEATURE_DIM2,num_classes)
        self.criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=FEATURE_DIM2)
    def forward(self, x, label=None):
        loss_cent = 0
        loss_tri = 0
        feature = self.encoder(x)
        if not label == None:
            logit = self.classifier(feature)
            loss_cent = self.criterion_cent(feature, label)
            return feature,logit,loss_cent,loss_tri
        else:
            logit = None
            return feature,None

class MobileNetV2m_addcenter(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2m_addcenter, self).__init__()
        # backbone网络
        self.encoder = MobileNetV2m_Base(num_classes = num_classes)
        self.classifier = AddMarginProduct(FEATURE_DIM2,num_classes)
        self.criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=FEATURE_DIM2)
    def forward(self, x, label=None):
        loss_cent = 0
        loss_tri = 0
        feature = self.encoder(x)
        if not label == None:
            logit = self.classifier(feature,label)
            loss_cent = self.criterion_cent(feature, label)
            return feature,logit,loss_cent,loss_tri
        else:
            logit = None
            return feature,None

class MobileNetV2m_arccenter(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2m_arccenter, self).__init__()
        # backbone网络
        self.encoder = MobileNetV2m_Base(num_classes = num_classes)
        self.classifier = ArcMarginProduct(FEATURE_DIM2,num_classes,easy_margin=True)
        self.criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=FEATURE_DIM2)
    def forward(self, x, label=None):
        loss_cent = 0
        loss_tri = 0
        feature = self.encoder(x)
        if not label == None:
            logit = self.classifier(feature,label)
            loss_cent = self.criterion_cent(feature, label)
            return feature,logit,loss_cent,loss_tri
        else:
            logit = None
            return feature

