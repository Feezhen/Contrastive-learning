import torch
import torch.nn as nn
import numpy as np
import time
from utils import batch_cos_distance


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=107, feat_dim=256):
        super(CenterLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # 中心点
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        # centers的学习率
        self.lr = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        feat_dim = x.size(1)
        # x: (batch_size, feat_dim)
        # centers: (num_classes, feat_dim)
        # 提取batch中存在的类别的中心向量

        center_batch = self.centers.index_select(0, labels.long())

        diff = center_batch - x
        loss = (x - center_batch).pow(2).sum() / 2.0 / batch_size

        # print(x.size())
        # print(center_batch.size())

        # 计算更新值
        # count: 存储batch中类别出现次数的矩阵, 根据更新规则, 每个类对应的值为出现次数+1
        counts = self.centers.new_ones(self.centers.size(0))
        ones = self.centers.new_ones(labels.size(0))
        grad_centers = self.centers.new_zeros(self.centers.size())
        # 按label中的索引计算类别出现次数
        counts = counts.scatter_add_(0, labels.long(), ones)
        # 将矩阵reshape为与特征向量一致
        grad_centers.scatter_add_(0, labels.unsqueeze(1).expand(x.size()).long(), diff)
        # 计算更新值
        grad_centers = grad_centers / counts.view(-1, 1)
        # print(counts.size())
        self.centers = nn.Parameter(self.centers - self.lr * grad_centers)

        return loss

    def _center_initialize(self, gen_num=200):
        centers = torch.zeros(self.num_classes, self.feat_dim)
        for i in range(self.num_classes):
            max_euclidean_dist = 0
            for j in range(gen_num):
                rand_feat = torch.randn(self.feat_dim)

                euclidean_dist = (rand_feat - centers[:j+1])**2
                euclidean_dist = euclidean_dist.sum(dim=1, keepdim=True)
                euclidean_dist = torch.sqrt(euclidean_dist).sum(dim=0, keepdim=True)

                if euclidean_dist > max_euclidean_dist:
                    max_euclidean_dist = euclidean_dist
                    centers[i] = rand_feat

        return nn.Parameter(centers)


if __name__ == '__main__':
    loss = CenterLoss()

    test_data = torch.randn(8, 256).cuda()
    test_label = torch.Tensor(np.random.randint(0, 107, size=(8))).long().cuda()
    out = loss(test_data, test_label)

    print(out)