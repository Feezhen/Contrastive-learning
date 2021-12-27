import torch
import torch.nn as nn
import numpy as np
import os
import math
import sys
sys.path.append('../')
import params

class CurricularNCE_loss(nn.Module):
    '''
    NCE Loss with Curricular
    
    '''
    def __init__(self, gamma, keep_weight, T=0.07, m = 0.5, mlp=False):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CurricularNCE_loss, self).__init__()
        self.gamma = gamma
        self.keep_weight = keep_weight
        self.T = T
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.register_buffer('t', torch.zeros(1))
        self.alpha = 0.01
        self.weight_valid = False
        self.weight_valid_threshold = 0.8
        self.weight_scale = 5.0
        self.loss = 0

    @torch.no_grad()
    def del_tensor_ele(self, tensor, dim, index):
        """
        Delete an element from tensor
        tensor: source tensor
        dim: The dimension in which the element resides
        index: the index of the element
        """
        return tensor[torch.arange(tensor.size(dim))!=index]

    @torch.no_grad()
    def get_negatives(self, feature1, feature2, labels, index):
        """
        get negative samples from batch
        """
        neg_sample1 = feature1
        neg_sample2 = feature2
        for j in range(0, labels.shape[0]):
            if labels[j] == labels[index]:   
                neg_sample1 = self.del_tensor_ele(tensor=neg_sample1, dim=0, index=j)
                neg_sample2 = self.del_tensor_ele(tensor=neg_sample2, dim=0, index=j)

        neg_samples = torch.cat([neg_sample1, neg_sample2], dim=0)
        add_tensor = torch.zeros(1,neg_samples.shape[1]).cuda()
        while neg_samples.shape[0] < (labels.shape[0]-1)*2:
            add_tensor[0] = neg_samples[np.random.randint(0, neg_samples.shape[0])]
            neg_samples = torch.cat([neg_samples, add_tensor], dim=0)
        return neg_samples

    def forward(self, feature1, feature2, labels):
        feature1 = nn.functional.normalize(feature1, dim=1)
        feature2 = nn.functional.normalize(feature2, dim=1)
        batch_size = feature1.shape[0]
        #特征相似度
        similarity = torch.einsum('nc,kc->nk', [feature1, feature2])#nan
        similarity = similarity.clamp(-1, 1)
        with torch.no_grad():
            origin_similarity = similarity.clone()
        #对比学习正负样本对标签
        labels_temp = labels.unsqueeze(1)
        pos_labels = torch.eq(labels_temp, labels_temp.T).long().cuda()#正样本label
        neg_labels = torch.eq(pos_labels, 0).long().cuda()

        pos_labels_mask = (pos_labels == 1)
        target_similarity = similarity[pos_labels_mask] #区分出正样本cos
        with torch.no_grad():
            self.t = target_similarity.mean() * self.alpha + (1-self.alpha) * self.t
            if self.t > self.weight_valid_threshold and self.weight_valid == False:
                self.weight_valid = True
                print('CurricularNCE_loss weight valid' .center(30, '-'))
        if self.weight_valid: #开始使用focal权重
            weight_pos = pos_labels.clone().float().cuda()#正样本mask
            weight_neg = neg_labels.clone().float().cuda()
            weight_pos *= self.keep_weight #label乘以倍数
            weight_neg *= self.keep_weight
            # 相似度和label的乘积
            sim_pos = torch.mul(similarity, pos_labels)
            sim_neg = torch.mul(similarity, neg_labels)
            # 正负样本困难度(权重)
            diff_pos = pos_labels - sim_pos
            diff_neg = sim_neg
            diff_neg = diff_neg.clamp(0, 1)
            # 正负样本权重
            diff_pos = torch.pow(diff_pos, self.gamma) #exponent小数没有问题
            diff_neg = torch.pow(diff_neg, self.gamma)
            # diff_pos = torch.mul(diff_pos, pos_labels)#保证diff对应好正负样本
            diff_pos += weight_pos
            diff_neg += weight_neg
            diff_pos *= self.weight_scale
            diff_neg *= self.weight_scale

        sin_theta = torch.sqrt(1.0 - torch.pow(target_similarity, 2))
        cos_theta_m = target_similarity*self.cos_m - sin_theta*self.sin_m
        final_target_similarity = torch.where(target_similarity > self.threshold, cos_theta_m, target_similarity-self.mm) #确保单调递减
        similarity[pos_labels_mask] = final_target_similarity #把正样本位置变成cos_theta_m

        target_similarity_per_row = torch.mul(similarity, pos_labels).sum(dim=1)
        pos_labels_per_row = pos_labels.sum(dim=1)
        target_similarity_per_row = (target_similarity_per_row / pos_labels_per_row).view(-1, 1) #每行的正样本cos_theta_m均值
        
        # sin_theta_per_row = torch.sqrt(1.0 - torch.pow(target_similarity_per_row, 2))
        # cos_theta_m_per_row = target_similarity_per_row*self.cos_m - sin_theta_per_row*self.sin_m
        mask = similarity > target_similarity_per_row #困难样本
        # final_target_similarity = torch.where(target_similarity > self.threshold, cos_theta_m, target_similarity-self.mm)

        hard_example = similarity[mask] #困难样本
        similarity[mask] = hard_example * (self.t + hard_example) #负样本cos
        similarity[pos_labels_mask] = final_target_similarity #正样本cos

        logits = similarity / self.T
        logits = torch.exp(logits)

        if self.weight_valid:
            logit_pos = torch.mul(logits, diff_pos).sum(dim=1)
            logit_neg = torch.mul(logits, diff_neg).sum(dim=1)
        else:
            logit_pos = torch.mul(logits, pos_labels).sum(dim=1)
            logit_neg = torch.mul(logits, neg_labels).sum(dim=1)

        self.loss = -torch.log((logit_pos) / (logit_neg + logit_pos)).sum() / batch_size# - torch.log((d_pos.sum() / torch.sum(pos_labels == 1)).pow(2))
        
        return self.loss, self.t

if __name__ == '__main__':
    args = params.get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # loss = NCE_Loss().cuda()
    loss = CurricularNCE_loss(gamma=2, keep_weight=0.1, T=0.07, m = 0.5).cuda()

    feature1 = torch.tensor([[1.,2.,3.], [1.,-9.,3.], [3.,2.,-3.]]).cuda()
    feature2 = torch.tensor([[1.,2.,4.], [1.,-9.,6.], [4.,2.,-3.]]).cuda()
    label = torch.tensor([1, 2, 3]).cuda()
    
    outloss, t = loss(feature1, feature2, label)

    print(outloss)