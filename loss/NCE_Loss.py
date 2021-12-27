import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append('..')
import params

from torchvision.transforms.transforms import ToTensor

class NCE_Loss(nn.Module):
    """
    NCE loss.
    By gqc
    """
    def __init__(self, T=0.07, mlp=False):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(NCE_Loss, self).__init__()
        self.T = T
        self.loss = 0
        self.sigmoid = nn.Sigmoid()

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
        """
        Input:
            feature1: a batch of query images features
            feature2: a batch of key images features
            labels: a batch of positive labels
        Output:
            logits, targets
        """
        feature1 = nn.functional.normalize(feature1, dim=1)
        feature2 = nn.functional.normalize(feature2, dim=1)
        batch_size = feature1.shape[0]

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [feature1, feature2]).unsqueeze(-1)
        # # negative logits: NxK
        # l_neg = torch.Tensor(batch_size, (batch_size-1)*2).cuda()
        # for i in range(0, batch_size):
        #     neg_samples = self.get_negatives(feature1, feature2, labels, i)
        #     l_neg[i] = torch.einsum('c,kc->k', [feature1[i], neg_samples])

        # # logits: Nx(1+K)
        # logits = torch.cat([l_pos, l_neg], dim=1)

        # # apply temperature
        # logits /= self.T

        # labels: positive key indicators
        # new_labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # criteria = nn.CrossEntropyLoss()

        distance = torch.einsum('nc,kc->nk', [feature1, feature2])
        logits = distance / self.T
        # print(logits)
        pos_labels = torch.zeros(logits.shape, dtype=torch.long).cuda()
        neg_labels = torch.zeros(logits.shape, dtype=torch.long).cuda()
        for i in range(0, batch_size):
            pos_labels[i] = labels.eq(labels[i]).float()
            neg_labels[i] = torch.where(pos_labels[i] == 1, torch.full_like(pos_labels[i], 0), torch.full_like(pos_labels[i], 1))
            # print(pos_labels[i])
            # print(neg_labels[i])
            
        # self.loss = criteria(logits, new_labels)
        logits = torch.exp(logits)
        # print(logits)
        # 原始距离度量值
        d_pos = torch.mul(distance, pos_labels).sum(dim=1)
        d_neg = torch.mul(distance, neg_labels).sum(dim=1)
        # 自然对数度量值
        l_pos = torch.mul(logits, pos_labels).sum(dim=1)
        l_neg = torch.mul(logits, neg_labels).sum(dim=1)
        # l_total = logits.sum(dim=1)
        # print(pos_labels)
        # print(neg_labels)
        # print(l_pos)
        # print(l_neg)
        # print(l_total)
        # print(l_pos + l_neg)
        # print(l_neg)
        self.loss = -torch.log(l_pos / (l_neg + l_pos)).sum() / batch_size# - torch.log((d_pos.sum() / torch.sum(pos_labels == 1)).pow(2))
        # self.loss = -torch.log(1 / (l_neg + 1)).sum() / batch_size - torch.log(l_pos / (1/self.T + l_pos)).sum() / batch_size
        # print(torch.sum(new_labels == 1))
        # print((d_pos.sum() / torch.sum(new_labels == 1)).pow(2))

        return self.loss

class NCE_Loss2(nn.Module):
    """
    NCE loss.
    By gqc
    """
    def __init__(self, gamma, keep_weight, T=0.07, mlp=False):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(NCE_Loss2, self).__init__()
        self.gamma = gamma
        self.T = T
        self.loss = 0
        self.sigmoid = nn.Sigmoid()
        self.keep_weight = keep_weight
        # self.relu = nn.ReLU(inplace=False)

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

    # '''
    def forward(self, feature1, feature2, labels):
        """
        Input:
            feature1: a batch of query images features
            feature2: a batch of key images features
            labels: a batch of positive labels
        Output:
            logits, targets
        """
        feature1 = nn.functional.normalize(feature1, dim=1)
        feature2 = nn.functional.normalize(feature2, dim=1)
        batch_size = feature1.shape[0]
        #特征相似度
        similarity = torch.einsum('nc,kc->nk', [feature1, feature2])#nan
        similarity = similarity.clamp(-1, 1)
        
        logits = similarity / self.T
        logits = torch.exp(logits) #这也是nan

        labels_temp = labels.unsqueeze(1)
        pos_labels = torch.eq(labels_temp, labels_temp.T).long().cuda()#正样本label
        neg_labels = torch.eq(pos_labels, 0).long().cuda()
        mask_pos = pos_labels.clone().float().cuda()#正样本mask
        mask_neg = neg_labels.clone().float().cuda()
        mask_pos *= self.keep_weight #label乘以倍数
        mask_neg *= self.keep_weight
        # self.loss = criteria(logits, new_labels)
        
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
        # diff_neg = torch.mul(diff_neg, neg_labels)
        diff_pos += mask_pos
        diff_neg += mask_neg 
        # pri
        l_pos = torch.mul(logits, diff_pos).sum(dim=1) #+ float(1e-8) 
        l_neg = torch.mul(logits, diff_neg).sum(dim=1) #+ float(1e-8)

        self.loss = -torch.log((l_pos) / (l_neg + l_pos)).sum() / batch_size# - torch.log((d_pos.sum() / torch.sum(pos_labels == 1)).pow(2))
        
        return self.loss
    '''
    def forward(self, feature1, feature2, labels):
        """
        Input:
            feature1: a batch of query images features
            feature2: a batch of key images features
            labels: a batch of positive labels
        Output:
            logits, targets
        """
        feature1 = nn.functional.normalize(feature1, dim=1)
        feature2 = nn.functional.normalize(feature2, dim=1)
        batch_size = feature1.shape[0]

        similarity = torch.einsum('nc,kc->nk', [feature1, feature2])#nan
        similarity = self.relu(similarity)
       
        labels_temp = labels.unsqueeze(1)
        pos_labels = torch.eq(labels_temp, labels_temp.T).long().cuda()#正样本label
        neg_labels = torch.eq(pos_labels, 0).long().cuda()
        mask_pos = pos_labels.clone().float().cuda()#正样本mask
        mask_neg = neg_labels.clone().float().cuda()
        mask_pos *= self.keep_weight #label乘以倍数
        mask_neg *= self.keep_weight

        # 相似度和label的乘积
        sim_pos = torch.mul(similarity, pos_labels)
        sim_neg = torch.mul(similarity, neg_labels)
        # 正负样本困难度(权重)
        diff_pos = pos_labels - sim_pos
        diff_neg = sim_neg
        # 正负样本权重
        diff_pos = torch.pow(diff_pos, self.gamma) #exponent小数没有问题
        diff_neg = torch.pow(diff_neg, self.gamma)
        diff_pos = torch.mul(diff_pos, pos_labels)#保证diff对应好正负样本
        diff_neg = torch.mul(diff_neg, neg_labels)
        diff_pos += mask_pos
        diff_neg += mask_neg
        #focal权重
        focal_weight = (diff_pos.sum(dim=1) + diff_neg.sum(dim=1)) / batch_size
        
        logits = similarity / self.T
        logits = torch.exp(logits) #这也是nan
        # print('-----logits------\n', logits, '\n') 
        # 相似度和label的乘积
        l_pos = torch.mul(logits, pos_labels).sum(dim=1)
        l_neg = torch.mul(logits, neg_labels).sum(dim=1)
        # 正负样本困难度
        
        # l_pos = torch.mul(logits, diff_pos).sum(dim=1) #+ float(1e-8) 
        # l_neg = torch.mul(logits, diff_neg).sum(dim=1) #+ float(1e-8)
        
        self.loss = torch.mul(-torch.log((l_pos) / (l_neg + l_pos)),focal_weight).sum() / batch_size# - torch.log((d_pos.sum() / torch.sum(pos_labels == 1)).pow(2))
        
        return self.loss
    '''

if __name__ == '__main__':
    args = params.get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # loss = NCE_Loss().cuda()
    loss = NCE_Loss2(gamma=1, keep_weight=0.1)

    feature1 = torch.tensor([[1.,2.,3.], [1.,-9.,3.], [3.,2.,-3.]]).cuda()
    feature2 = torch.tensor([[1.,2.,1.], [1.,-9.,5.], [4.,2.,-3.]]).cuda()
    label = torch.tensor([1, 2, 3]).cuda()
    
    out = loss(feature1, feature2, label)

    print(out)