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
    def __init__(self, beta, T=0.07, mlp=False):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(NCE_Loss2, self).__init__()
        self.beta = beta
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
        # print('distance\n', distance)
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
        # 相似度和label的乘积
        sim_pos = torch.mul(distance, pos_labels)
        sim_neg = torch.mul(distance, neg_labels)
        # 正负样本困难度
        diff_pos = pos_labels - sim_pos
        diff_neg = sim_neg
        # 困难度幂次方
        diff_pos = torch.pow(diff_pos, self.beta)
        # n_pos = int(pos_labels.sum(1).sum(0))
        # diff_pos_sum = float(diff_pos.sum(1).sum(0))
        # diff_pos_avg = float(diff_pos_sum / n_pos)
        diff_neg = torch.pow(diff_neg, self.beta)
        # diff_pos += pos_labels
        # diff_neg += neg_labels 
        # print('diff_pos    diff_neg')
        # print(diff_pos.cpu().numpy(), '\n', diff_neg.cpu().numpy())
        # print(logits)
        # 原始距离度量值
        d_pos = torch.mul(distance, pos_labels).sum(dim=1)
        d_neg = torch.mul(distance, neg_labels).sum(dim=1)
        # 自然对数度量值
        # l_pos = torch.mul(logits, pos_labels).sum(dim=1)
        # l_neg = torch.mul(logits, neg_labels).sum(dim=1) * diff_pos_avg
        l_pos = torch.mul(logits, diff_pos).sum(dim=1)
        l_neg = torch.mul(logits, diff_neg).sum(dim=1)
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


if __name__ == '__main__':
    args = params.get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # loss = NCE_Loss().cuda()
    loss = NCE_Loss2(beta=1)

    feature1 = torch.tensor([[1.,2.,3.], [2.,5.,6.]]).cuda()
    feature2 = torch.tensor([[3.,5.,3.], [2.,7.,11.]]).cuda()
    test_label = torch.tensor([1,2]).cuda()
    # print(feature1.shape)
    # print(test_label)
    # feature1 = nn.functional.normalize(feature1, dim=1)
    # feature2 = nn.functional.normalize(feature2, dim=1)
    # print(feature1)
    # label_new = torch.where(test_label == 1, 1, 0)
    # print(label_new)
    # result =test_data2.eq(test_data1).float()
    # print(result)
    # print(test_data1[0])
    # l_neg = torch.randn(3, 4).cuda()
    # for i in range(0, test_data1.shape[0]):
    #     neg = get_negatives(test_data1, test_data2, i)
    #     l_neg_tmp = torch.einsum('c,kc->k', [test_data1[i], neg])
    # for i in range(4):
    #     index = torch.where(test_label == test_label[i])
    #     print(index)
    #     pos = feature2.index_select(0, index[0])
    #     print(pos)
    out = loss(feature1, feature2, test_label)

    print(out)