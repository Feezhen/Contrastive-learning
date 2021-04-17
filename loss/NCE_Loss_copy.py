import torch
import torch.nn as nn
import numpy as np
import time

class NCE_Loss(nn.Module):
    """
    NCE loss.
    By gqc
    """
    def __init__(self, T=0.2, mlp=False):
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
        new_labels = torch.zeros(logits.shape, dtype=torch.long).cuda()
        for i in range(0, batch_size):
            new_labels[i] = labels.eq(labels[i]).float()
        #     # print(new_labels[i])
        # self.loss = criteria(logits, new_labels)
        logits = torch.exp(logits)
        # print(logits)
        d_pos = torch.mul(distance, new_labels).sum(dim=1)
        l_pos = torch.mul(logits, new_labels).sum(dim=1)
        # print(l_pos)
        l_neg = logits.sum(dim=1)
        # print(l_neg)
        self.loss = -torch.log(l_pos / l_neg).sum() / batch_size# - torch.log((d_pos.sum() / torch.sum(new_labels == 1)).pow(2))
        # print(torch.sum(new_labels == 1))
        # print((d_pos.sum() / torch.sum(new_labels == 1)).pow(2))

        return self.loss


if __name__ == '__main__':
    loss = NCE_Loss()

    feature1 = torch.tensor([[1.,2.,3.], [4.,5.,6.], [4.,5.,6.],[1.,2.,3.]]).cuda()
    feature2 = torch.tensor([[1.,2.,3.], [4.,5.,6.], [4.,5.,6.],[1.,2.,3.]],).cuda()
    test_label = torch.randn(4).cuda()
    print(feature1.shape)
    feature1 = nn.functional.normalize(feature1, dim=1)
    feature2 = nn.functional.normalize(feature2, dim=1)
    print(feature1)
    # result =test_data2.eq(test_data1).float()
    # print(result)
    # print(test_data1[0])
    # l_neg = torch.randn(3, 4).cuda()
    # for i in range(0, test_data1.shape[0]):
    #     neg = get_negatives(test_data1, test_data2, i)
    #     l_neg_tmp = torch.einsum('c,kc->k', [test_data1[i], neg])
    
    out = loss(feature1, feature2, test_label)

    print(out)