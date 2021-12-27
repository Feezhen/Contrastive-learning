import torch
import torch.nn as nn

class Focal_loss(nn.Module):
    def __init__(self, gamma=2) -> None:
        super(Focal_loss, self).__init__()
        self.gamma = gamma
        self.loss = 0

    def forward(self, output, label):
        ce_loss = nn.functional.cross_entropy(output, label, reduction='none')
        pt = torch.exp(-ce_loss) #概率
        # batch求平均
        self.loss = ((1-pt) ** self.gamma * ce_loss).mean()

        return self.loss

if __name__ == '__main__':
    output = torch.tensor([[10., 2., 1.],
                            [6., 6., 1.]]).cuda()
    label = torch.tensor([0, 1]).cuda()
    focal_criterion = Focal_loss().cuda()
    loss = focal_criterion(output, label)
    print(loss)