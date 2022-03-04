from params import get_args

args = get_args()
print(args.focal)

import torch
device = 'cuda'
pos_labels = torch.tensor([[2., 3.], [1., .5]], device=device)
gamma = 0.2
pos_labels = torch.pow(pos_labels, exponent=gamma)