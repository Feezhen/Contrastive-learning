import torch
from torch.utils import data
from tqdm import tqdm

def getStat(data_loader: data.DataLoader):
    """Get the mean and std value for the dataset."""
    assert data_loader.batch_size == 1, "batch_size should be 1"

    mean = torch.tensor(0.)
    std = torch.tensor(0.)
    tqdm_batch = tqdm(data_loader, desc='INFO: calculating data mean and std')
    for data1, _, _ in tqdm_batch:
        mean += torch.mean(data1[0, 0, :, :])
        std += torch.std(data1[0, 0, :, :])

    mean.div_(len(data_loader))
    std.div_(len(data_loader))
    print("mean:\n", mean)
    print("std:\n", std)
    tqdm_batch.close()

    return mean, std