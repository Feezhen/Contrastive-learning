
# from random import sample, shuffle
import time
import os
from tqdm import tqdm
import torch
from torch._C import Value
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import logging
#---------------
from params import get_args
from model.ResNet_big import SupConResNet
from utils import AverageMeter, adjust_learning_rate, TwoCropTransform, mkdir
from utils import LOG, warmup_learning_rate
from loss.NCE_Loss import NCE_Loss2
from loss.SupConLoss import SupConLoss

logging.getLogger('tensorflow').disabled = True
def set_loader(args, log):
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not sopported')
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    mkdir(args.data_folder)
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    else:
        raise ValueError(args.dataset)
    
    log.info('------Building data_loader--------')
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    return train_loader

def set_model(args, log):
    log.info('-------Building model:{}, criterion------'.format(args.model))
    model = SupConResNet()
    criterion = NCE_Loss2(beta=args.beta, T=.1)
    # criterion = SupConLoss()
    if args.gpu is not None and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_optimizer(args, model, log):
    log.info('-------Building optimizer------')
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

def main():
    args = get_args()
    # 检查相应文件夹是否已经创建
    mkdir('./ckpt')
    mkdir(args.log_dir)
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    log_path =args.log_dir + '/'+ args.model + '_' + args.dataset + '_' + time_now + "_contrastive_lr" + str(args.lr) + "_batchsize" + str(args.batch_size)
    mkdir(log_path)
    logger = LOG(log_path)
    # 指定使用的gpu
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.device_count() > 1:
        logger.info('----lets use multi GPUS----')
    logger.info('-----Using GPU:{}.-----'.format(args.gpu))
    # 数据集 模型 损失 优化器定义
    train_loader = set_loader(args, logger)
    model, criterion = set_model(args, logger)
    optimizer = set_optimizer(args, model, logger)
    writer = SummaryWriter(log_dir=log_path)
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        model.train()
        tqdm_batch = tqdm(train_loader, desc='------train: Epoch{}-----'.format(epoch))
        loss_tracker = AverageMeter('loss', ':.4e')
        for idx, (images, labels) in enumerate(tqdm_batch):
            # images = torch.cat([images[0], images[1]], dim=0)
            # images = images.cuda()
            images1, images2 = images[0], images[1]
            images1, images2 = images1.cuda(), images2.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
            # compute loss
            features1 = model(images1)
            features2 = model(images2)
            # features = model(images)
            # features = torch.nn.functional.normalize(features, dim=1)
            # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features1, features2, labels)
            # loss = criterion(features, labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), 1)

        tqdm_batch.close()
        logger.info("-----Nces: {:.4f} ({:.4f})"\
                     .format(loss_tracker.val, loss_tracker.avg))
        writer.add_scalar('Loss', loss_tracker.avg, epoch)

        logger.info('------Lr: {:.8f}'.format(optimizer.param_groups[0]['lr']))

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                log_path, 'ckpt_epoch_{}.pth'.format(epoch)
            )
            save_model(model, optimizer, args, epoch, save_file)

    writer.close()
    save_file = os.path.join(
        log_path, 'ckpt_last_epoch.pth'
    )
    save_model(model, optimizer, args, epoch, save_file)

def save_model(model, optimizer, args, epoch, save_file):
    print('====>saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

if __name__ == '__main__':
    main()