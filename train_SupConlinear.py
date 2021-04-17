
# from random import sample, shuffle
import time
import os
from tqdm import tqdm
import logging
import torch
# from torch._C import Value
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
#---------------
from params import get_args
from model.ResNet_big import SupConResNet, LinearClassifier
from utils import AverageMeter, adjust_learning_rate, TwoCropTransform, mkdir
from utils import LOG, warmup_learning_rate, accuracy
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
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    mkdir(args.data_folder)
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=args.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=args.data_folder,
                                        train=False,
                                        transform=val_transform)                                
    else:
        raise ValueError(args.dataset)
    
    log.info('------Building data_loader--------')
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True
    )
    return train_loader, val_loader

def set_model(args, log, ckpt_path):
    log.info('-------Building model:{}, criterion------'.format(args.model))
    model = SupConResNet()
    # criterion = NCE_Loss2(beta=args.beta, T=.1)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=args.model, num_classes=args.n_cls)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if args.gpu is not None and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion

def set_optimizer(args, model, log):
    log.info('-------Building optimizer------')
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

def validate(val_loader, model, classifier, criterion, args):
    model.eval()
    classifier.eval()

    # loss_tracker = AverageMeter('eval loss', ':.4e')
    top1_tracker = AverageMeter('top1', ':.3e')
    tqdm_batch = tqdm(val_loader, desc='-----eval----')
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm_batch):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            #forward
            output = classifier(model.encoder(images))
            
            #update metric
            acc1 = accuracy(output, labels, topk=(1,))
            top1_tracker.update(acc1[0].item(), bsz)

    tqdm_batch.close()

    return top1_tracker.avg

def main():
    best_acc = 0
    args = get_args()
    # 检查相应文件夹是否已经创建
    mkdir('./ckpt')
    mkdir(args.log_dir)
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    log_path =args.log_dir + '/'+ args.model + '_' + args.dataset + '_' + time_now + "_linear_lr" + str(args.lr) + "_batchsize" + str(args.batch_size)
    mkdir(log_path)
    logger = LOG(log_path)
    # 指定使用的gpu
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.device_count() > 1:
        logger.info('----lets use multi GPUS----')
    logger.info('-----Using GPU:{}.-----'.format(args.gpu))
    # 数据集 模型 损失 优化器定义
    train_loader, val_loader = set_loader(args, logger)
    ckpt_path = './log/Resnet50_cifar10_20210406-2057_contrastive_lr0.01_batchsize64/ckpt_epoch_450.pth'
    model, classifier, criterion = set_model(args, logger, ckpt_path)
    optimizer = set_optimizer(args, classifier, logger)
    writer = SummaryWriter(log_dir=log_path)
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        model.eval()
        classifier.train()

        tqdm_batch = tqdm(train_loader, desc='------train: Epoch{}-----'.format(epoch))
        loss_tracker = AverageMeter('loss', ':.4e')
        top1 = AverageMeter('top1', ':.3e')

        for idx, (images, labels) in enumerate(tqdm_batch):
            
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
            # compute loss
            with torch.no_grad():
                features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update metric
            loss_tracker.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0].item(), bsz)

        tqdm_batch.close()
        logger.info("-----Loss: {:.4f} ({:.4f})"\
                     .format(loss_tracker.val, loss_tracker.avg))
        logger.info("-----train_top1: {top1.val:.3f} ({top1.avg:.3f})"\
                     .format(top1=top1))
        logger.info('------Lr: {:.8f}'.format(optimizer.param_groups[0]['lr']))
        writer.add_scalar('Loss', loss_tracker.avg, epoch)
        writer.add_scalar('train_top1', top1.avg, epoch)

        val_acc = validate(val_loader, model, classifier, criterion, args)
        if val_acc > best_acc:
            best_acc = val_acc
        logger.info('---val_acc: {}, best_acc: {}, epoch: {}-----'\
                    .format(val_acc, best_acc, epoch))
        writer.add_scalar('val_acc', val_acc, epoch)

    logger.info('the best accuracy : {:.2f}'.format(best_acc))
    writer.close()

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