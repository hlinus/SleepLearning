import shutil
import time
import numpy as np
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)

from sleeplearning.lib.utils import SleepLearningDataset
from sleeplearning.lib.model import Net

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SleepLearning')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5 * 1e-5, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-weight-loss', action='store_true', default=False,
                    help='disables class weighted loss')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


def main():
    global best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        # remap to GPU0
        remap_storage = lambda storage, loc: storage.cuda(0)
    else:
        # remap storage to CPU (needed for model load)
        remap_storage = lambda storage, loc: storage
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    four_label = {'N1':'NREM', 'N2':'NREM', 'N3':'NREM', 'N4':'NREM', 'WAKE':'WAKE', 'REM':'REM', 'Artifact':'Artifact'}
    six_label = {'WAKE': 'WAKE', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'N4': 'Artifact', 'REM': 'REM', 'Artifact': 'Artifact'}
    train_ds = SleepLearningDataset('samplewise/train', six_label)
    val_ds = SleepLearningDataset('samplewise/test', six_label)
    print("TRAIN: ", train_ds.dataset_info)
    print("VAL: ", val_ds.dataset_info)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, **kwargs)

    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size,
                             shuffle=False, **kwargs)

    class_weights = torch.from_numpy(train_ds.weights).float() \
        if not args.no_weight_loss \
        else torch.from_numpy(np.ones(train_ds.weights.shape)).float()
    print("train class weights: ", class_weights)
    model = Net(num_classes=len(train_ds.dataset_info['class_distribution'].keys()))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(root_dir + args.resume):
            print("=> loading checkpoint '{}'".format(root_dir + args.resume))
            checkpoint = torch.load(root_dir + args.resume, map_location=remap_storage)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}), Prec@1: {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(root_dir + args.resume))

    if args.cuda:
        model.cuda()
        class_weights = class_weights.cuda()
        criterion.cuda()

    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, args)
        prec1 = validation(val_loader, model, criterion, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader: DataLoader, model: Net, criterion: F, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).long()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top2.update(prec2[0], data.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            print('Epoch: [{0}]x[{1}/{2}]\t'
                 'Time {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                  'Loss {loss.val:.1f} ({loss.avg:.1f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@2 {top2.val:.2f} ({top2.avg:.2f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top2=top2))


def validation(val_loader: DataLoader, model: Net, criterion: F, args) -> float:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()
    end = time.time()
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(
            target).long()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top2.update(prec2[0], data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test:  [{0}/{0}]\t\t'
          'Time ({batch_time.avg:.1f})\t'
          'Loss ({loss.avg:.1f})\t'
          'Prec@1 ({top1.avg:.2f})\t\t'
          'Prec@2 ({top2.avg:.2f})'.format(
        len(val_loader), batch_time=batch_time,
        loss=losses, top1=top1, top2=top2))
    return top1.avg


def test(test_loader: DataLoader, model: Net, cuda) -> np.array:
    model.eval()
    prediction = np.array([])
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(target).long()
        # compute output
        output = model(data)
        _, pred = output.topk(1, 1, True, True)
        prediction = np.append(prediction, pred.data.cpu().numpy())
    return prediction.astype(int)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).data.cpu().numpy())
    return res


def save_checkpoint(state, is_best, filename=root_dir + '/models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, root_dir+ '/models/model_best.pth.tar')


if __name__ == '__main__':
    main()