from typing import List, Tuple
import shutil
import time
import numpy as np
import os
import json
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os
import torch
from sklearn.pipeline import FeatureUnion
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.loaders.subject import Subject


from sleeplearning.lib.model import Net


class SleepLearningDataset(object):
    """Sleep Learning dataset."""

    def __init__(self, foldr: str, class_remapping: dict, transform = None):
        self.dir = '../data/processed/' + foldr + '/'
        self.X = [filename for filename in os.listdir(self.dir) if filename.endswith(".npz")]
        self.class_remapping = class_remapping

        self.transform = transform
        with open(self.dir +'dataset_info.json') as data_file:
            self.dataset_info = json.load(data_file)

        remapped_distribution = {}
        for k in class_remapping.values():
            remapped_distribution[k] = 0
        for k, v in self.dataset_info['class_distribution'].items():
            remapped_distribution[class_remapping[k]] += v
        self.dataset_info['class_distribution'] = remapped_distribution

        self.class_int_mapping = {}
        for i, k in enumerate(self.dataset_info['class_distribution'].keys()):
            self.class_int_mapping[k] = i

        self.weights = np.zeros(len(self.dataset_info['class_distribution'].keys()))
        total_classes = 0.0
        for i, (k, v) in enumerate(self.dataset_info['class_distribution'].items()):
            total_classes += v
            self.weights[i] = v+1  # smooth
        self.weights = total_classes/self.weights

    def __getitem__(self, index):
        sample = np.load(self.dir + self.X[index])

        x = sample['x']
        y_str = Subject.sleep_stages_labels[int(sample['y'])]
        y_str_remapped = self.class_remapping[y_str]
        y_ = int(self.class_int_mapping[y_str_remapped])

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x).double()

        return x, y_

    def __len__(self):
        return len(self.X)


def train_epoch(train_loader: DataLoader, model: Net, criterion, optimizer,
                epoch, cuda, log_interval):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).long()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        (prec1, prec2), _ = accuracy(output, target, topk=(1, 2))
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

        if batch_idx % log_interval == 0:
            print('Epoch: [{0}]x[{1}/{2}]\t'
                 'Time {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                  'Loss {loss.val:.1f} ({loss.avg:.1f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@2 {top2.val:.2f} ({top2.avg:.2f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top2=top2))


def validation_epoch(val_loader: DataLoader, model: Net, criterion, cuda: bool) \
        -> Tuple[float, np.array]:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    prediction = np.array([])
    model.eval()
    end = time.time()
    predictions = []
    for data, target in val_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(
            target).long()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        (prec1, prec2), prediction = accuracy(output, target, topk=(1, 2))
        predictions.append(prediction)
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
    predictions = torch.cat(predictions)
    return top1.avg, predictions


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


def accuracy(output, target, topk=(1,)) -> Tuple[List[List[float]], torch.autograd.Variable]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # TODO: keep it as torch Variable and only convert for output if needed
        res.append((correct_k.mul_(100.0 / batch_size)).data.cpu().numpy())
    return res, pred[0]


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


def save_checkpoint(state, is_best, filename=root_dir + '/models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, root_dir+ '/models/model_best.pth.tar')