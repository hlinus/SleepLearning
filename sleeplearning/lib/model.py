"""
Sl Classifier class in the style of scikit-learn
based on https://github.com/facebookresearch/SentEval/blob/master/senteval/
tools/classifier.py
"""
from __future__ import absolute_import, division, unicode_literals

import os
import shutil
import time
import numpy as np
import copy
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_normal
from torch.utils.data import DataLoader
from typing import List, Tuple
from sleeplearning.lib.feature_extractor import FeatureExtractor
from sleeplearning.lib.logger import Logger
from sleeplearning.lib.utils import SleepLearningDataset, AverageMeter, \
    get_optimizer


class Overfit(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super(Overfit, self).__init__()
        kernel_size = 3
        padding = (kernel_size//2, kernel_size//2)
        self.conv1 = nn.Conv2d(input_shape[0], 48, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=padding)
        self.fc1 = nn.Linear(32*input_shape[1]//4*input_shape[2]//4, 8000)
        self.fc2 = nn.Linear(8000, 5000)
        self.fc3 = nn.Linear(5000, 3000)
        self.fc4 = nn.Linear(3000, 1000)
        self.fc5 = nn.Linear(1000, num_classes)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal(m.weight.data)
            xavier_normal(m.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


class SleepStage(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super(SleepStage, self).__init__()
        kernel_size = 3
        padding = (kernel_size // 2, kernel_size // 2)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=kernel_size,
                               padding=padding)
        self.fc1 = nn.Linear(32*(input_shape[1]//4)*(input_shape[2]//4), 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal(m.weight.data)
            xavier_normal(m.bias.data)

    def forward(self, x):
        x = F.max_pool2d(x, (2, 2))
        x = self.conv1(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.6, training=self.training)
        #x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc3(x))
        return x


class SlClassifier(object):
    def __init__(self, inputdim, nclasses, ts, seed, log_dir = None,
                 comment: str = None):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set general settings
        self.inputdim = inputdim
        self.nclasses = nclasses

        # train setings
        # TODO: select model according to ts.model
        self.model = SleepStage(input_shape=inputdim, num_classes=nclasses)
        # initialize layers with xavier
        self.model.weights_init()

        self.batch_size = ts['batch_size']
        self.max_epoch = ts['epochs']
        optim_fn, optim_params = get_optimizer(ts['optim'])
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.cudaEfficient = ts['cuda']

        # default values
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tenacity = 5
        self.best_acc_ = 0
        self.logger = Logger(log_dir)

        if self.cudaEfficient:
            self.remap_storage = lambda storage, loc: storage.cuda(0)
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
            self.model.cuda()
            self.criterion.cuda()
        else:
            # remap storage to CPU (needed for model load if no GPU avail.)
            self.remap_storage = lambda storage, loc: storage
            self.kwargs = {}

    def fit(self, train_ds: SleepLearningDataset, val_ds: SleepLearningDataset,
            early_stop=True):
        self.nepoch = 1
        bestmodel = copy.deepcopy(self.model)
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, **self.kwargs)

        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=True, **self.kwargs)

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(train_loader, self.nepoch)
            accuracy, _ = self.score(val_loader)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        self.best_acc_ = bestaccuracy

    def trainepoch(self, train_loader: DataLoader, epoch):
        self.model.train()
        torch.set_grad_enabled(True)
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top2 = AverageMeter()

        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.cudaEfficient:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).long()

            # compute output
            output = self.model(data)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            (prec1, prec2), _ = self.accuracy_(output, target, topk=(1, 2))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1, data.size(0))
            top2.update(prec2, data.size(0))

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log the loss
            if self.logger is not None:
                step = (self.nepoch - 1) * len(train_loader) + batch_idx
                self.logger.scalar_summary('loss/train',loss, step)
                self.logger.scalar_summary('acc/train', prec1, step)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Train: [{0}]x[{1}/{1}]\t'
              'Time {batch_time.sum:.1f}\t'
              'Loss {loss.avg:.2f}\t'
              'Prec@1 {top1.avg:.2f}\t'
              'Prec@2 {top2.avg:.2f}'.format(
            epoch, len(train_loader), batch_time=batch_time,
            loss=losses, top1=top1, top2=top2))

        self.nepoch += 1
        return losses.vals, top1.vals

    def score(self, val_loader: DataLoader) -> Tuple[float, np.array]:
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top2 = AverageMeter()
        prediction = np.array([])
        self.model.eval()
        torch.set_grad_enabled(False)
        end = time.time()
        predictions = np.array([], dtype=int)
        targets = np.array([], dtype=int)
        for batch_idx, (data, target) in enumerate(val_loader):
            if self.cudaEfficient:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(
                target).long()

            # compute output
            output = self.model(data)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            (prec1, prec2), prediction = self.accuracy_(output, target,
                                                        topk=(1, 2))
            predictions = np.append(predictions, prediction)
            targets = np.append(targets, target.data.numpy())
            losses.update(loss.item(), data.size(0))
            top1.update(prec1, data.size(0))
            top2.update(prec2, data.size(0))

            # log the loss and accuracy
            if self.logger is not None:
                step = (self.nepoch - 2)*len(val_loader)+batch_idx
                self.logger.scalar_summary('loss/val', loss, step)
                self.logger.scalar_summary('acc/val', prec1, step)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # log accuracy and confusion matrix
        if self.logger is not None:
            self.logger.cm_summary(targets, predictions, self.nepoch-1)
            self.logger.scalar_summary('acc/val_epoch', top1.avg, self.nepoch-1)

        print('Val:  [{0}/{0}]\t\t'
              'Time {batch_time.sum:.1f}\t'
              'Loss {loss.avg:.2f}\t'
              'Prec@1 {top1.avg:.2f}\t'
              'Prec@2 {top2.avg:.2f}'.format(
            len(val_loader), batch_time=batch_time,
            loss=losses, top1=top1, top2=top2))

        return top1.avg, predictions

    def predict(self, test_dir):
        test_ds = SleepLearningDataset(test_dir, self.nclasses,
                                       FeatureExtractor(
                                           self.feats).get_features(),
                                       self.neighbors)

        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, **self.kwargs)

        self.model.eval()
        prediction = np.array([])
        for data, target in test_loader:
            if self.cudaEfficient:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True).float(), Variable(
                target).long()
            # compute output
            output = self.model(data)
            _, pred = output.topk(1, 1, True, True)
            prediction = np.append(prediction, pred.data.cpu().numpy())
        return prediction.astype(int)

    def predict_proba(self, devX):
        # TODO: fix dummy implementation
        self.model.eval()
        probas = []
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
            if not probas:
                probas = vals
            else:
                probas = np.concatenate(probas, vals, axis=0)
        return probas

    def restore(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir,
                                map_location=self.remap_storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.feats = checkpoint['feats']
        print("=> loaded model (epoch {}), Prec@1: {}"
              .format(checkpoint['epoch'],
                      checkpoint['best_prec1']))

    @staticmethod
    def accuracy_(output, target, topk=(1,)) -> Tuple[
        List[float], torch.autograd.Variable]:
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        topk_acc = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            topk_acc.append(
                (correct_k.mul_(100.0 / batch_size)).data.cpu().numpy().item())
        return topk_acc, pred[0]

    @staticmethod
    def save_checkpoint_(state, is_best, dir):
        filename = os.path.join(dir, 'checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(dir, 'model_best.pth.tar'))


