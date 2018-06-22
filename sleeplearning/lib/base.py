import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.feature_extractor import *
import shutil
import time
import numpy as np
import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple
from sleeplearning.lib.utils import AverageMeter


class Base(object):
    def __init__(self, model, optimizer, criterion, logger, cuda):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.cudaEfficient = cuda

        self.tenacity = 100
        self.best_acc_ = 0

        if self.cudaEfficient:
            self.remap_storage = lambda storage, loc: storage.cuda(0)
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
            self.model.cuda()
            self.criterion.cuda()
        else:
            # remap storage to CPU (needed for model load if no GPU avail.)
            self.remap_storage = lambda storage, loc: storage
            self.kwargs = {}

    def fit(self, train_loader, val_loader, max_epoch = 10,
            early_stop=True):
        self.nepoch = 1
        bestmodel = copy.deepcopy(self.model)
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Training
        while not stop_train and self.nepoch <= max_epoch:
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
        predictions = np.array([], dtype=int)
        targets = np.array([], dtype=int)

        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.cudaEfficient:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).long()

            # compute output
            output = self.model(data)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            (prec1, prec2), prediction = self.accuracy_(output, target, topk=(1, 2))

            predictions = np.append(predictions, prediction)
            targets = np.append(targets, target.data.cpu().numpy())

            losses.update(loss.item(), data.size(0))
            top1.update(prec1, data.size(0))
            top2.update(prec2, data.size(0))

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # log accuracy and confusion matrix
        if self.logger is not None:
            self.logger.cm_summary(predictions, targets, 'cm/train',
                                   self.nepoch,
                                   ['W', 'N1', 'N2', 'N3', 'REM'])
            self.logger.scalar_summary('acc/train', top1.avg, self.nepoch)
            self.logger.scalar_summary('loss/train', losses.avg,
                                       self.nepoch)

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
            targets = np.append(targets, target.data.cpu().numpy())
            losses.update(loss.item(), data.size(0))
            top1.update(prec1, data.size(0))
            top2.update(prec2, data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # log accuracy and confusion matrix
        if self.logger is not None:
            self.logger.cm_summary(predictions, targets, 'cm/val', self.nepoch-1, ['W','N1','N2','N3', 'REM'])
            self.logger.scalar_summary('acc/val', top1.avg, self.nepoch-1)
            self.logger.scalar_summary('loss/val', losses.avg, self.nepoch - 1)

        print('Val:  [{0}/{0}]\t\t'
              'Time {batch_time.sum:.1f}\t'
              'Loss {loss.avg:.2f}\t'
              'Prec@1 {top1.avg:.2f}\t'
              'Prec@2 {top2.avg:.2f}'.format(
            len(val_loader), batch_time=batch_time,
            loss=losses, top1=top1, top2=top2))

        return top1.avg, predictions

    def predict(self, test_loader):
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
