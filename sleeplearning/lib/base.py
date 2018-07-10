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
import sleeplearning.lib.utils as utils


class Base(object):
    def __init__(self, logger=None, cuda=None, verbose=False):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.arch = None
        self.ms = None
        self.ds = None
        self.logger = logger
        self.nepoch = -1
        self.cudaEfficient = cuda
        self.verbose = verbose
        self.tenacity = 10000
        self.best_acc_ = None
        self.last_acc = None

        if self.cudaEfficient:
            self.remap_storage = lambda storage, loc: storage.cuda(0)
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            # remap storage to CPU (needed for model load if no GPU avail.)
            self.remap_storage = lambda storage, loc: storage
            self.kwargs = {}

    def fit(self, arch, ms, data_dir, loader, train_csv, val_csv, channels,
            nclasses, fold,
            nbrs, batch_size_train, batch_size_val, oversample):

        ldr = utils.get_loader(loader)

        print("\nTRAINING SET: ")
        train_ds = utils.SleepLearningDataset(data_dir, train_csv, fold,
                                              nclasses,
                                              FeatureExtractor(
                                                  channels).get_features(),
                                              nbrs,
                                              ldr, verbose=self.verbose)

        print("\nVAL SET: ")
        val_ds = utils.SleepLearningDataset(data_dir, val_csv, fold,
                                            nclasses,
                                            FeatureExtractor(
                                                channels).get_features(),
                                            nbrs,
                                            ldr, verbose=self.verbose)

        print("TRAIN LOADER:")
        train_loader = utils.get_sampler(train_ds, batch_size_train,
                                         oversample, self.cudaEfficient,
                                         verbose=True)
        print("\nVAL LOADER:")
        val_loader = utils.get_sampler(val_ds, batch_size_val,
                                       False,
                                       self.cudaEfficient, verbose=True)
        print("\n\n")

        # save parameters for model reloading
        ms['input_dim'] = train_ds.dataset_info['input_shape']
        ms['nclasses'] = nclasses
        self.arch = arch
        self.ms = ms
        self.ds = {
            'loader': loader,
            'channels': channels,
            'nbrs': nbrs,
            'nclasses': nclasses,
            'train_dist': list(train_ds.dataset_info['class_distribution']
                               .values())
        }
        self.model = utils.get_model_arch(arch, ms)
        self.criterion, self.optimizer = utils.get_model(self.model, ms,
                                                         self.ds['train_dist'],
                                                         self.cudaEfficient)


        # Training loop
        self.nepoch = 1
        bestmodel = copy.deepcopy(self.model)
        bestaccuracy = -1
        stop_train = False
        while not stop_train and self.nepoch <= ms['epochs']:
            tr_loss, tr_acc, tr_tar, tr_pred = self.trainepoch(train_loader, self.nepoch)
            val_loss, self.last_acc, val_tar, val_pred = self.score(val_loader)
            # log accuracy and confusion matrix
            if self.logger is not None:
                self.logger.scalar_summary('acc/train', tr_acc, self.nepoch)
                self.logger.scalar_summary('loss/train', tr_loss,
                                           self.nepoch)
                self.logger.scalar_summary('acc/val', self.last_acc, self.nepoch)
                self.logger.scalar_summary('loss/val', val_loss, self.nepoch)
                np.savez(
                    os.path.join(self.logger.log_dir, 'pred_train_last.npz'),
                    predictions=np.array(tr_pred), targets=np.array(tr_tar))
                np.savez(
                    os.path.join(self.logger.log_dir, 'pred_val_last.npz'),
                    predictions=np.array(val_pred), targets=np.array(val_tar))

                if self.last_acc > bestaccuracy or self.nepoch == ms['epochs']:

                    self.save_checkpoint_({
                        'epoch': self.nepoch,
                        'arch': self.arch,
                        'ms': self.ms,
                        'ds': self.ds,
                        'state_dict': self.model.state_dict(),
                        'best_prec1': max(bestaccuracy, self.last_acc),
                        'optimizer': self.optimizer.state_dict(),
                    }, self.last_acc > bestaccuracy,
                                          self.logger.log_dir)

                if self.last_acc > bestaccuracy:
                    shutil.copyfile(
                        os.path.join(self.logger.log_dir, 'pred_train_last.npz'),
                        os.path.join(self.logger.log_dir, 'pred_train_best.npz'))
                    shutil.copyfile(
                        os.path.join(self.logger.log_dir, 'pred_val_last.npz'),
                        os.path.join(self.logger.log_dir, 'pred_val_best.npz'))

                    bestmodel = copy.deepcopy(self.model)
                    self.logger.cm_summary(tr_pred, tr_tar, 'cm/train',
                                           self.nepoch,
                                           ['W', 'N1', 'N2', 'N3', 'REM'])
                    self.logger.cm_summary(val_pred, val_tar, 'cm/val',
                                           self.nepoch,
                                           ['W', 'N1', 'N2', 'N3', 'REM'])

            bestaccuracy = max(bestaccuracy, self.last_acc)
            self.nepoch += 1

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

        print('Train: [{0}]x[{1}/{1}]\t'
              'Time {batch_time.sum:.1f}\t'
              'Loss {loss.avg:.2f}\t'
              'Prec@1 {top1.avg:.2f}\t'
              'Prec@2 {top2.avg:.2f}'.format(
            epoch, len(train_loader), batch_time=batch_time,
            loss=losses, top1=top1, top2=top2))

        return losses.avg, top1.avg, targets, predictions

    def score(self, val_loader: DataLoader) -> Tuple[float, float, np.array,
                                                     np.array]:
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
            data, target = Variable(data).float(), Variable(target).long()

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

        print('Val:  [{0}/{0}]\t\t'
              'Time {batch_time.sum:.1f}\t'
              'Loss {loss.avg:.2f}\t'
              'Prec@1 {top1.avg:.2f}\t'
              'Prec@2 {top2.avg:.2f}'.format(
            len(val_loader), batch_time=batch_time,
            loss=losses, top1=top1, top2=top2))

        return losses.avg, top1.avg, targets, predictions

    def predict(self, subject_path):
        import tempfile
        subject_path = os.path.normpath(os.path.abspath(subject_path))
        data_dir, subject_name = os.path.split(subject_path)
        temp_path = tempfile.mkdtemp()
        val_csv = os.path.join(temp_path, 'tmp_csv')
        np.savetxt(val_csv, np.array([subject_name]), delimiter=",", fmt='%s')

        ldr = utils.get_loader(self.ds['loader'])
        test_ds = utils.SleepLearningDataset(data_dir, val_csv, 0,
                                            self.ms['nclasses'],
                                            FeatureExtractor(
                                                self.ds['channels'])
                                                .get_features(),
                                            self.ds['nbrs'], ldr,
                                             verbose=self.verbose)

        test_loader = utils.get_sampler(test_ds, 100, False,
                                        self.cudaEfficient, verbose=True)
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

        self.ds = checkpoint['ds']
        self.ms = checkpoint['ms']
        self.arch = checkpoint['arch']
        self.nepoch = checkpoint['epoch']
        self.best_acc_ = checkpoint['best_prec1']
        self.model = utils.get_model_arch(self.arch, self.ms)
        self.criterion, self.optimizer = utils.get_model(self.model, self.ms,
                                                         self.ds['train_dist'],
                                                         self.cudaEfficient)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.verbose:
            print("=> loaded model (epoch {}), Prec@1: {}"
                  .format(self.nepoch, self.best_acc_))

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


    def save_checkpoint_(self, state, is_best, dir):
        filename = os.path.join(dir, 'checkpoint.pth.tar')
        torch.save(state, filename)
        if self.logger is not None:
            self.logger._run.add_artifact(filename)
        if is_best:
            best_name = os.path.join(dir, 'model_best.pth.tar')
            shutil.copyfile(filename, best_name)
            if self.logger is not None:
                self.logger._run.add_artifact(best_name)