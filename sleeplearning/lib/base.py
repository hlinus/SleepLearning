import os
import sys
import shutil
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
#root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
#sys.path.insert(0, root_dir)
from sleeplearning.lib.feature_extractor import *
#from sleeplearning.lib.utils import utils
from sleeplearning.lib import utils
import numpy as np
#from sleeplearning.lib.granger_loss import GrangerLoss
#from sleeplearning.lib.transforms import SensorDropout


class Base(object):
    def __init__(self, logger=None, cuda=None, verbose=False):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.arch = None
        self.ms = None
        self.ds = None
        self.fold = None
        self.logger = logger
        self.nepoch = -1
        self.cudaEfficient = cuda
        self.verbose = verbose
        self.tenacity = 5
        self.best_acc_ = None
        self.best: dict = None
        self.last_acc = None

        if self.cudaEfficient:
            self.remap_storage = lambda storage, loc: storage.cuda(0)
            self.kwargs = {'num_workers': 4, 'pin_memory': False}
        else:
            # remap storage to CPU (needed for model load if no GPU avail.)
            self.remap_storage = lambda storage, loc: storage
            self.kwargs = {}

    def fit(self, arch, ms, data_dir, loader, train_csv, val_csv, channels,
            nclasses, fold,
            nbrs, batch_size_train, batch_size_val, oversample,
            early_stop=False):

        ldr = utils.get_loader(loader)

        sensor_dropout = None #SensorDropout((.5,.5,.5,.5))

        print("\nTRAINING SET: ", end="")
        train_ds = utils.SleepLearningDataset(data_dir, train_csv, fold,
                                              nclasses,
                                              FeatureExtractor(
                                                  channels).get_features(),
                                              nbrs,
                                              ldr, discard_arts=True,
                                              transform=sensor_dropout,
                                              verbose=self.verbose)

        print("\nVAL SET: ", end="")
        val_ds = utils.SleepLearningDataset(data_dir, val_csv, fold,
                                            nclasses,
                                            FeatureExtractor(
                                                channels).get_features(),
                                            nbrs,
                                            ldr,
                                            verbose=self.verbose)

        print("\nTRAIN LOADER:")
        train_loader = utils.get_sampler(train_ds, batch_size_train,
                                         oversample, True, self.kwargs,
                                         verbose=True)
        print("\nVAL LOADER:")
        val_loader = utils.get_sampler(val_ds, batch_size_val,
                                       False, False, self.kwargs,
                                       verbose=True)

        # save parameters for model reloading
        ms['input_dim'] = train_ds.dataset_info['input_shape']
        ms['nclasses'] = nclasses
        self.arch = arch
        self.ms = ms
        self.fold = fold
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

        if self.cudaEfficient:
            self.model.cuda()
            self.criterion.cuda()

        print("\nMODEL:")
        print(self.model)
        nbr_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print("\n # OF TRAINABLE PARAMETERS:", nbr_trainable_params)
        nbr__non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad)
        print("\n # OF FROZEN PARAMETERS:", nbr__non_trainable_params)
        print("\n\n")

        # Training loop
        self.nepoch = 0
        bestmodel = copy.deepcopy(self.model)
        bestopt = copy.deepcopy(self.optimizer)
        bestaccuracy = -1
        bestepoch = -1
        stop_train = False
        early_stop_count = 0
        self.tenacity = 5

        while not stop_train and self.nepoch < ms['epochs']:
            self.nepoch += 1
            tr_metrics, tr_tar, tr_pred = self.trainepoch_(train_loader,
                                                           self.nepoch)
            val_metrics, val_tar, val_pred = self.valepoch_(
                val_loader)

            # log accuracy and confusion matrix
            if self.logger is not None:
                for tag, val in tr_metrics.items():
                    if val is not None:
                        self.logger.scalar_summary(tag+'/train', val.avg,
                                                   self.nepoch)

                for tag, val in val_metrics.items():
                    if val is not None:
                        self.logger.scalar_summary(tag+'/val', val.avg,
                                                   self.nepoch)
                self.last_acc = val_metrics['top1'].avg
                if self.last_acc > bestaccuracy:
                    self.logger.cm_summary(tr_pred, tr_tar, 'cm/train',
                                           self.nepoch,
                                           ['W', 'N1', 'N2', 'N3', 'REM'])
                    self.logger.cm_summary(val_pred, val_tar, 'cm/val',
                                           self.nepoch,
                                           ['W', 'N1', 'N2', 'N3', 'REM'])

            # handling early stop
            if self.last_acc > bestaccuracy:
                bestaccuracy = self.last_acc
                bestmodel = copy.deepcopy(self.model)
                bestopt = copy.deepcopy(self.optimizer)
                bestepoch = self.nepoch
            elif early_stop:
                early_stop_count += 1
                print(f"Patience: {early_stop_count}")
                if early_stop_count >= self.tenacity:
                    stop_train = True
                    print("EARLY STOPPING!")

        self.best = {
            'model': bestmodel,
            'optimizer': bestopt,
            'accuracy': bestaccuracy,
            'epoch': bestepoch
        }
        self.best_acc_ = bestaccuracy

    def score(self, subject_path, probs=False) -> Tuple[float, np.ndarray,
                                                        np.ndarray]:
        import tempfile
        subject_path = os.path.normpath(os.path.abspath(subject_path))
        data_dir, subject_name = os.path.split(subject_path)
        temp_path = tempfile.mkdtemp()
        tmp_csv = os.path.join(temp_path, 'tmp_csv')
        np.savetxt(tmp_csv, np.array([subject_name]), delimiter=",", fmt='%s')
        test_loader = self.get_loader_from_csv_(data_dir, tmp_csv, 0, 100,
                                                False)
        output: dict = None
        metrics: dict = None
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                if self.cudaEfficient:
                    data, target = data.cuda(), target.cuda()
                # compute output
                batch_out, loss, metrics = self.predict_batch_(data, target,
                                                               metrics)
                if output is None:
                    output = {'y_true': [], 'y_probs': [], 'y_pred': []}
                output['y_probs'].append(F.softmax(batch_out['logits'], dim=1)
                                         .data.cpu().numpy())
                output['y_true'].append(target.data.cpu().numpy())
                output['y_pred'].append(batch_out['y_pred'])
        for k, v in output.items():
            output[k] = np.concatenate(output[k], axis=0)

        y_pred = output['y_probs'] if probs else output['y_pred']
        y_true = output['y_true']
        return metrics['top1'].avg, y_pred, y_true

    def predict(self, subject_path):
        import tempfile
        subject_path = os.path.normpath(os.path.abspath(subject_path))
        data_dir, subject_name = os.path.split(subject_path)
        temp_path = tempfile.mkdtemp()
        tmp_csv = os.path.join(temp_path, 'tmp_csv')
        np.savetxt(tmp_csv, np.array([subject_name]), delimiter=",", fmt='%s')
        test_loader = self.get_loader_from_csv_(data_dir, tmp_csv, 0, 100,
                                                False)
        self.model.eval()
        prediction = None
        with torch.no_grad():
            for data, target in test_loader:
                if self.cudaEfficient:
                    data, target = data.cuda(), target.cuda()
                # compute output
                output = self.model(data)
                _, pred = output.topk(1, 1, True, True)
                prediction = np.append(prediction, pred.data.cpu().numpy())
        return prediction.astype(int)

    def predict_proba(self, subject_path):
        import tempfile
        subject_path = os.path.normpath(os.path.abspath(subject_path))
        data_dir, subject_name = os.path.split(subject_path)
        temp_path = tempfile.mkdtemp()
        tmp_csv = os.path.join(temp_path, 'tmp_csv')
        np.savetxt(tmp_csv, np.array([subject_name]), delimiter=",", fmt='%s')
        test_loader = self.get_loader_from_csv_(data_dir, tmp_csv, 0, 100,
                                                False)
        self.model.eval()
        probas = None
        with torch.no_grad():
            for data, target in test_loader:
                if self.cudaEfficient:
                    data, target = data.cuda(), target.cuda()
                # compute output
                output = self.model(data)
                vals = F.softmax(output, dim=1).data.numpy()
                if probas is None:
                    probas = vals
                else:
                    probas = np.append(probas, vals, axis=0)
        return probas

    def restore(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir,
                                map_location=self.remap_storage)
        self.ds = checkpoint['ds']
        self.ms = checkpoint['ms']
        self.arch = checkpoint['arch']
        self.nepoch = checkpoint['epoch']
        self.best_acc_ = checkpoint['best_acc']
        self.last_acc = checkpoint['last_acc']
        self.model = utils.get_model_arch(self.arch, self.ms)
        # TODO: Make sure correct model restored (same modelstr?)
        self.criterion, self.optimizer = utils.get_model(self.model, self.ms,
                                                         self.ds['train_dist'],
                                                         self.cudaEfficient)

        self.model.load_state_dict(checkpoint['state_dict'])
        # Only load if optimizer is not null (average of trained models)
        if checkpoint['optimizer']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.cudaEfficient:
            self.model.cuda()
            self.criterion.cuda()

        if self.verbose:
            print("=> loaded model (epoch {}), Prec@1: {}"
                  .format(self.nepoch, self.best_acc_))

    def trainepoch_(self,
                    tr_loader: DataLoader,
                    epoch: int) -> Tuple[dict, np.ndarray, np.ndarray]:
        self.model.train()
        batch_time = utils.AverageMeter()
        metrics: dict = None
        predictions: np.array = np.array([], dtype=int)
        targets: np.array = np.array([], dtype=int)

        end = time.time()
        for batch_idx, (data, target) in enumerate(tr_loader):
            if self.cudaEfficient:
                data, target = data.cuda(), target.cuda()

            # compute output
            output, batchloss, metrics = self.predict_batch_(data, target,
                                                                metrics)

            predictions = np.append(predictions, output['y_pred'])
            targets = np.append(targets, target.data.cpu().numpy())

            # compute gradient and do Adam step if model has any trainable
            # parameters (not just averaging trained experts)
            if self.optimizer:
                self.optimizer.zero_grad()
                batchloss.backward()
                self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(f'Train: [{self.nepoch}]x[{len(tr_loader)}/{len(tr_loader)}]\t'
              f'Time {batch_time.sum:.1f}\t'
              f'Loss {metrics["loss"].avg:.2f}\t'
              f'Prec@1 {metrics["top1"].avg:.2f}\t'
              f'Prec@2 {metrics["top2"].avg:.2f}')

        return metrics, targets, predictions

    def valepoch_(self, val_loader: DataLoader) -> Tuple[dict,
                                                         np.ndarray,
                                                         np.ndarray]:
        batch_time = utils.AverageMeter()
        metrics: dict = None
        self.model.eval()

        end = time.time()
        predictions = np.array([], dtype=int)
        targets = np.array([], dtype=int)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if self.cudaEfficient:
                    data, target = data.cuda(), target.cuda()

                # compute output
                output, _, metrics = self.predict_batch_(data, target, metrics)

                predictions = np.append(predictions, output['y_pred'])
                targets = np.append(targets, target.data.cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        print(f'Val: [{self.nepoch}]x[{len(val_loader)}/{len(val_loader)}]\t'
              f'Time {batch_time.sum:.1f}\t'
              f'Loss {metrics["loss"].avg:.2f}\t'
              f'Prec@1 {metrics["top1"].avg:.2f}\t'
              f'Prec@2 {metrics["top2"].avg:.2f}')

        return metrics, targets, predictions

    def predict_batch_(self,
                       data,
                       target,
                       metrics: dict = None) -> Tuple[Dict, torch.Tensor, Dict]:
        batchout = self.model(data)
        batchloss = self.criterion(batchout['logits'], target)

        if metrics is None:
            metrics: dict = {'loss': utils.AverageMeter(), 'top1':
                utils.AverageMeter(),'top2': utils.AverageMeter()}

        # for granger loss ignore aux predictors after loss comp
        #if isinstance(self.criterion, GrangerLoss):
        #    for k, v in batchloss.items():
        #        metrics[k].update(v.item(), data.size(0))
        #else:
        metrics['loss'].update(batchloss.item(), data.size(0))

        # measure accuracy
        (prec1, prec2), y_pred = self.accuracy_(
            batchout['logits'], target, topk=(1, 2))
        batchout['y_pred'] = y_pred
        metrics['top1'].update(prec1, data.size(0))
        metrics['top2'].update(prec2, data.size(0))

        return batchout, batchloss, metrics

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

    def get_loader_from_csv_(self, data_dir, csv_path, fold, batch_size,
                             shuffle):
        ldr = utils.get_loader(self.ds['loader'])
        dataset = utils.SleepLearningDataset(data_dir, csv_path, fold,
                                             self.ms['nclasses'],
                                             FeatureExtractor(
                                                 self.ds['channels'])
                                             .get_features(),
                                             self.ds['nbrs'], ldr,
                                             verbose=self.verbose)

        data_loader = utils.get_sampler(dataset, batch_size, False, shuffle,
                                        self.kwargs,
                                        verbose=self.verbose)
        return data_loader

    def save_checkpoint_(self, save_best_only: bool = False):
        state = {
            'epoch': self.best['epoch'],
            'arch': self.arch,
            'ms': self.ms,
            'modelstr': str(self.best['model']),
            'ds': self.ds,
            'fold': self.fold,
            'state_dict': self.best['model'].state_dict(),
            'last_acc': self.best['accuracy'],
            'best_acc': self.best['accuracy'],
            'optimizer': self.best['optimizer'].state_dict() if self.best[
                'optimizer'] else None
        }

        best_name = os.path.join(self.logger.log_dir, 'model_best.pth.tar')
        torch.save(state, best_name)
        if self.logger._run is not None:
            self.logger._run.add_artifact(best_name)

        if not save_best_only:
            filename = os.path.join(self.logger.log_dir, 'checkpoint.pth.tar')
            state = {
                'epoch': self.nepoch,
                'arch': self.arch,
                'ms': self.ms,
                'modelstr': str(self.model),
                'ds': self.ds,
                'fold': self.fold,
                'state_dict': self.model.state_dict(),
                'last_acc': self.last_acc,
                'best_acc': self.best_acc_,
                'optimizer': self.optimizer.state_dict() if self.optimizer
                else None,
            }
            torch.save(state, filename)
            if self.logger._run is not None:
                self.logger._run.add_artifact(filename)