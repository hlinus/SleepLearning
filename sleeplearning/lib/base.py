import json
import os
import shutil
import sys
from typing import List
import numpy as np
import torch

from torch.utils.data import DataLoader
from scipy.signal import resample
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.loaders.subject import Subject
from sleeplearning.lib.utils import SleepLearningDataset, train_epoch, \
    validation_epoch, save_checkpoint
from sleeplearning.lib.model import Net
from sleeplearning.lib.feature_extractor import feats


class SleepLearning(object):
    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.manual_seed(1)
            # remap to GPU0
            self.remap_storage = lambda storage, loc: storage.cuda(0)
        else:
            # remap storage to CPU (needed for model load if no GPU avail.)
            self.remap_storage = lambda storage, loc: storage
        torch.manual_seed(1)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        three_label = {'N1': 'NREM', 'N2': 'NREM', 'N3': 'NREM', 'N4': 'NREM',
                      'WAKE': 'WAKE', 'REM': 'REM', 'Artifact': 'WAKE'}

        five_label = {'WAKE': 'WAKE', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3',
                      'N4': 'WAKE', 'REM': 'REM', 'Artifact': 'WAKE'}
        self.label_remapping = five_label

    def train(self, ts: dict):
        best_prec1 = 0
        train_dir = os.path.join('newDS', 'train')
        test_dir = os.path.join('newDS', 'test')
        train_ds = SleepLearningDataset(train_dir, self.label_remapping, feats)
        test_ds = SleepLearningDataset(test_dir, self.label_remapping, feats)
        print("TRAIN: ", train_ds.dataset_info)
        print("TEST: ", test_ds.dataset_info)

        train_loader = DataLoader(train_ds, batch_size=ts['batch-size'],
                                  shuffle=True, **self.kwargs)

        val_loader = DataLoader(test_ds, batch_size=ts['test-batch-size'],
                                shuffle=False, **self.kwargs)

        class_weights = torch.from_numpy(train_ds.weights).float() \
            if ts['weight-loss'] \
            else torch.from_numpy(np.ones(train_ds.weights.shape)).float()
        print("train class weights: ", class_weights)
        model = Net(
            num_classes=len(train_ds.dataset_info['class_distribution'].keys()))

        # initialize layers with xavier
        model.weights_init()

        # define loss function (criterion) and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=ts['lr'])

        # optionally resume from a checkpoint
        if ts['resume'] != '':
            if os.path.isfile(root_dir + ts['resume']):
                print(
                    "=> loading checkpoint '{}'".format(root_dir + ts['resume']))
                checkpoint = torch.load(root_dir + ts['resume'],
                                        map_location=self.remap_storage)
                #args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}), Prec@1: {}"
                      .format(ts['resume'], checkpoint['epoch'],
                              checkpoint['best_prec1']))
            else:
                print("=> no checkpoint found at '{}'".format(
                    root_dir + ts['resume']))

        if ts['cuda']:
            model.cuda()
            class_weights = class_weights.cuda()
            criterion.cuda()

        for epoch in range(1, ts['epochs'] + 1):
            train_epoch(train_loader, model, criterion, optimizer, epoch,
                        self.cuda)
            prec1, _ = validation_epoch(val_loader, model, criterion, self.cuda)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    def predict(self, subject: Subject, cuda = True):
        tmp_foldr = 'ZZ_tmp'
        outdir = '../data/processed/' + tmp_foldr + '/'
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        SleepLearning.create_dataset([subject], tmp_foldr)
        val_ds = SleepLearningDataset(tmp_foldr, self.label_remapping, feats)

        val_loader = DataLoader(val_ds, batch_size=200,
                                shuffle=False, **self.kwargs)
        truth = []
        for batch, y in val_loader:
            truth = truth + list(y.numpy())

        model = Net(
            num_classes=len(val_ds.dataset_info['class_distribution'].keys()))
        class_weights = torch.from_numpy(np.ones(val_ds.weights.shape)).float()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        if cuda and torch.cuda.is_available():
            model.cuda()
            class_weights.cuda()
            criterion.cuda()
        else:
            cuda = False

        checkpoint = torch.load(root_dir + '/models/model_best.pth.tar',
                                map_location=self.remap_storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model (epoch {}), Prec@1: {}"
              .format(checkpoint['epoch'],
                      checkpoint['best_prec1']))

        acc, prediction = validation_epoch(val_loader, model, criterion,
                                         cuda)
        return prediction.data.cpu().numpy().astype(int), truth, acc

    @staticmethod
    def create_dataset(subjects: List[Subject], output_foldr: str):
        subject_labels = []
        outdir = '../data/processed/' + output_foldr + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            for subject in subjects:
                subject_labels.append(subject.label)
                psgs_reshaped = {}
                # pad all channels with zeros
                for k, psgs in subject.psgs.items():
                    psgs1 = psgs.reshape(
                        (-1, subject.sampling_rate_ * subject.epoch_length))
                    if subject.sampling_rate_ > 100:
                        # downsample to 100 Hz
                        psgs1 = resample(psgs1, subject.epoch_length * 100,
                                         axis=1)
                    psgs_reshaped[k] = psgs1
                np.savez(outdir + subject.label, subject_label=subject.label,
                         psgs=psgs_reshaped, labels=subject.hypnogram)
        else:
            raise ValueError('ERROR: the given dataset folder already exists!')
