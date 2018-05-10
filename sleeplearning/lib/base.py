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
from sleeplearning.lib.feature_extractor import *
from sklearn.externals import joblib

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

    def train(self,
              batch_size: int,
              test_batch_size: int,
              lr: int,
              resume: str,
              cuda: bool,
              weight_loss: bool,
              epochs: int,
              train_ds: SleepLearningDataset,
              val_ds: SleepLearningDataset,
              out_dir: str):
        best_prec1 = 0
        start_epoch = 1
        print("TRAIN: ", train_ds.dataset_info)
        print("TEST: ", val_ds.dataset_info)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, **self.kwargs)

        val_loader = DataLoader(val_ds, batch_size=test_batch_size,
                                shuffle=False, **self.kwargs)

        class_weights = torch.from_numpy(train_ds.weights).float() \
            if weight_loss \
            else torch.from_numpy(np.ones(train_ds.weights.shape)).float()
        print("train class weights: ", class_weights)
        model = Net(
            num_classes=len(train_ds.dataset_info['class_distribution'].keys()))

        # initialize layers with xavier
        model.weights_init()

        # define loss function (criterion) and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # optionally resume from a checkpoint
        if resume != '':
            if os.path.isfile(root_dir + resume):
                print(
                    "=> loading checkpoint '{}'".format(root_dir + resume))
                checkpoint = torch.load(root_dir + resume,
                                        map_location=self.remap_storage)
                #args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}), Prec@1: {}"
                      .format(resume, checkpoint['epoch'],
                              checkpoint['best_prec1']))
            else:
                print("=> no checkpoint found at '{}'".format(
                    root_dir + resume))
                return

        if cuda:
            model.cuda()
            class_weights = class_weights.cuda()
            criterion.cuda()

        for epoch in range(start_epoch, epochs + 1):
            train_epoch(train_loader, model, criterion, optimizer, epoch,
                        self.cuda)
            prec1, _ = validation_epoch(val_loader, model, criterion, self.cuda)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'train_subjects': train_ds.dataset_info['subjects']
            }, is_best, dir=out_dir)

    def predict(self, subject: Subject, cuda=True, num_labels=5):
        tmp_foldr = 'ZZ_tmp'
        outdir = '../data/processed/' + tmp_foldr + '/'
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        SleepLearning.create_dataset([subject], tmp_foldr)
        feats = joblib.load(os.path.join(root_dir, 'models', 'feature_extractor_caro.pkl'))
        val_ds = SleepLearningDataset(tmp_foldr, num_labels, feats)

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
