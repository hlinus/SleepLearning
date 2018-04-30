import json
import os
import shutil
import sys
from argparse import Namespace
from typing import List
import numpy as np
import torch
from sklearn.pipeline import FeatureUnion
from torch.utils.data import DataLoader

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

        four_label = {'N1': 'NREM', 'N2': 'NREM', 'N3': 'NREM', 'N4': 'NREM',
                      'WAKE': 'WAKE', 'REM': 'REM', 'Artifact': 'Artifact'}

        self.label_remapping = {'WAKE': 'WAKE', 'N1': 'N1', 'N2': 'N2',
                                'N3': 'N3',
                                'N4': 'WAKE', 'REM': 'REM', 'Artifact': 'WAKE'}

    def train(self, args: Namespace):
        best_prec1 = 0
        train_ds = SleepLearningDataset('caro-log/train', self.label_remapping)
        val_ds = SleepLearningDataset('caro-log/test', self.label_remapping)
        print("TRAIN: ", train_ds.dataset_info)
        print("VAL: ", val_ds.dataset_info)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, **self.kwargs)

        val_loader = DataLoader(val_ds, batch_size=args.test_batch_size,
                                shuffle=False, **self.kwargs)

        class_weights = torch.from_numpy(train_ds.weights).float() \
            if not args.no_weight_loss \
            else torch.from_numpy(np.ones(train_ds.weights.shape)).float()
        print("train class weights: ", class_weights)
        model = Net(
            num_classes=len(train_ds.dataset_info['class_distribution'].keys()))

        # initialize layers with xavier
        model.weights_init()

        # define loss function (criterion) and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(root_dir + args.resume):
                print(
                    "=> loading checkpoint '{}'".format(root_dir + args.resume))
                checkpoint = torch.load(root_dir + args.resume,
                                        map_location=self.remap_storage)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}), Prec@1: {}"
                      .format(args.resume, checkpoint['epoch'],
                              checkpoint['best_prec1']))
            else:
                print("=> no checkpoint found at '{}'".format(
                    root_dir + args.resume))

        if self.cuda:
            model.cuda()
            class_weights = class_weights.cuda()
            criterion.cuda()

        for epoch in range(1, args.epochs + 1):
            train_epoch(train_loader, model, criterion, optimizer, epoch,
                        self.cuda, args.log_interval)
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

    def predict(self, subject: Subject):
        tmp_foldr = 'ZZ_tmp'
        outdir = '../data/processed/' + tmp_foldr + '/'
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        SleepLearning.create_dataset([subject], 4, feats, True, tmp_foldr)
        val_ds = SleepLearningDataset(tmp_foldr, self.label_remapping)

        val_loader = DataLoader(val_ds, batch_size=200,
                                shuffle=False, **self.kwargs)
        model = Net(
            num_classes=len(val_ds.dataset_info['class_distribution'].keys()))
        class_weights = torch.from_numpy(np.ones(val_ds.weights.shape)).float()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        if self.cuda:
            model.cuda()
            class_weights.cuda()
            criterion.cuda()

        checkpoint = torch.load(root_dir + '/models/model_best.pth.tar',
                                map_location=self.remap_storage)
        model.load_state_dict(checkpoint['state_dict'])

        _, prediction = validation_epoch(val_loader, model, criterion,
                                         self.cuda)
        return prediction.data.cpu().numpy().astype(int)

    @staticmethod
    def create_dataset(subjects: List[Subject], neighbors: int,
                       feature_union: FeatureUnion, discard_arts: bool,
                       output_foldr: str):
        assert (neighbors % 2 == 0)
        class_distribution = np.zeros(
            len(Subject.sleep_stages_labels.keys()))
        subject_labels = []
        feature_matrix = 0  # initialize
        outdir = '../data/processed/' + output_foldr + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            for subject in subjects:
                subject_labels.append(subject.label)
                samples_per_epoch = subject.epoch_length * subject.sampling_rate_
                num_epochs = len(subject.hypnogram)
                padded_channels = {}
                psgs_reshaped = {}
                # pad all channels with zeros
                for k, psgs in subject.psgs.items():
                    psgs1 = psgs.reshape(
                        (-1, subject.sampling_rate_ * subject.epoch_length))
                    psgs_reshaped[k] = psgs1
                    padded_channels[k] = np.pad(psgs, (
                            neighbors // 2) * samples_per_epoch,
                                                mode='constant',
                                                constant_values=0)
                # [num_epochs X num_channels X freq_domain X time_domain]
                feature_matrix = feature_union.fit_transform(psgs_reshaped)
                # pad with zeros before and after (additional '#neighbors' epochs)
                feature_matrix = np.pad(feature_matrix, (
                    (neighbors // 2, neighbors // 2), (0, 0), (0, 0), (0, 0)),
                                        mode='constant')
                # create samples with neighbors
                feature_matrix = np.array([np.concatenate(
                    feature_matrix[i - neighbors // 2:i + neighbors // 2 + 1],
                    axis=2) for i
                    in range(neighbors // 2, num_epochs + neighbors // 2)])

                for e, (sample, label_int) in enumerate(
                        zip(feature_matrix, subject.hypnogram)):
                    label = Subject.sleep_stages_labels[label_int]
                    if discard_arts and label == 'Artifact':
                        continue
                    class_distribution[label_int] += 1
                    id = subject.label + '_epoch_' + '{0:0>5}'.format(
                        e) + '_' + str(neighbors) + 'N_' + label
                    sample = {'id': id, 'x': sample, 'y': label_int}
                    np.savez(outdir + sample['id'], x=sample['x'],
                             y=sample['y'])

            dataset_info = {}
            class_distribution_dict = {}
            for i in range(len(class_distribution)):
                class_distribution_dict[Subject.sleep_stages_labels[i]] = int(
                    class_distribution[i])
            dataset_info['subjects'] = subject_labels
            dataset_info['class_distribution'] = class_distribution_dict
            dataset_info['input_shape'] = feature_matrix[0].shape
            j = json.dumps(dataset_info, indent=4)
            f = open(outdir + 'dataset_info.json', 'w')
            print(j, file=f)
            f.close()
        else:
            raise ValueError('ERROR: the given dataset folder already exists!')
