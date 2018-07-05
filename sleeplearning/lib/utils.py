import sys
import os
import torch
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import inspect
import time
import pandas as pd
import re
import numpy as np
from scipy.signal import resample
from torch import optim
from torch.utils.data import DataLoader
from typing import List
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from sleeplearning.lib.loaders.baseloader import BaseLoader
from sleeplearning.lib.feature_extractor import FeatureExtractor
from sleeplearning.lib.models import *
from sleeplearning.lib.loaders import *


class SleepLearningDataset(object):
    """Sleep Learning dataset."""

    def __init__(self, data_dir: str, subject_csv: str, cvfold: int,
                 num_labels: int, feature_extractor,neighbors, loader,
                 discard_arts=True, transform=None, verbose=False):
        assert(neighbors % 2 == 0)

        if num_labels == 5:
            class_remapping = {0: 0, 1: 1, 2: 2, 3: 3,
                  4: 3, 5: 4}
        elif num_labels == 3:
            class_remapping = {0: 0, 1: 1, 2: 1,
                               3: 1,
                               4: 1, 5: 2}

        config_hash = hash((data_dir, num_labels, feature_extractor, neighbors,
                           discard_arts, transform))
        self.dir = os.path.join(data_dir, 'transformed_'+str(config_hash))
        self.transform = transform
        self.X = []
        self.targets = []
        # processed dataset does not yet exist?
        subject_labels = []
        if not os.path.isdir(self.dir):
            subject_files = pd.read_csv(subject_csv, header=None)[cvfold].dropna().tolist()
            class_distribution = np.zeros(
                len(BaseLoader.sleep_stages_labels.keys()), dtype=int)
            #subject_labels = []
            for subject_file in subject_files:
                start = time.time()
                subject = loader(os.path.join(data_dir, subject_file))
                subject_labels.append(subject.label)
                psgs_reshaped = {}
                for k, psgs in subject.psgs.items():
                    psgs1 = psgs.reshape(
                        (-1, subject.sampling_rate_ * subject.epoch_length))
                    if subject.sampling_rate_ > 100:
                        # downsample to 100 Hz
                        psgs1 = resample(psgs1, subject.epoch_length * 100,
                                         axis=1)
                    psgs_reshaped[k] = psgs1
                # if spectogram used:
                # [num_epochs X num_channels X freq_domain X time_domain]
                # else
                # [num_epochs X num_channels X time_domain]
                feature_matrix = feature_extractor.fit_transform(psgs_reshaped)
                del psgs_reshaped
                num_epochs = feature_matrix.shape[0]

                if neighbors > 0:
                    # pad with zeros before and after (additional '#neighbors' epochs)
                    pad_width = (
                        (neighbors // 2, neighbors // 2), (0, 0), (0, 0))
                    concat_axis = 1
                    if feature_matrix.ndim == 4:
                        pad_width += ((0, 0),)
                        concat_axis = 2
                    feature_matrix = np.pad(feature_matrix, pad_width,
                                            mode='constant')
                    # create samples with neighbors
                    feature_matrix = np.array([np.concatenate(
                        feature_matrix[i - neighbors // 2:i + neighbors // 2 + 1],
                        axis=concat_axis) for i
                        in range(neighbors // 2, num_epochs + neighbors // 2)])

                for e, (sample, label_int) in enumerate(
                        zip(feature_matrix, subject.hypnogram)):
                    label = BaseLoader.sleep_stages_labels[label_int]
                    if discard_arts and label == 'Artifact':
                        continue
                    class_distribution[label_int] += 1
                    id = subject.label + '_epoch_' + '{0:0>5}'.format(
                        e) + '_' + str(neighbors) + 'N_' + label

                    label_int = class_remapping[label_int]
                    self.targets.append(label_int)
                    sample = {'id': id, 'x': sample, 'y': label_int}
                    self.X.append(sample)
                if verbose:
                    print('loaded {} [{:.2f}s]'.format(subject_file,
                                                       time.time()-start))

        self.dataset_info = {}
        class_distribution_dict = {}
        for i in range(len(class_distribution)):
            if class_distribution[i] > 0:
                class_distribution_dict[BaseLoader.sleep_stages_labels[i]] = \
                    int(class_distribution[i])
        self.dataset_info['subjects'] = subject_labels
        self.dataset_info['class_distribution'] = class_distribution_dict
        self.dataset_info['input_shape'] = feature_matrix[0].shape

    def __getitem__(self, index):
        sample = self.X[index]
        x = sample['x']
        y_ = sample['y']

        if self.transform is not None:
            x = self.transform(x)
        #x = torch.from_numpy(x)#.double()

        return x, y_

    def __len__(self):
        return len(self.X)


def get_sampler(ds: SleepLearningDataset,
                batch_size: int, oversample: bool, cuda: bool,
                verbose):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if oversample:
        class_count = np.fromiter(ds.dataset_info['class_distribution'].values(), dtype=np.float32)
        weight = 1. / class_count
        samples_weight = weight[ds.targets].astype(np.float32)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        sampler = RandomSampler(ds)

    if verbose:
        print('\nclass distribution:', ds.dataset_info['class_distribution'])
        print('input shape:', ds.dataset_info['input_shape'])
        print('oversample:', oversample)

    dataloader = DataLoader(ds, batch_size=batch_size, sampler=sampler, **kwargs)
    return dataloader


def get_loader(s):
    ind = [i for i in range(len(s)) if str.isupper(s[i])]
    module_name = ''.join(
        [s[i] + '_' if (i + 1) in ind else str.lower(s[i])
         for i in range(len(s))])
    loader = eval(module_name + '.' + s)
    return loader

def get_network(ts):
    ind = [i for i in range(len(ts['model'])) if str.isupper(ts['model'][i])]
    module_name = ''.join(
        [ts['model'][i] + '_' if (i + 1) in ind else str.lower(ts['model'][i])
         for i in range(len(ts['model']))])
    model = eval(module_name + '.' + ts['model'])(ts)
    return model


def get_model(ts, weighted_loss, train_ds, cuda):
    ts['input_dim'] = train_ds.dataset_info['input_shape']
    model = get_network(ts)

    optim_fn, optim_params = get_optimizer(ts['optim'])

    optimizer = optim_fn(model.parameters(), **optim_params)
    # TODO: refactor to get_loss(train_ds, weighted_loss: bool)
    if weighted_loss:
        # TODO: assure weights are in correct order
        counts = np.fromiter(
            train_ds.dataset_info['class_distribution'].values(),
            dtype=float)
        normed_counts = counts / np.min(counts)
        weights = np.reciprocal(normed_counts).astype(np.float32)
    else:
        weights = np.ones(ts['nclasses'])
    print("\nCLASS WEIGHTS (LOSS): ", weights)
    weights = torch.from_numpy(weights).type(torch.FloatTensor)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    print('\n', model)
    print('\n')

    if cuda:
        model.cuda()
        criterion.cuda()
        weights.cuda()

    return model, criterion, optimizer


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count