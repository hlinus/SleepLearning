import inspect
import os
import re
import sys
import numpy as np
import torch
from scipy.signal import resample
from torch import optim
from torch.utils.data import DataLoader
from typing import List

from sleeplearning.lib.feature_extractor import FeatureExtractor

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.loaders.subject import Subject


def load_data(dir: os.path, num_labels: int, feats: dict, neighbors: int,
              batch_size, cuda, verbose = False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    ds = SleepLearningDataset(dir, num_labels,
                                    FeatureExtractor(feats).get_features(),
                                    neighbors)
    if verbose: print(ds.dataset_info)
    loader = DataLoader(ds, batch_size=batch_size,
                              shuffle=True, **kwargs)
    return loader, ds.dataset_info['input_shape']


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


class SleepLearningDataset(object):
    """Sleep Learning dataset."""

    def __init__(self, foldr: str, num_labels: int, feature_extractor,
                 neighbors, discard_arts=True, transform=None):
        assert(neighbors % 2 == 0)
        if num_labels == 5:
            class_remapping = {'WAKE': 'WAKE', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3',
                  'N4': 'WAKE', 'REM': 'REM', 'Artifact': 'WAKE'}
        else:
            class_remapping = {'WAKE': 'WAKE', 'N1': 'NREM', 'N2': 'NREM',
                               'N3': 'NREM',
                               'N4': 'NREM', 'REM': 'REM', 'Artifact': 'WAKE'}

        dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.dir = os.path.join(dir, 'data', 'processed', foldr)
        self.X = []
        self.class_remapping = class_remapping
        self.labels = np.array([], dtype=int)
        self.transform = transform
        subject_files = [filename for filename in
                         os.listdir(self.dir) if
                         filename.endswith(".npz")]
        class_distribution = np.zeros(
            len(Subject.sleep_stages_labels.keys()))
        subject_labels = []
        for subject_file in subject_files:
            subject = np.load(os.path.join(self.dir, subject_file))
            subject_labels.append(subject['subject_label'].item())
            psgs_reshaped = subject['psgs'].item()

            # [num_epochs X num_channels X freq_domain X time_domain]
            feature_matrix = feature_extractor.fit_transform(psgs_reshaped)
            del psgs_reshaped
            num_epochs = feature_matrix.shape[0]

            # TODO: neighbors for 1D features
            if feature_matrix.ndim == 3:
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
                    zip(feature_matrix, subject['labels'])):
                label = Subject.sleep_stages_labels[label_int]
                if discard_arts and label == 'Artifact':
                    continue
                class_distribution[label_int] += 1
                id = subject[
                         'subject_label'].item() + '_epoch_' + '{0:0>5}'.format(
                    e) + '_' + str(neighbors) + 'N_' + label
                sample = {'id': id, 'x': sample, 'y': label_int}
                self.labels = np.append(self.labels, label_int)
                self.X.append(sample)

        self.dataset_info = {}
        class_distribution_dict = {}
        for i in range(len(class_distribution)):
            class_distribution_dict[Subject.sleep_stages_labels[i]] = int(
                class_distribution[i])
        self.dataset_info['subjects'] = subject_labels
        self.dataset_info['class_distribution'] = class_distribution_dict
        self.dataset_info['input_shape'] = feature_matrix[0].shape

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
        sample = self.X[index]
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