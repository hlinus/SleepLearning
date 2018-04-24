from typing import List, Tuple

import torch
from sklearn.pipeline import FeatureUnion
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sleeplearning.lib.base import SleepLearning
import numpy as np
import os
import json


class TwoDScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        norm = (x - np.mean(x, axis=(2,3), keepdims=True) ) /  (np.std(x, axis=(2,3), keepdims=True)+1e-4)
        return norm


class Spectrogram(BaseEstimator, TransformerMixin):
    """
    Computes the spectrogram of specific channel
    """
    def __init__(self, channel: str, sampling_rate: int, window: int, stride: int):
        self.channel = channel
        self.sampling_rate = sampling_rate
        self.window = window
        self.stride = stride

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        psg = x[self.channel]
        padding = self.window // 2 - self.stride // 2
        psg = np.pad(psg, pad_width=((0, 0), (padding, padding)), mode='edge')
        f, t, sxx = signal.spectrogram(psg, fs=self.sampling_rate,
                                       nperseg=self.window,
                                       noverlap=self.window - self.stride,
                                       scaling='density', mode='psd')
        sxx = sxx[:, np.newaxis, : ,:]
        return sxx


class PowerSpectralDensityMean(BaseEstimator, TransformerMixin):
    """
    Computes the mean power spectral density from a spectrogram and repeats the
     values to 'output_dim'
    """
    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        psd = np.mean(x, axis=2, keepdims=True)
        rep = np.repeat(psd, self.output_dim, axis=2)
        return rep


class PowerSpectralDensitySum(BaseEstimator, TransformerMixin):
    """
    Computes the sum power spectral density from a spectrogram and repeats the
     values to 'output_dim'
    """
    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        psd = np.sum(x, axis=2, keepdims=True)
        rep = np.repeat(psd, self.output_dim, axis=2)
        return rep


class CutFrequencies(BaseEstimator, TransformerMixin):
    """
    Cuts out the spectrogram between lower and upper frequency
    """
    def __init__(self, window: int, sampling_rate: int, lower: float, upper: float):
        self.f = np.fft.rfftfreq(window, 1.0 / sampling_rate)
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        cut = x[:, :, np.logical_and(self.f >= self.lower, self.f <= self.upper), :]
        return cut


def create_dataset(subjects: List[SleepLearning], neighbors: int, feature_union: FeatureUnion, output_foldr: str):
    assert(neighbors % 2 == 0)
    class_distribution = np.zeros(
        len(SleepLearning.sleep_stages_labels.keys()))
    subject_labels = []
    feature_matrix  = 0 # initialize
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
            feature_matrix = np.pad(feature_matrix, ((neighbors//2, neighbors//2), (0, 0), (0, 0), (0, 0)), mode='constant')
            # create samples with neighbors
            feature_matrix = np.array([np.concatenate(
                feature_matrix[i - neighbors // 2:i + neighbors // 2 + 1], axis=2) for i
                      in range(neighbors // 2, num_epochs + neighbors // 2)])

            for e, (sample, label_int) in enumerate(zip(feature_matrix, subject.hypnogram)):
                class_distribution[label_int] += 1
                label = SleepLearning.sleep_stages_labels[label_int]
                id = subject.label + '_epoch_' + '{0:0>5}'.format(
                    e) + '_' + str(neighbors) + 'N_' + label
                sample = {'id': id, 'x': sample, 'y': label_int}
                np.savez(outdir + sample['id'], x=sample['x'], y=sample['y'])

        dataset_info = {}
        class_distribution_dict = {}
        for i in range(len(class_distribution)):
            class_distribution_dict[SleepLearning.sleep_stages_labels[i]] = int(
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
        y_str = SleepLearning.sleep_stages_labels[int(sample['y'])]
        y_str_remapped = self.class_remapping[y_str]
        y_ = int(self.class_int_mapping[y_str_remapped])

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x).double()

        return x, y_

    def __len__(self):
        return len(self.X)