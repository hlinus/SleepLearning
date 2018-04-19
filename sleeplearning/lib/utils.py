from typing import List

import torch
from sklearn.pipeline import FeatureUnion
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sleeplearning.lib.base import SleepLearning
import numpy as np
import os


class TwoDScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler for a single sample
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = (x - np.mean(x)) / (np.std(x)+1e-5)
        return x


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
        psg = np.pad(psg, pad_width=padding, mode='edge')
        f, t, sxx = signal.spectrogram(psg, fs=self.sampling_rate,
                                       nperseg=self.window,
                                       noverlap=self.window - self.stride,
                                       scaling='density', mode='psd')
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
        pxx = np.asarray([np.mean(x, 0) for i in range(self.output_dim)])
        return pxx

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
        cut = x[np.logical_and(self.f >= self.lower, self.f <= self.upper), :]
        #print('output dim: ', cut.shape)
        return cut


def create_dataset(subjects: List[SleepLearning], neighbors: int, feature_union: FeatureUnion,
                   output_foldr: str):
    assert(neighbors % 2 == 0)
    outdir = '../data/processed/' + output_foldr + '/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        filenames = []
        labels = []
        for subject in subjects:
            samples_per_epoch = subject.epoch_length * subject.sampling_rate_
            num_epochs = len(subject.hypnogram)
            padded_channels = {}

            # pad all channels with zeros
            for k, v in subject.psgs.items():
                padded_channels[k] = np.pad(v, (
                        neighbors // 2) * samples_per_epoch,
                                            mode='constant',
                                            constant_values=0)

            for e in range(neighbors // 2, num_epochs + neighbors // 2):
                sample = {}

                from_epoch = e - neighbors // 2
                to_epoch = e + neighbors // 2 + 1

                for k, chan in padded_channels.items():
                    sample[k] = chan[(from_epoch) * samples_per_epoch
                                     :(to_epoch) * samples_per_epoch]

                transformed = feature_union.fit_transform(sample)
                num_features = len(feature_union.get_params()['transformer_list'])
                # assure the correct dimensions
                # channels X len(epoch) X len(frequencies)
                transformed = transformed.reshape(transformed.shape[0], num_features, -1)
                transformed = np.swapaxes(transformed, 0, 1)
                transformed = np.swapaxes(transformed, 1, 2)
                label_int = subject.hypnogram[e - neighbors // 2]
                label = SleepLearning.sleep_stages_labels[label_int]
                filename = subject.label + '_epoch_' + '{0:0>5}'.format(
                    e - neighbors // 2) + '_' + str(neighbors) + 'N_' + label
                filenames.append(filename)
                labels.append(label_int)
                np.savez(outdir + filename, x=transformed, y=label_int)
    else:
        raise ValueError('ERROR: the given dataset folder already exists!')

class SleepLearningDataset(object):
    """Sleep Learning dataset."""

    def __init__(self, foldr, transform = None):
        self.dir = '../data/processed/' + foldr + '/'
        self.X = [filename for filename in os.listdir(self.dir)]
        self.transform = transform

    def __getitem__(self, index):
        sample = np.load(self.dir + self.X[index])

        x = sample['x']
        y_ = int(sample['y'])

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x).double()

        return x, y_

    def __len__(self):
        return len(self.X)