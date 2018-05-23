import numpy as np
from scipy import signal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List, Tuple


class FeatureExtractor():
    def __init__(self, features: dict):
        pipelines = []
        sampling_rate = features['sampling_rate']
        window = features['window']
        stride = features['stride']

        f = np.fft.rfftfreq(window, 1.0 / sampling_rate)
        outdim = 1

        for i, channel in enumerate(features['channels']):
            transformers = []
            transformers.append(('s1', Spectrogram(channel['channel'],
                                    sampling_rate, window,
                                    stride)))
            transformers.append(('cut',CutFrequencies(window,
                                                sampling_rate,
                                                channel['cut_lower'],
                                                channel['cut_upper'])))

            if channel['psd'] == 'mean':
                transformers.append(('psd', PowerSpectralDensityMean(outdim)))
            elif channel['psd'] == 'sum':
                transformers.append(('psd', PowerSpectralDensitySum(outdim)))
            else:
                # infer output dimension for psd
                # TODO: mamke it work if psd feature is before non-psd feature
                outdim = len(np.where(np.logical_and(f>=channel['cut_lower'],
                                                f<=channel['cut_upper']))[0])
            if channel['log']:
                transformers.append(('log',LogTransform()))
            if channel['scale'] == '2D':
                transformers.append(('2Dscale',TwoDScaler()))
            pipelines.append(('p'+str(i), Pipeline(transformers)))
        self.features = FeatureUnion(pipelines, n_jobs=2)

    def get_features(self) -> FeatureUnion:
        return self.features


class TwoDScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        norm = (x - np.mean(x, axis=(2, 3), keepdims=True)) \
               / (np.std(x, axis=(2, 3), keepdims=True) + 1e-4)
        return norm


class Spectrogram(BaseEstimator, TransformerMixin):
    """
    Computes the spectrogram of specific channel
    """

    def __init__(self, channel: str, sampling_rate: int, window: int,
                 stride: int):
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
        sxx = sxx[:, np.newaxis, :, :]
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

    def __init__(self, window: int, sampling_rate: int, lower: float,
                 upper: float):
        self.window = window
        self.sampling_rate = sampling_rate
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        f = np.fft.rfftfreq(self.window, 1.0 / self.sampling_rate)
        cut = x[:, :,
              np.logical_and(f >= self.lower, f <= self.upper), :]
        return cut


class LogTransform(BaseEstimator, TransformerMixin):
    """
    Computes the log transform of the given features
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.log(x + 1e-4)