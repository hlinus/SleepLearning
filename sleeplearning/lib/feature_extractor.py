import numpy as np
from scipy import signal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List, Tuple


class FeatureExtractor():
    def __init__(self, features: dict):
        pipelines = []

        #f = np.fft.rfftfreq(window, 1.0 / sampling_rate)
        outdim = None
        f = None
        #outdim = len(np.where(np.logical_and(f >= params['cut_lower'],
        #                                f<=params['cut_upper']))[0])

        for i, (channel, channel_opts) in enumerate(features['channels']):
            transformers = []
            transformers.append(('selector', ChannelSelector(channel)))
            for j, t in enumerate(channel_opts):
                extractor = eval(t)
                if t.startswith("Spectrogram"):
                    between_brackets = t[t.find("(") + 1:t.find(")")].split(',')
                    opts = dict([(k.strip(), int(v)) for (k,v) in [x.split('=') for x in between_brackets]])
                    if f is None:
                        f = np.fft.rfftfreq(opts['window'], 1.0 / opts['fs'])
                elif t.startswith("Cut"):
                    between_brackets = t[t.find("(") + 1:t.find(")")].split(',')
                    opts = dict([(k.strip(), int(v)) for (k, v) in
                                 [x.split('=') for x in between_brackets]])
                    if outdim is None:
                        outdim = len(
                        np.where(np.logical_and(f >= opts['lower'],
                                                    f <= opts['upper']))[0])
                elif t.startswith("Power"):
                    extractor.output_dim = outdim
                transformers.append(('s'+str(j), extractor))
            pipelines.append(('p'+str(i), Pipeline(transformers)))
        self.features = FeatureUnion(pipelines, n_jobs=2)

    def get_features(self) -> FeatureUnion:
        return self.features


class ChannelSelector(BaseEstimator, TransformerMixin):
    """
        Selects channel for the following transforms
    """

    def __init__(self, channel: str):
        self.channel = channel

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.channel]


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


class OneDScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        norm = (x - np.mean(x, axis=1, keepdims=True)) \
               / (np.std(x, axis=1, keepdims=True) + 1e-4)
        norm = np.expand_dims(norm, axis=1)
        return norm


class Spectrogram(BaseEstimator, TransformerMixin):
    """
    Computes the spectrogram of specific channel
    """

    def __init__(self, fs: int, window: int,
                 stride: int):
        self.sampling_rate = fs
        self.window = window
        self.stride = stride

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        psg = x
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

    def __init__(self, output_dim: int = 1):
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

    def __init__(self, fs: int, window: int, lower: float,
                 upper: float):
        self.window = window
        self.sampling_rate = fs
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