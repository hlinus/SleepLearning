import numpy as np
from scipy import signal
from scipy.signal import firwin, lfilter, resample
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class FeatureExtractor(object):
    def __init__(self, features: dict):
        pipelines = []
        outdim = None
        f = None

        for i, (channel, channel_opts) in enumerate(features):
            transformers = [('selector', ChannelSelector(channel))]
            for j, t in enumerate(channel_opts):
                extractor = eval(t)
                if t.startswith("Spectrogram"):
                    between_brackets = t[t.find("(") + 1:t.find(")")].split(',')
                    opts = dict([(k.strip(), int(v)) for (k,v)
                                 in [x.split('=') for x in between_brackets]])
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
        x = x[self.channel]
        x = x[:, np.newaxis, :]
        return x


class ConvToInt16(BaseEstimator, TransformerMixin):
    """
        Convert input to 16 bit integer values
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.astype(np.int16)


class BandPass(BaseEstimator, TransformerMixin):
    """
        Band filter 1D signal
    """
    def __init__(self, fs, lowpass, highpass):
        self.lowpass = lowpass
        self.highpass = highpass
        self.fs = fs

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        Fs = self.fs / 2.0
        # one = np.array(1)
        fir = firwin(51, [self.highpass / Fs, self.lowpass / Fs], pass_zero=False,
                      window='hamming', scale=True)
        x = lfilter(fir, 2, x)
        return x


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


class TwoDFreqScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        mean = np.mean(x, axis=2, keepdims=True)
        std = np.std(x, axis=2, keepdims=True)
        norm = (x - mean) / (std + 1e-10)
        return norm


class TwoDFreqSubjScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        mean = np.mean(x, axis=(0, 2), keepdims=True)
        std = np.std(x, axis=(0, 2), keepdims=True)
        norm = (x - mean) / (std + 1e-10)
        return norm


class TwoDTimeSubjScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        mean = np.mean(x, axis=(0, 3), keepdims=True)
        std = np.std(x, axis=(0, 3), keepdims=True)
        norm = (x - mean) / (std + 1e-10)
        return norm


class TwoDTimeScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        mean = np.mean(x, axis=3, keepdims=True)
        std = np.std(x, axis=3, keepdims=True)
        norm = (x - mean) / (std + 1e-10)
        return norm


class Resample(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self, epoch_len: int = 30, fs: int = 100):
        self.epoch_len = epoch_len
        self.fs = fs

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = resample(x, self.epoch_len * self.fs, axis=2)
        return x


class OneDScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        norm = (x - np.mean(x, axis=2, keepdims=True)) \
               / (np.std(x, axis=2, keepdims=True) + 1e-4)
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
        psg = x[:,0,:]
        #psg = x
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