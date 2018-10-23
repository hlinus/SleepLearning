import numpy as np
import os
from scipy import signal
from scipy.signal import firwin, lfilter, resample, resample_poly
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class FeatureExtractor(object):
    def __init__(self, features: dict):
        pipelines = []
        outdim = None
        f = None
        if features is None:
            raise ValueError('No channels/features specified. Please set ds '
                             'dict!')
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


class Z3(BaseEstimator, TransformerMixin):
    """
        Selects channel for the following transforms
    """
    def __init__(self, fs=200):
        self.fs = fs

    def fit(self, x, y=None):
        return self

    def transform(self, x):

        SRATE = 100  # Hz
        LOWPASS = 45.0  # Hz
        HIGHPASS = 0.3  # Hz
        LOWPASSEOG = 12.0  # Hz
        Fs = self.fs / 2.0
        one = np.array(1)
        bEEG = firwin(51, [HIGHPASS / Fs, LOWPASS / Fs], pass_zero=False,
                      window='hamming', scale=True)
        eeg = lfilter(bEEG, one, x.reshape(-1))

        if self.fs != 100:
            P = 100
            Q = self.fs
            eeg = resample_poly(eeg, P, Q)

        totalEpochs = int(len(eeg) / 30.0 / SRATE)
        data_length = 32 * 32 * 1 * totalEpochs
        data = np.empty([data_length], dtype=np.float32)
        window = np.hamming(128)
        epochSize = 32 * 32 * 1

        # STFT based spectrogram computation
        for i in range(totalEpochs):
            for j in range(0, 3000 - 128 - 1, 90):
                tIDX = int(j / 90)
                frame1 = abs(
                    np.fft.fft(eeg[i * 3000 + j: i * 3000 + j + 128] * window))

                data[i * epochSize + tIDX * 32: i * epochSize + tIDX * 32 + 32] = frame1[
                                                                             0:32]
        data = data.reshape(totalEpochs, 32, 32)
        data = data[:, np.newaxis, :, :]
        return data


class SleepStage(BaseEstimator, TransformerMixin):
    """
        Selects channel for the following transforms
    """
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, eeg1):

        eeg1 = eeg1.reshape(-1)
        sample_rate = 100
        window = sample_rate * 2  # window size
        #stride = 16  # signal compression ratio 16 is default
        stride = 100

        # other info
        iters = len(eeg1) / stride
        samples_per_epoch = int(30 * sample_rate)
        num_epochs = int(len(eeg1) / samples_per_epoch)
        epoch_size = int(samples_per_epoch / stride)

        # add padding
        padding = window / 2 - stride / 2
        eeg1 = np.pad(eeg1, pad_width=int(padding), mode='edge')

        # obtain magnitude spectrograms
        f, t, eeg1_spectrogram = signal.spectrogram(eeg1, fs=sample_rate,
                                                    nperseg=window,
                                                    noverlap=window - stride,
                                                    scaling='density', mode='magnitude')


        # obtain 'logged' PSD spectrograms
        eeg1_spectrogram = (eeg1_spectrogram / 2) ** 2 / window


        # move to log scale
        if True:
            "## Moving to log scale"
            eeg1_spectrogram = np.log(eeg1_spectrogram + 0.00000001)

        # [optionally] compress spectrogram to reduce dimensionality
        nu = np.fft.rfftfreq(window, 1.0 / sample_rate)
        cutoff = 30  # 12
        energy_limit = 30
        eeg1_spectrogram = eeg1_spectrogram[np.logical_and(nu >= 0.5, nu <= cutoff), :]


        # Perform per-dimension normalization
        "## Performing per-dimension normalization"
        for i in range(np.shape(eeg1_spectrogram)[0]):
            eeg1_spectrogram[i] = (eeg1_spectrogram[i] - np.mean(
                eeg1_spectrogram[i])) / np.std(eeg1_spectrogram[i])

        # construct multidimensional input to return
        # data: num(epochs) X num(channels) X size(spectrum) x len(epoch)
        spectrum_size = np.shape(eeg1_spectrogram)[0]
        data = np.ndarray(shape=(num_epochs, 1, spectrum_size, epoch_size))
        for i in range(num_epochs):
            data[i, 0, :, :] = eeg1_spectrogram[:, i * epoch_size:(i + 1) * epoch_size]
        return data


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


class QuantileNormalization(BaseEstimator, TransformerMixin):
    """
        Quantile normalization with linear interpolation based on precomputed
        quantiles averaged over the whole dataset
    """

    def __init__(self, channel: str):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        ref = np.load(os.path.join('..', 'data', 'quantiles_rs40_5M.npy'))
        ori = x.reshape(-1)
        s1 = float(ref.shape[0])  # size in
        s2 = float(ori.shape[0])  # size out
        ori_new = ori.copy()
        tmp = np.interp(np.arange(s2) / (s2 - 1) * (s1 - 1),
                        np.arange(s1), ref[:])
        ori_new[np.argsort(ori[:])] = tmp
        return ori_new.reshape(x.shape[0], 1, -1)

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
               / (np.std(x, axis=(2, 3), keepdims=True))
        return norm


class TwoDFreqEpochScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler per frequency, channel, subject and
        epoch
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
        Zero mean and unit variance scaler per frequency, channel and subject
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
        Zero mean and unit variance scaler per time step, channel and subject
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


class TwoDFrequencyTimeSubjScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler per time step, frequency, channel
        and subject
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        std = np.std(x, axis=(0, 2, 3), keepdims=True)
        norm = (x - mean) / (std + 1e-10)
        return norm


class TwoDTimeEpochScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler per time, channel, subject and
        epoch
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


class ResamplePoly(BaseEstimator, TransformerMixin):
    """
        Resample subject epoch-wise using polyphase filtering
    """

    def __init__(self, epoch_len: int = 30, fs: int = 100):
        self.epoch_len = epoch_len
        self.fs = fs

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = resample_poly(x, 100, self.fs, axis=2)
        return x


class ResamplePoly2(BaseEstimator, TransformerMixin):
    """
        Resample subject using polyphase filtering
    """

    def __init__(self, epoch_len: int = 30, fs: int = 200, target_fs: int = 100):
        self.epoch_len = epoch_len
        self.fs = fs
        self.target_fs = target_fs

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        xflat = x.reshape(-1)
        x_down = resample_poly(xflat, self.target_fs, self.fs)
        x_down = x_down.reshape((-1, x.shape[1], self.target_fs*self.epoch_len))
        return x_down


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


class ZeroOneScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        norm = (x - np.min(x, axis=(2, 3), keepdims=True)) \
               / (np.max(x, axis=(2, 3), keepdims=True) - np.min(x, axis=(2,
                                                                          3), keepdims=True))
        return norm


class ZeroOneSubjectScaler(BaseEstimator, TransformerMixin):
    """
        Zero mean and unit variance scaler
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        norm = (x - np.min(x, axis=(0, 1, 2, 3), keepdims=True)) \
               / (np.max(x, axis=(0, 1, 2, 3), keepdims=True) - np.min(x,
                                                                      axis=(
                                                                          0,
                                                                          1, 2,
                                                                          3), keepdims=True))
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
        # [num_epochs, 1, nfreqbins, time_domain]
        sxx = sxx[:, np.newaxis, :, :]
        return sxx


class SpectrogramMultiTaper(BaseEstimator, TransformerMixin):
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
        from lspopt import spectrogram_lspopt
        f, t, sxx = spectrogram_lspopt(psg, self.sampling_rate,
                                       c_parameter=5.0, nperseg=self.window,
                                       noverlap=self.window - self.stride,)
        # [num_epochs, 1, nfreqbins, time_domain]
        sxx = sxx[:, np.newaxis, :, :]
        return sxx


class SpectrogramM(BaseEstimator, TransformerMixin):
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
                                       scaling='density', mode='magnitude')
        # [num_epochs, 1, nfreqbins, time_domain]
        sxx = sxx[:, np.newaxis, :, :]
        return sxx

class Spectrogram2(BaseEstimator, TransformerMixin):
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
        psg_flat = psg.reshape(-1)
        psg_flat = np.pad(psg_flat, pad_width=(padding, padding), mode='edge')
        psg = np.pad(psg, pad_width=((0, 0), (padding, padding)), mode='edge')
        f, t, sxx = signal.spectrogram(psg_flat, fs=self.sampling_rate,
                                       nperseg=self.window,
                                       noverlap=self.window - self.stride,
                                       scaling='density', mode='psd')
        sxx = np.swapaxes(sxx.reshape(sxx.shape[0], x.shape[0], -1), 0, 1)
        sxx = sxx[:, np.newaxis, :, :]
        return sxx


class PowerSpectralDensityMean(BaseEstimator, TransformerMixin):
    """
    Computes the mean power spectral density from a spectrogram and repeats the
     values to 'output_dim'
    """

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        psd = np.mean(x, axis=2, keepdims=True)
        rep = np.repeat(psd, 76, axis=2)
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
        rep = np.repeat(psd, 76, axis=2)
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


class LogTransform2(BaseEstimator, TransformerMixin):
    """
    Computes the log transform of the given features
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.log(x + 1e-7) + 1