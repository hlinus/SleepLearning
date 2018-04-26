from abc import ABC
from typing import Tuple

import numpy as np
from scipy import signal


class Subject(ABC):
    """Base class which contains the data related to a single day/night of of a
    single subject. There is a classmethod for every support file format which
    can be used to read in the file and store it as sleeplearning object. To
    support a new file format / dataset, a new class method has to be created.

    Attributes
    ----------

    psgs : iterable of dictionaries
        Uniform raw data for various input formats and taken from one subject.
        It is in an iterator over dictionaries, where dictionary key are various
        descriptors of individual  polysomnographic (PSG) records.
        The dictionary contains:
         * TODO

    spectograms_: dictionary

    """
    sleep_stages_labels = {0: 'WAKE', 1: "N1", 2: 'N2', 3: 'N3', 4: 'N4',
                           5: 'REM', 6: 'Artifact'}

    def __init__(self, path: str, epoch_length: int, verbose = False):
        self.path = path
        self.label = None
        self.psgs = None
        self.spectograms_ = {}
        self.hypnogram = None
        self.sampling_rate_ = None  #sampling_rate
        self.epoch_length = epoch_length  #epoch_size
        self.window = None  #self.sampling_rate_ * 2
        self.stride =  None  #self.sampling_rate_


    def get_spectrograms(self, channel: str, window: int, stride: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the spectogram for a specific channel and for every epoch and
        return a tuple of (frequencies,times,[spectrograms]) where spectrograms
        is a numpy array containing a spectrogram for every epoch
        :param channel: channel key as stored in self.psgs
        :param window: window size of FFT
        :param stride: for overlapping windows
        :return: frequencies [fs/2+1], times [epoch size*fs/stride],
        spectrogram (magnitudes) [total epochs, fs/2+1, epoch size*fs/stride ]
        """
        if channel in self.spectograms_:
            return self.spectograms_[channel]
        f = t = 0
        Sxxs = []

        # reshape to [num epochs, samples per epoch]
        psgs = self.psgs[channel].reshape(
            (-1, self.sampling_rate_ * self.epoch_length))

        padding = window // 2 - stride // 2
        psgs = np.pad(psgs, pad_width=((0, 0), (padding, padding)), mode='edge')

        for psg in psgs:
            psg_clean = psg
            f, t, Sxx = signal.spectrogram(psg_clean, fs=self.sampling_rate_,
                                           nperseg=window,
                                           noverlap=window - stride,
                                           scaling='density', mode='magnitude')
            Sxxs.append(Sxx)
        self.spectograms_[channel] = (f, t, np.array(Sxxs))
        return self.spectograms_[channel]

    def get_psds(self, channel: str, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral densities for a specific channel and for
        every epoch.
        :param channel: channel key as stored in self.psgs
        :return: frequencies [fs/2+1], psds [numEpochs, fs/2+1]
        """
        pxxs = []

        # reshape to [num epochs, samples per epoch]
        psgs = self.psgs[channel].reshape(
            (-1, self.sampling_rate_ * self.epoch_length))

        padding = window // 2 - stride // 2
        psgs = np.pad(psgs, pad_width=((0, 0), (padding, padding)), mode='edge')
        f = 0
        for psg in psgs:
            f, pxx = signal.welch(psg, fs=self.sampling_rate_,
                                  nperseg=window,
                                  noverlap=window - stride,
                                  scaling='density')
            pxxs.append(pxx)
        return f, np.array(pxxs)