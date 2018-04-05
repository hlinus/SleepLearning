from scipy import signal

import scipy.io
import numpy as np
from typing import Tuple


class SleepLearning(object):
    """Base class which contains the data related to a sample of sleep learning

    Attributes
    ----------
    id_ : string identifier of subject

    psgs_ : iterable of dictionaries
        Uniform raw data for various input formats and taken from one subject.
        It is in an iterator over dictionaries, where dictionary key are various
        descriptors of individual  polysomnographic (PSG) records.
        The dictionary contains:
         * TODO

    spectograms_: dictionary


    prediction_ : array, shape = [numEpochs]
        Predicted  sleep stages
    """
    sleep_stages_labels =  {0: 'WAKE', 1: "N1", 2: 'N2', 3: 'N3', 4: 'N4', 5: 'REM', 6: 'Artifact'}

    def __init__(self, id, psgs: dict, hypnogram, sampling_rate, epoch_size):
        self.id_ = id
        self.psgs_ = psgs
        self.spectograms_ = {}
        self.hypnogram = hypnogram
        self.sampling_rate_ = sampling_rate
        self.epoch_size = epoch_size
        self.window = self.sampling_rate_ * 2
        self.stride = 250 # 50
        self.prediction = np.array([])

    def test(self):
        print("hi")
    ## Input parsing functions

    # Example
    # TODO: add example
    @classmethod
    def _read_mat(cls, id: str, filepath: str, psg_dict: dict, hypnogram_key: str = None,
                  epoch_size=20, sampling_rate=250):
        psg = {}
        mat = scipy.io.loadmat(filepath)
        num_labels = None

        if hypnogram_key is not None:
            hypnogram = mat[hypnogram_key][0]
            num_labels = len(mat[hypnogram_key][0])
        else:
            hypnogram = None

        for k, v in psg_dict.items():
            num_samples = mat[v].shape[1]
            if num_labels is None:
                # assuming maximal possible epochs given samples
                num_labels = num_samples // (
                            epoch_size * sampling_rate)
            samples_wo_label = num_samples - (num_labels *
                                              epoch_size * sampling_rate)
            print(k + ": cutting ", samples_wo_label / sampling_rate,
                  "seconds without label at the end")
            eeg_cut = mat[v][0][:-samples_wo_label]  # [np.newaxis, :]
            # eeg_cut = eeg_cut.reshape(num_labels, -1)
            psg[k] = eeg_cut


        return cls(id, psg, hypnogram, sampling_rate, epoch_size)

    def get_spectograms(self, psg_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if psg_key in self.spectograms_:
            return self.spectograms_[psg_key]
        f = t = 0
        Sxxs = []
        # reshape to [num epochs, samples per epoch]
        psgs = self.psgs_[psg_key].reshape((-1, self.sampling_rate_ * self.epoch_size))
        padding = self.window // 2 - self.stride // 2
        psgs = np.pad(psgs, pad_width=((0, 0), (padding, padding)), mode='edge')
        for psg in psgs:
            f, t, Sxx = signal.spectrogram(psg, fs=self.sampling_rate_, nperseg=self.window,
                                           noverlap=self.window - self.stride,
                                           scaling='density', mode='magnitude')
            Sxxs.append(Sxx)
        self.spectograms_[psg_key] = (f, t, np.array(Sxxs))
        return self.spectograms_[psg_key]

    def get_periodograms(self, psg_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param psg_key:
        :return:
        """
        f, t, Sxxs = self.get_spectograms(psg_key)
        psds = []
        for Sxx in Sxxs:
            psd = (Sxx / 2) ** 2 / self.window
            psd = np.sum(psd, axis=1)
            psds.append(psd)
        return f, np.array(psds)
