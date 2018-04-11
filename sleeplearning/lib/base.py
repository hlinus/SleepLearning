from scipy import signal

import scipy.io
import numpy as np
from typing import Tuple


class SleepLearning(object):
    """Base class which contains the data related to a single day/night of of a
    single subject. There is a classmethod for every support file format which
    can be used to read in the file and store it as sleeplearning object. To
    support a new file format / dataset, a new class method has to be created.

    Attributes
    ----------

    psgs_ : iterable of dictionaries
        Uniform raw data for various input formats and taken from one subject.
        It is in an iterator over dictionaries, where dictionary key are various
        descriptors of individual  polysomnographic (PSG) records.
        The dictionary contains:
         * TODO

    spectograms_: dictionary

    """
    sleep_stages_labels = {0: 'WAKE', 1: "N1", 2: 'N2', 3: 'N3', 4: 'N4',
                           5: 'REM', 6: 'Artifact'}

    def __init__(self, id, psgs: dict, hypnogram, artefact_data, sampling_rate,
                 epoch_size):
        self.id_ = id
        self.psgs_ = psgs
        self.spectograms_ = {}
        self.hypnogram = hypnogram
        self.sampling_rate_ = sampling_rate
        self.epoch_size = epoch_size
        self.window = self.sampling_rate_ * 2
        self.stride = self.sampling_rate_
        self.artefact_data = artefact_data

    ## Input parsing functions for different datasets
    @classmethod
    def read_caro(cls, filepath: str):
        psg = {}
        psg_dict = {'EEG': 'EEG_data_filt', 'EOGR': 'EOGR_data_filt',
                    'EOGL': 'EOGL_data_filt', 'EMG': 'EMG_data_filt'}

        mat = scipy.io.loadmat(filepath)
        artefact_data = {'artefacts': mat['artfact_per4s'][0], 'epoch_size': 4}
        sampling_rate = int(mat['sampling_rate'][0][0])
        epoch_size = int(mat['epoch_size_scoring_sec'][0][0])

        num_labels = None

        if 'sleepStage_score' in mat:
            hypnogram = mat['sleepStage_score'][0]
            num_labels = len(mat['sleepStage_score'][0])
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

        subject_label = filepath.split('/')[-1][5:-12]
        return cls(subject_label, psg, hypnogram, artefact_data, sampling_rate,
                   epoch_size)

    @classmethod
    def _read_physionet_npz(cls, id: str, filepath: str):
        psg = {}
        with np.load(filepath) as f:
            psg['EEG'] = f["x"].reshape(-1)
            labels = f["y"]
            labels[labels == 5] = 6  # UNK -> ARTEFACT
            labels[labels == 4] = 5  # REM is now 5
            sampling_rate = int(f["fs"])
        return cls(id, psg, labels, None, sampling_rate, epoch_size=30)

    def get_spectrograms(self, channel: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the spectogram for a specific channel and for every epoch and
        return a tuple of (frequencies,times,[spectrograms]) where spectrograms
        is a numpy array containing a spectrogram for every epoch
        :param channel: channel key as stored in self.psgs_
        :return: frequencies [fs/2+1], times [epoch size*fs/stride],
        spectrogram (magnitudes) [total epochs, fs/2+1, epoch size*fs/stride ]
        """
        if channel in self.spectograms_:
            return self.spectograms_[channel]
        f = t = 0
        Sxxs = []
        artefacts = np.repeat(self.artefact_data['artefacts'],
                              self.artefact_data[
                                  'epoch_size'] * self.sampling_rate_)
        # reshape to [num epochs, samples per epoch]
        psgs = self.psgs_[channel].reshape(
            (-1, self.sampling_rate_ * self.epoch_size))
        artefacts = artefacts.reshape(
            (-1, self.sampling_rate_ * self.epoch_size))
        padding = self.window // 2 - self.stride // 2
        psgs = np.pad(psgs, pad_width=((0, 0), (padding, padding)), mode='edge')
        artefacts = np.pad(artefacts, pad_width=((0, 0), (padding, padding)),
                           mode='edge')
        for psg, artefact in zip(psgs, artefacts):
            psg_clean = psg  # psg[artefact == 0]
            f, t, Sxx = signal.spectrogram(psg_clean, fs=self.sampling_rate_,
                                           nperseg=self.window,
                                           noverlap=self.window - self.stride,
                                           scaling='density', mode='magnitude')
            Sxxs.append(Sxx)
        self.spectograms_[channel] = (f, t, np.array(Sxxs))
        return self.spectograms_[channel]

    def get_psds(self, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral densities for a specific channel and for
        every epoch.
        :param channel: channel key as stored in self.psgs_
        :return: frequencies [fs/2+1], psds [numEpochs, fs/2+1]
        """
        pxxs = []
        artefacts = np.repeat(self.artefact_data['artefacts'],
                              self.artefact_data[
                                  'epoch_size'] * self.sampling_rate_)
        # reshape to [num epochs, samples per epoch]
        psgs = self.psgs_[channel].reshape(
            (-1, self.sampling_rate_ * self.epoch_size))
        artefacts = artefacts.reshape(
            (-1, self.sampling_rate_ * self.epoch_size))

        padding = self.window // 2 - self.stride // 2
        psgs = np.pad(psgs, pad_width=((0, 0), (padding, padding)), mode='edge')
        artefacts = np.pad(artefacts, pad_width=((0, 0), (padding, padding)),
                           mode='edge')
        f = 0
        for psg, artefact in zip(psgs, artefacts):
            psg_clean = psg[artefact == 0]
            f, pxx = signal.welch(psg_clean, fs=self.sampling_rate_,
                                  nperseg=self.window,
                                  noverlap=self.window - self.stride,
                                  scaling='density')
            pxxs.append(pxx)
        return f, np.array(pxxs)
