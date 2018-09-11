import numpy as np
from wfdb import rdrecord
from wfdb import rdann
import os
from sleeplearning.lib.loaders.baseloader import BaseLoader

EPOCH_TIME = 30


class Physionet18(BaseLoader):
    """
    Loader for PhysioNet/CinC Challenge 2018
    (https://physionet.org/physiobank/database/challenge/2018/)
    """
    def __init__(self, path: str, epoch_length: int = EPOCH_TIME,
                 verbose: bool = False):
        super().__init__(path, epoch_length)
        filename_base = os.path.basename(os.path.dirname(path + '/'))
        if filename_base == 'tr07-0023':
            raise ValueError(f"subject {filename_base} is corrupt and can "
                             f"not be loaded!")
        data = rdrecord(os.path.join(path, filename_base), physical=False)
        annot = rdann(os.path.join(path, filename_base),
                                 extension='arousal')
        self.sampling_rate_ = annot.fs
        self.label = filename_base
        annot_times = annot.sample
        annot = annot.aux_note
        # filter sleep stages (discard arousal annotations)
        sleep_stages = [x for x in list(zip(annot, annot_times))
                        if x[0] in ['W', 'N1', 'N2', 'N3', 'R']]
        annot, annot_times = [np.array(t) for t in zip(*sleep_stages)]
        # Make sure all sleep phase annotations are a multiple of EPOCH_TIME
        assert(np.all(annot_times % EPOCH_TIME == 0))
        sleep_stage_durations = (annot_times[1:]-annot_times[:-1]) \
                                // self.sampling_rate_ // EPOCH_TIME
        num_labels = (annot_times[-1]-annot_times[0]) // self.sampling_rate_ \
                     // EPOCH_TIME
        # remove last sleep phase since we don't know when it ends
        # TODO: check if we can assume it ends at the end of the recording
        annot = annot[:-1]
        # convert to integer labels
        int_label_mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 5}
        annot = [int_label_mapping[x] for x in annot]

        self.hypnogram = np.repeat(annot, sleep_stage_durations)
        assert(len(self.hypnogram) == num_labels)
        self.psgs = {}
        for channel, signal in zip(data.sig_name, data.d_signal.T):
            # cut unlabeled signal parts (start and end)
            signal_cut = signal[annot_times[0]:annot_times[-1]]
            assert(len(signal_cut)// self.sampling_rate_ // EPOCH_TIME
                   == num_labels)
            self.psgs[channel] = signal_cut.astype(np.int16)
