import numpy as np
# fix bug: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# details: https://github.com/conda-forge/pygridgen-feedstock/issues/10
import matplotlib
matplotlib.use("Agg")
from wfdb import rdrecord
from wfdb import rdann
import os
from sleeplearning.lib.loaders.baseloader import BaseLoader

EPOCH_TIME = 30


class Physionet18(BaseLoader):
    """
    Loader for PhysioNet/CinC Challenge 2018 (https://physionet.org/physiobank/database/challenge/2018/)
    """
    def __init__(self, path: str, epoch_length: int = EPOCH_TIME, verbose: bool = False):
        super().__init__(path, epoch_length)
        filename_base = os.path.basename(os.path.dirname(path + '/'))
        data = rdrecord(os.path.join(path, filename_base), physical=False)
        annotations = rdann(os.path.join(path, filename_base),
                                 extension='arousal')
        self.sampling_rate_ = annotations.fs
        self.label = filename_base
        annotation_times = annotations.sample
        annotations = annotations.aux_note
        # filter sleep stages (discard arousal annotations)
        sleep_stages = [x for x in list(zip(annotations, annotation_times)) if x[0] in ['W', 'N1', 'N2', 'N3', 'R']]
        annotations, annotation_times = [np.array(t) for t in zip(*sleep_stages)]
        sleep_stage_durations = (annotation_times[1:]-annotation_times[:-1]) // self.sampling_rate_ // EPOCH_TIME
        num_labels = (annotation_times[-1]-annotation_times[0]) // self.sampling_rate_ // EPOCH_TIME
        # remove last sleep phase since we don't know when it ends
        # TODO: check if we can assume it ends at the end of the recording
        annotations = annotations[:-1]
        # convert to integer labels
        int_label_mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 5}
        annotations = [int_label_mapping[x] for x in annotations]

        self.hypnogram = np.repeat(annotations, sleep_stage_durations)
        assert(len(self.hypnogram) == num_labels)
        self.psgs = {}
        for channel, signal in zip(data.sig_name, data.d_signal.T):
            # cut unlabeled signal parts (start and end)
            signal_cut = signal[annotation_times[0]:annotation_times[-1]]
            assert(len(signal_cut)// self.sampling_rate_ // EPOCH_TIME == num_labels)
            self.psgs[channel] = signal_cut.astype(np.int16)
