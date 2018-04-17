import os

import numpy as np

from sleeplearning.lib.base import SleepLearning


class Physionet(SleepLearning):
    def __init__(self, path: str, epoch_length: int = 30):
        super().__init__(path, epoch_length)
        self.psgs = {}
        with np.load(self.path) as f:
            self.psgs['EEG'] = f["x"].reshape(-1)
            self.hypnogram = f["y"]
            self.hypnogram[self.hypnogram == 5] = 6  # UNK -> ARTEFACT
            self.hypnogram[self.hypnogram == 4] = 5  # REM is now 5
            self.sampling_rate_ = int(f["fs"])
            self.epoch_length = 30
            self.label = os.path.basename(self.path)[:-4]