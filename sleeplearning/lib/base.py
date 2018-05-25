import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from scipy.signal import resample
from sleeplearning.lib.model import SlClassifier
from sleeplearning.lib.loaders.subject import Subject
from sleeplearning.lib.utils import SleepLearningDataset
from sleeplearning.lib.feature_extractor import *


class SleepLearning(object):
    def __init__(self):
        pass

    @staticmethod
    def train(train_dir, val_dir, ts, feats, nclasses, neighbors, seed, log_dir):
        train_ds = SleepLearning.load_data(train_dir, nclasses, feats,
                                           neighbors)
        val_ds = SleepLearning.load_data(val_dir, nclasses, feats,
                                           neighbors)
        inputdim = train_ds.dataset_info['input_shape']
        clf = SlClassifier(inputdim, nclasses, ts, seed, log_dir=log_dir)
        clf.fit(train_ds, val_ds)
        return clf.best_acc_

    @staticmethod
    def load_data(dir: os.path, num_labels: int, feats: dict, neighbors: int):
        ds = SleepLearningDataset(dir, num_labels,
                                        FeatureExtractor(feats).get_features(),
                                        neighbors)
        return ds

    @staticmethod
    def create_dataset(subjects: List[Subject], output_foldr: str):
        subject_labels = []
        outdir = '../data/processed/' + output_foldr + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            for subject in subjects:
                subject_labels.append(subject.label)
                psgs_reshaped = {}
                # pad all channels with zeros
                for k, psgs in subject.psgs.items():
                    psgs1 = psgs.reshape(
                        (-1, subject.sampling_rate_ * subject.epoch_length))
                    if subject.sampling_rate_ > 100:
                        # downsample to 100 Hz
                        psgs1 = resample(psgs1, subject.epoch_length * 100,
                                         axis=1)
                    psgs_reshaped[k] = psgs1
                np.savez(outdir + subject.label, subject_label=subject.label,
                         psgs=psgs_reshaped, labels=subject.hypnogram)
        else:
            raise ValueError('ERROR: the given dataset folder already exists!')
