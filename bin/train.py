import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import torch
from sklearn.pipeline import Pipeline, FeatureUnion
from sleeplearning.lib.utils import SleepLearningDataset
from sleeplearning.lib.feature_extractor import Spectrogram, CutFrequencies, \
    LogTransform, TwoDScaler, PowerSpectralDensityMean
from sleeplearning.lib.base import SleepLearning
from sacred.observers import FileStorageObserver
from sacred import Experiment, Ingredient

carods_ingredient = Ingredient('carods')
ex = Experiment(base_dir=os.path.join(root_dir, 'sleeplearning', 'lib'), ingredients=[carods_ingredient])
ex.observers.append(FileStorageObserver.create(os.path.join(root_dir, 'exp_logs')))


@carods_ingredient.config
def carods_cfg():
    # default dataset settings
    train_dir = os.path.join('newDS', 'train')
    test_dir = os.path.join('newDS', 'test')
    num_labels = 5

    # feature extraction settings
    eeg_spectrogram = Pipeline([
        ('spectrogram',
         Spectrogram(channel='EEG', sampling_rate=100, window=156, stride=100)),
        ('cutter',
         CutFrequencies(window=156, sampling_rate=100, lower=0, upper=25)),
        ('log', LogTransform()),
        ('standard', TwoDScaler())
    ])

    emg_psd = Pipeline([
        ('spectrogram',
         Spectrogram(channel='EMG', sampling_rate=100, window=156, stride=100)),
        ('cutter',
         CutFrequencies(window=156, sampling_rate=100, lower=0, upper=60)),
        ('psd', PowerSpectralDensityMean(output_dim=40)),
        ('log', LogTransform()),
        ('standard', TwoDScaler())
    ])

    eogl = Pipeline([
        ('spectrogram',
         Spectrogram(channel='EOGL', sampling_rate=100, window=156,
                     stride=100)),
        ('cutter',
         CutFrequencies(window=156, sampling_rate=100, lower=0, upper=60)),
        ('psd', PowerSpectralDensityMean(output_dim=40)),
        ('log', LogTransform()),
        ('standard', TwoDScaler())
    ])

    eogr = Pipeline([
        ('spectrogram',
         Spectrogram(channel='EOGR', sampling_rate=100, window=156,
                     stride=100)),
        ('cutter',
         CutFrequencies(window=156, sampling_rate=100, lower=0, upper=60)),
        ('psd', PowerSpectralDensityMean(output_dim=40)),
        ('log', LogTransform()),
        ('standard', TwoDScaler())
    ])
    neighbors = 4

    feats = FeatureUnion(
        [('eeg_spectrogram', eeg_spectrogram), ('emg_psd', emg_psd),
         ('eogl_psd', eogl), ('eogr_psd', eogr)], n_jobs=2)



@ex.config
def cfg():
    # training settings
    batch_size = 32
    test_batch_size = 512
    epochs = 20
    lr = 5*1e-5
    resume = ''
    cuda = torch.cuda.is_available()
    weight_loss = False


@carods_ingredient.capture
def load_data(train_dir: os.path, test_dir: os.path, num_labels: int,
              feats, neighbors: int):
    if num_labels not in [3, 5]:
        raise ValueError('num_labels must be either 3 or 5')
    train_ds = SleepLearningDataset(train_dir, num_labels, feats, neighbors)
    val_ds = SleepLearningDataset(test_dir, num_labels, feats, neighbors)
    return train_ds, val_ds

@ex.automain
def main(_config):
    train_ds, val_ds = load_data()
    SleepLearning().train(_config['batch_size'],
                          _config['test_batch_size'],
                          _config['lr'],
                          _config['resume'],
                          _config['cuda'],
                          _config['weight_loss'],
                          _config['epochs'],
                          train_ds=train_ds,
                          val_ds=val_ds,
                          out_dir=ex.observers[0].dir)
