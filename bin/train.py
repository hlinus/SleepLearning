import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import torch
from sleeplearning.lib.utils import SleepLearningDataset
from sleeplearning.lib.feature_extractor import FeatureExtractor
from sleeplearning.lib.base import SleepLearning
from sacred.observers import TinyDbObserver
from sacred import Experiment, Ingredient
import platform
carods_ingredient = Ingredient('carods')
physiods_ingredient = Ingredient('physiods')
ex = Experiment(base_dir=os.path.join(root_dir, 'sleeplearning', 'lib'))
ex.observers.append(
    TinyDbObserver.create(os.path.join(root_dir, 'train_logs_'+str(platform.node()))))


@ex.named_config
def physio_chal18():
    # default dataset settings
    train_dir = os.path.join('physionet_chal18', 'train')
    val_dir = os.path.join('physionet_chal18', 'val')
    num_labels = 5

    # feature extraction settings
    neighbors = 4

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': [
            {'channel': 'F4-M1',
             'cut_lower': 0,
             'cut_upper': 25,
             'psd': 'none',
             'log': True,
             'scale': '2D'},
            {'channel': 'CHEST',
             'cut_lower': 0,
             'cut_upper': 60,
             'psd': 'mean',
             'log': True,
             'scale': '2D'},
            {'channel': 'E1-M2',
             'cut_lower': 0,
             'cut_upper': 60,
             'psd': 'mean',
             'log': True,
             'scale': '2D'}
        ]
    }


@ex.named_config
def physiods():
    # default dataset settings
    train_dir = os.path.join('physionet_mini2', 'train')
    val_dir = os.path.join('physionet_mini2', 'val')
    num_labels = 5

    # feature extraction settings
    neighbors = 0

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': [
            {'channel': 'EEG Fpz-Cz',
             'cut_lower': 0,
             'cut_upper': 25,
             'psd': 'none',
             'log': False,
             'scale': '2D'}
        ]
    }


@ex.named_config
def carods():
    # default dataset settings
    train_dir = os.path.join('newDS', 'train')
    val_dir = os.path.join('newDS', 'test')
    num_labels = 5

    # feature extraction settings
    neighbors = 4

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': [
            {'channel': 'EEG',
             'cut_lower': 0,
             'cut_upper': 25,
             'psd': 'none',
             'log': True,
             'scale': '2D'},
            {'channel': 'EMG',
             'cut_lower': 0,
             'cut_upper': 60,
             'psd': 'mean',
             'log': True,
             'scale': '2D'},
            {'channel': 'EOGL',
             'cut_lower': 0,
             'cut_upper': 60,
             'psd': 'mean',
             'log': True,
             'scale': '2D'},
            {'channel': 'EOGR',
             'cut_lower': 0,
             'cut_upper': 60,
             'psd': 'mean',
             'log': True,
             'scale': '2D'}
        ]
    }


@ex.config
def cfg():
    # training settings
    batch_size = 32
    test_batch_size = 512
    epochs = 50
    lr = 5 * 1e-5
    resume = ''
    cuda = torch.cuda.is_available()
    weight_loss = False


@ex.capture
def load_data(train_dir: os.path, val_dir: os.path, num_labels: int,
              feats: dict, neighbors: int):
    if num_labels not in [3, 5]:
        raise ValueError('num_labels must be either 3 or 5')
    train_ds = SleepLearningDataset(train_dir, num_labels, FeatureExtractor(feats).get_features(), neighbors)
    val_ds = SleepLearningDataset(val_dir, num_labels, FeatureExtractor(feats).get_features(), neighbors)
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
                          out_dir=os.path.join(root_dir, 'models'))