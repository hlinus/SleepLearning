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
#ex.observers.append(
#    TinyDbObserver.create(os.path.join(root_dir, 'train_logs_'+str(platform.node()))))


@ex.named_config
def physio_chal18():
    # default dataset settings
    train_dir = os.path.join('physionet_chal18', 'train')
    val_dir = os.path.join('physionet_chal18', 'val')

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
    train_dir = os.path.join('physionet_mini', 'train')
    val_dir = os.path.join('physionet_mini', 'val')

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': [
            {'channel': 'EEG Fpz-Cz',
             'cut_lower': 0,
             'cut_upper': 100,
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
    # feature settings
    nclasses = 5
    neighbors = 0

    # training settings
    ts = {
        'model': 'SleepStage',
        'batch_size' : 32,
        'epochs' : 50,
        'optim' : 'adam,lr=0.00005',
        'cuda' : torch.cuda.is_available(),
    }
    seed = 42



@ex.automain
def main(train_dir, val_dir, ts, feats, nclasses, neighbors, seed):
    print("seed: ", seed)
    best_prec1 = SleepLearning(seed).train(train_dir, val_dir, ts, feats, nclasses,
                                     neighbors, seed)
    return best_prec1