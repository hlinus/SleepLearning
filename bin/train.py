import os
import sys
import torch
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import platform
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.base import SleepLearning
carods_ingredient = Ingredient('carods')
physiods_ingredient = Ingredient('physiods')
ex = Experiment(base_dir=os.path.join(root_dir, 'sleeplearning', 'lib'))


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
    logfoldr = 'debug'



@ex.automain
def main(train_dir, val_dir, ts, feats, nclasses, neighbors, seed, _run):
    run_id = '_'.join(_run.meta_info['options']['UPDATE']) + \
             "_{:%Y-%m-%d-%H-%M-%S}".format(_run.start_time)
    log_dir = os.path.join('..', 'logs', platform.node(),run_id)
    ex.observers.append(FileStorageObserver.create(log_dir))
    print("seed: ", seed)
    best_prec1 = SleepLearning().train(train_dir, val_dir, ts, feats, nclasses,
                                     neighbors, seed, log_dir)
    return best_prec1