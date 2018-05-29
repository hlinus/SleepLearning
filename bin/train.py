import os
import sys
import torch
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import platform
import numpy as np

from sleeplearning.lib.models.deep_sleep_net import DeepSleepNet
from sleeplearning.lib.models.sleep_stage import SleepStage

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
import sleeplearning.lib.utils as utils

carods_ingredient = Ingredient('carods')
physiods_ingredient = Ingredient('physiods')
ex = Experiment(base_dir=os.path.join(root_dir, 'sleeplearning', 'lib'))
log_base_dir = os.path.join('..', 'logs', platform.node())
ex.observers.append(FileStorageObserver.create(log_base_dir))


@ex.named_config
def physio_chal18():
    # default dataset settings
    train_dir = os.path.join('physionet_chal18', 'train')
    val_dir = os.path.join('physionet_chal18', 'val')

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': {
            'F4-M1': {
                'cut_lower': 0,
                'cut_upper': 25,
                'psd': 'none',
                'log': True,
                'scale': '2D'},
            'CHEST': {
                'cut_lower': 0,
                'cut_upper': 60,
                'psd': 'mean',
                'log': True,
                'scale': '2D'},
            'E1-M2': {
                'cut_lower': 0,
                'cut_upper': 60,
                'psd': 'mean',
                'log': True,
                'scale': '2D'}
        }
    }


@ex.named_config
def sleepedfminideepsleep():
    # default dataset settings
    train_dir = os.path.join('sleepedfmini', 'train')
    val_dir = os.path.join('sleepedfmini', 'val')

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': {
            'EEG-Fpz-Cz': {
                # 'spectrogram' : {'fs': 100, 'window': 156, 'stride': 100}
                # 'cut' : {'lower': 0, 'upper': 100}
                # 'psd' : {'type': 'mean'}
                'log': False,
                'scale': '1D'
            }
        }
    }

@ex.named_config
def sleepedfmini():
    # default dataset settings
    train_dir = os.path.join('sleepedfmini', 'train')
    val_dir = os.path.join('sleepedfmini', 'val')

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': {
            'EEG-Fpz-Cz': {
                'spectrogram' : {'fs': 100, 'window': 156, 'stride': 100},
                'cut' : {'lower': 0, 'upper': 100},
                'log': False,
                'scale': '1D'
            }
        }
    }



@ex.named_config
def sleepedf():
    # default dataset settings
    train_dir = os.path.join('sleepedf', 'train')
    val_dir = os.path.join('sleepedf', 'val')

    feats = {
        'sampling_rate': 100,
        'window': 156,
        'stride': 100,
        'channels': {
            'EEG-Fpz-Cz': {
                #'spectrogram' : {'fs': 100, 'window': 156, 'stride': 100}
                # 'cut' : {'lower': 0, 'upper': 100}
                #'psd' : {'type': 'mean'}
                'log': False,
                'scale': '1D'
            }
        }
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
        'channels': {
            'EEG': {
                'cut_lower': 0,
                'cut_upper': 25,
                'psd': 'none',
                'log': True,
                'scale': '2D'
            },
            'EMG': {
                'cut_lower': 0,
                'cut_upper': 60,
                'psd': 'mean',
                'log': True,
                'scale': '2D'
            },
            'EOGL': {
                'cut_lower': 0,
                'cut_upper': 60,
                'psd': 'mean',
                'log': True,
                'scale': '2D'
            },
            'EOGR': {
                'cut_lower': 0,
                'cut_upper': 60,
                'psd': 'mean',
                'log': True,
                'scale': '2D'
            }
        }
    }


@ex.config
def cfg():
    # feature settings
    nclasses = 5
    neighbors = 0

    # training settings
    ts = {
        'model': 'SleepStage',
        'batch_size': 32,
        'epochs': 50,
        'optim': 'adam,lr=0.00005',
        'cuda': torch.cuda.is_available(),
    }
    seed = 42



@ex.automain
def main(train_dir, val_dir, ts, feats, nclasses, neighbors, seed, _run):
    options = '_'.join(_run.meta_info['options']['UPDATE'] if 'UPDATE' in _run.meta_info['options'] else 'grid')
    log_dir = os.path.join(log_base_dir, str(_run._id), options)
    print("seed: ", seed)

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loader, inputdim = utils.load_data(train_dir, nclasses,
                                                     feats, neighbors,
                                                     ts['batch_size'],
                                                     ts['cuda'],
                                                     verbose=True)
    val_loader, _ = utils.load_data(val_dir, nclasses, feats,
                                            neighbors, ts['batch_size'],
                                            ts['cuda'])

    if ts['model'] == 'SleepStage':
        model = SleepStage(nclasses, inputdim)
    else:
        model = DeepSleepNet(nclasses, inputdim)
    optim_fn, optim_params = utils.get_optimizer(ts['optim'])

    # TODO: recheck if this works
    optimizer = optim_fn(model.parameters(), **optim_params)
    criterion = torch.nn.CrossEntropyLoss()
    logger = Logger(log_dir)

    if ts['cuda']:
        model.cuda()
        criterion.cuda()
        optimizer.cuda()

    # Fit the model
    clf = Base(model, optimizer, criterion, logger, ts['cuda'])
    clf.fit(train_loader, val_loader, max_epoch=ts['epochs'])

    return clf.best_acc_