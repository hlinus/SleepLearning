import os
import torch
import sys
from sacred import Experiment, Ingredient
from sacred.observers import MongoObserver
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
carods_ingredient = Ingredient('carods')
physiods_ingredient = Ingredient('physiods')
ex = Experiment(base_dir=os.path.join(root_dir, 'sleeplearning', 'lib'))
MONGO_OBSERVER = MongoObserver.create(url='mongodb://toor:y0qXDe3qumoawG0rPfnS'
                                          '@cab-e81-31/admin?authMechanism'
                                          '=SCRAM-SHA-1', db_name='sacred')
ex.observers.append(MONGO_OBSERVER)


@ex.config
def cfg():
    # comment for this run
    cmt = ''

    # default dataset settings
    data_dir = os.path.join('../../../physionet-challenge-train')


    # feature settings
    nclasses = 5
    neighbors = 0

    # training settings
    ts = {
        'model': 'MediumAdaptive',
        'batch_size_train': 32,
        'batch_size_val': 250,
        'dropout': .5,
        'epochs': 100,
        'optim': 'adam,lr=0.00005',
        'cuda': torch.cuda.is_available(),
        'weighted_loss': True,
        'oversample': False,
        'fold': None, # only specify for CV
        'log_dir': None
    }

    # seed
    seed = 42


@ex.named_config
def sleepedf():
    loader = 'Sleepedf'
    feats = {
        'channels': [
            ('EEG-Fpz-Cz', []),
        ]
    }

@ex.named_config
def three_channels_noscale():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', []),
            ('C4-M1', []),
            ('E1-M2', []),
        ]
    }

@ex.named_config
def three_channels():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', ['OneDScaler()']),
            ('C4-M1', ['OneDScaler()']),
            ('E1-M2', ['OneDScaler()']),
        ]
    }

@ex.named_config
def three_channels_int16():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', ['ConvToInt16()']),
            ('C4-M1', ['ConvToInt16()']),
            ('E1-M2', ['ConvToInt16()']),
        ]
    }

@ex.named_config
def three_channels_filt():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
            ('C4-M1', ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
            ('E1-M2', ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
        ]
    }

@ex.named_config
def seven_channels_int16():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', ['ConvToInt16()']),
            ('C4-M1', ['ConvToInt16()']),
            ('C3-M2', ['ConvToInt16()']),
            ('E1-M2', ['ConvToInt16()']),
            ('F4-M1', ['ConvToInt16()']),
            ('O1-M2', ['ConvToInt16()']),
            ('O2-M1', ['ConvToInt16()']),
        ]
    }

@ex.named_config
def one_channel():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', ['OneDScaler()']),
        ]
    }

@ex.named_config
def one_channel_int16():
    loader = 'Physionet18'
    feats = {
        'channels': [
            ('F3-M2', ['ConvToInt16()'])
        ]
    }