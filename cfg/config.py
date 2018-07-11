import os
import torch
import sys
#from sacred import Ingredient
from sacred import Experiment
from sacred.observers import MongoObserver
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)

basedir = os.path.join(root_dir, 'sleeplearning', 'lib')
#ex = Experiment(base_dir=basedir, ingredients=[general])
ex = Experiment(base_dir=basedir)
mongo_url = 'mongodb://toor:y0qXDe3qumoawG0rPfnS@cab-e81-31/admin?authMechanism' \
            '=SCRAM-SHA-1'
MONGO_OBSERVER = MongoObserver.create(url=mongo_url, db_name='sacred')
ex.observers.append(MONGO_OBSERVER)


@ex.config
def cfg():
    cmt = ''  # comment for this run
    cuda = torch.cuda.is_available()
    seed = 42  # for reproducibility
    log_dir = '../logs'

    # default dataset settings
    ds = {
        'channels': None,
        'data_dir': os.path.join('../../../physionet-challenge-train'),
        'train_csv': None,
        'val_csv': None,
        'batch_size_train': 32,
        'batch_size_val': 250,
        'loader': 'Physionet18',
        'nbrs': 0,
        'fold': None,  # only specify for CV
        'oversample': False,
        'nclasses': 5,
    }



@ex.named_config
def multvarnet():
    arch = 'MultivariateNet'

    ms = {
         'epochs': 200,
         'dropout': .5,
         'optim': 'adam,lr=0.00005',
         'fc_d' : [[4096,.3],[2048,.3],[1048,.1],[512,0]],
         'input_dim': None,  # will be set automatically
         'weighted_loss': True
    }


@ex.named_config
def E1M2():
    ds = {
        'channels': [('E1-M2', ['ConvToInt16()'])]
    }

@ex.named_config
def O2M1():
    ds = {
        'channels': [('O2-M1', ['ConvToInt16()'])]
    }

@ex.named_config
def C4M1():
    ds = {
        'channels': [('C4-M1', ['ConvToInt16()'])]
    }

@ex.named_config
def C3M2():
    ds = {
        'channels': [('C3-M2', ['ConvToInt16()'])]
    }

@ex.named_config
def F3M2():
    ds = {
        'channels': [('F3-M2', ['ConvToInt16()'])]
    }

@ex.named_config
def F4M1():
    ds = {
        'channels': [('F4-M1', ['ConvToInt16()'])]
    }

@ex.named_config
def O1M2():
    ds = {
        'channels': [('O1-M2', ['ConvToInt16()'])]
    }



@ex.named_config
def sleepedf():
    loader = 'Sleepedf'

    channels = [
            ('EEG-Fpz-Cz', []),
    ]


@ex.named_config
def three_channels_noscale():


    ds ={
        'channels': [
            ('F3-M2', []),
            ('C4-M1', []),
            ('E1-M2', []),
    ]
    }

@ex.named_config
def three_channels():


    ds ={
        'channels': [
            ('F3-M2', ['OneDScaler()']),
            ('C4-M1', ['OneDScaler()']),
            ('E1-M2', ['OneDScaler()']),
    ]
    }

@ex.named_config
def three_channels_int16():


    ds ={
        'channels': [
            ('F3-M2', ['ConvToInt16()']),
            ('C4-M1', ['ConvToInt16()']),
            ('E1-M2', ['ConvToInt16()']),
        ]
    }


@ex.named_config
def three_channels_filt():


    ds ={
        'channels': [
            ('F3-M2', ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
            ('C4-M1', ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
            ('E1-M2', ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
    ]
    }

@ex.named_config
def seven_channels_int16():

    ds ={
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

    ds ={
        'channels': [
            ('F3-M2', ['OneDScaler()']),
    ]
    }