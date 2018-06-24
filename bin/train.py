import os
import sys
import torch
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver, MongoObserver
import platform
import numpy as np
from sacred.stflow import LogFileWriter

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.models.deep_sleep_net import DeepFeatureNet
from sleeplearning.lib.models.sleep_stage import SleepStage

from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
import sleeplearning.lib.utils as utils
from sleeplearning.lib.loaders.physionet_challenge18 import PhysionetChallenge18
carods_ingredient = Ingredient('carods')
physiods_ingredient = Ingredient('physiods')
ex = Experiment(base_dir=os.path.join(root_dir, 'sleeplearning', 'lib'))
MONGO_OBSERVER = MongoObserver.create(url='mongodb://toor:y0qXDe3qumoawG0rPfnS'
                                          '@cab-e81-31/admin?authMechanism'
                                          '=SCRAM-SHA-1', db_name='sacred')
ex.observers.append(MONGO_OBSERVER)


@ex.named_config
def physio18_animal():
    # default dataset settings
    #train_dir = os.path.join('../../../physionet-challenge-train/debug')
    val_dir = os.path.join('../../../physionet-challenge-train/debug')

    feats = {
        'channels': [
            ('C4-M1', [
                #'BandPass( fs=100, highpass=0.3, lowpass=45)',
                'Spectrogram(fs=100, window=50, stride=25)',
                #'LogTransform()',
                'TwoDScaler()'
                ]),
            ('F3-M2', [
                #'BandPass( fs=100, highpass=0.3, lowpass=45)',
                'Spectrogram(fs=100, window=50, stride=25)',
                #'LogTransform()',
                'TwoDScaler()'
                ]),
            ('E1-M2', [
                #'BandPass(fs=100, highpass=0.3, lowpass=12)',
                'Spectrogram(fs=100, window=50, stride=25)',
                #'LogTransform()',
                'TwoDScaler()'
                ]),
        ]
    }

@ex.named_config
def physio18_dfn():
    feats = {
        'channels': [
            ('F3-M2', [
                #'BandPass( fs=100, highpass=0.3, lowpass=45)',
                #'LogTransform()',
                'OneDScaler()'
                ]),
            ('C4-M1', [
                # 'BandPass( fs=100, highpass=0.3, lowpass=45)',
                # 'LogTransform()',
                'OneDScaler()'
            ]),
            ('E1-M2', [
                # 'BandPass( fs=100, highpass=0.3, lowpass=45)',
                # 'LogTransform()',
                'OneDScaler()'
            ]),
        ]
    }

@ex.named_config
def three_channels_noscale():
    feats = {
        'channels': [
            ('F3-M2', []),
            ('C4-M1', []),
            ('E1-M2', []),
        ]
    }

@ex.named_config
def three_channels():
    feats = {
        'channels': [
            ('F3-M2', ['OneDScaler()']),
            ('C4-M1', ['OneDScaler()']),
            ('E1-M2', ['OneDScaler()']),
        ]
    }

@ex.named_config
def one_channel():
    feats = {
        'channels': [
            ('F3-M2', ['OneDScaler()']),
        ]
    }

@ex.config
def cfg():
    # comment for this run
    cmt = ''

    # default dataset settings
    train_dir = os.path.join('../../../physionet-challenge-train/debug')
    val_dir = os.path.join('../../../physionet-challenge-train/validation')

    # feature settings
    nclasses = 5
    neighbors = 0
    num_val_subjects = 40

    # training settings
    ts = {
        'model': 'DeepFeatureNet',
        'batch_size_train': 32,
        'batch_size_val': 128,
        'dropout': .5,
        'epochs': 100,
        'optim': 'adam,lr=0.00005',
        'cuda': torch.cuda.is_available(),
        'weighted_loss': False,
        'oversample': False
    }

    # seed
    seed = 42


@ex.main
def train(train_dir, val_dir, num_val_subjects, ts, feats, nclasses, neighbors,
         seed, _run):
    log_dir = os.path.join(root_dir, 'logs', str(_run._id), _run.experiment_info['name'])
    print("log_dir: ", log_dir)
    print("seed: ", seed)
    with LogFileWriter(ex):
        logger = Logger(log_dir, _run)

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("\nTRAINING SET: ")
    train_loader, dataset_info = utils.load_data(train_dir, nclasses, feats,
                                             neighbors, 1000,
                                             PhysionetChallenge18,
                                             ts['batch_size_train'],
                                             ts['oversample'], ts['cuda'],
                                             verbose=True)
    print("\nVAL SET: ")
    val_loader, _ = utils.load_data(val_dir, nclasses, feats, neighbors,
                                    num_val_subjects, PhysionetChallenge18,
                                    ts['batch_size_val'], ts['cuda'],
                                    verbose=True)

    if ts['model'] == 'SleepStage':
        model = SleepStage(nclasses, dataset_info['input_shape'])
    elif ts['model'] == 'DeepFeatureNet':
        model = DeepFeatureNet(nclasses, dataset_info['input_shape'], ts['dropout'])
    else:
        raise ValueError(ts['model'] + ' does not exist!')
    optim_fn, optim_params = utils.get_optimizer(ts['optim'])

    optimizer = optim_fn(model.parameters(), **optim_params)
    if ts['weighted_loss']:
        # TODO: assure weights are in correct order
        counts = np.fromiter(dataset_info['class_distribution'].values(),
                             dtype=float)
        normed_counts = counts / np.min(counts)
        weights = np.reciprocal(normed_counts).astype(np.float32)
    else:
        weights = np.ones(nclasses)
    print("\nCLASS WEIGHTS (LOSS): ", weights)
    weights = torch.from_numpy(weights).type(torch.FloatTensor)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    print('\n')

    if ts['cuda']:
        model.cuda()
        criterion.cuda()

    # Fit the model
    clf = Base(model, optimizer, criterion, logger, ts['cuda'])
    clf.fit(train_loader, val_loader, max_epoch=ts['epochs'])

    return clf.best_acc_

if __name__ == '__main__':
    args = sys.argv
    options = '_'.join(
        [x for x in args[2:] if 'train_dir' not in x and 'val_dir' not in x])
    args += ['--name', options]
    ex.run_commandline(argv=args)
