import os
import sys
import torch
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import platform
import numpy as np

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
log_base_dir = os.path.join('..', 'logs', platform.node())
ex.observers.append(FileStorageObserver.create(log_base_dir))


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
    # default dataset settings
    train_dir = os.path.join('../../../physionet-challenge-train/debug')
    val_dir = os.path.join('../../../physionet-challenge-train/validation')

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


@ex.config
def cfg():
    cmt = '' # comment
    # feature settings
    nclasses = 5
    neighbors = 0
    num_val_subjects = 2
    # training settings
    ts = {
        'model': 'DeepFeatureNet',
        'batch_size_train': 32,
        'batch_size_val': 128,
        'epochs': 50,
        'optim': 'adam,lr=0.00005',
        'cuda': torch.cuda.is_available(),
        'weighted_loss': True
    }
    seed = 42



@ex.automain
def main(train_dir, val_dir, num_val_subjects, ts, feats, nclasses, neighbors,
         seed, _run):
    l = _run.meta_info['options']['UPDATE'] if 'UPDATE' in \
                                               _run.meta_info['options'] \
                                            else 'grid'
    options = '_'.join([x for x in l if 'train_dir' not in x and 'val_dir' not in x])
    log_dir = os.path.join(log_base_dir, str(_run._id), options)
    print("log_dir: ", log_dir)
    print("seed: ", seed)

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("\nTRAIN SUBJECTS: ")
    train_loader, dataset_info = utils.load_data(train_dir, nclasses, feats,
                                             neighbors, 1000,
                                             PhysionetChallenge18,
                                             ts['batch_size_train'], ts['cuda'],
                                             verbose=True)
    print("\nVALIDATION SUBJECTS: ")
    val_loader, _ = utils.load_data(val_dir, nclasses, feats, neighbors,
                                    num_val_subjects, PhysionetChallenge18,
                                    ts['batch_size_val'], ts['cuda'],
                                    verbose=True)

    if ts['model'] == 'SleepStage':
        model = SleepStage(nclasses, dataset_info['input_shape'])
    elif ts['model'] == 'DeepFeatureNet':
        model = DeepFeatureNet(nclasses, dataset_info['input_shape'])
    else:
        raise ValueError(ts['model'] + ' does not exist!')
    optim_fn, optim_params = utils.get_optimizer(ts['optim'])

    optimizer = optim_fn(model.parameters(), **optim_params)
    if ts['weighted_loss']:
        # TODO: assure weights are in correct order
        counts = np.fromiter(dataset_info['class_distribution'].values(),
                             dtype=int)
        normed_counts = counts / np.min(counts)
        weights = np.reciprocal(normed_counts).astype(np.float32)
        print("weighted loss: ", weights)
    else:
        weights = np.ones(nclasses)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    logger = Logger(log_dir)

    if ts['cuda']:
        model.cuda()
        criterion.cuda()

    # Fit the model
    clf = Base(model, optimizer, criterion, logger, ts['cuda'])
    clf.fit(train_loader, val_loader, max_epoch=ts['epochs'])

    return clf.best_acc_