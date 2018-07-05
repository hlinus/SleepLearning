import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from cfg.config import *
import torch
import numpy as np
from sacred.stflow import LogFileWriter
from sleeplearning.lib.feature_extractor import FeatureExtractor
from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
import sleeplearning.lib.utils as utils
import json

@ex.main
def train(data_dir, loader, train_csv, val_csv, ts, feats, nclasses, neighbors,
          seed, _run):
    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if ts['fold'] is None:
        # TODO: do not overwrite
        ts['fold'] = 0 # pick first and only column in csv files
        log_dir = os.path.join(root_dir, 'logs', str(_run._id),
                           _run.experiment_info['name'])
    else:
        log_dir = os.path.join(ts['log_dir'],
                               'fold' + str(ts['fold']))

    print("log_dir: ", log_dir)
    print("seed: ", seed)
    with LogFileWriter(ex):
        logger = Logger(log_dir, _run)

    with open(os.path.abspath(os.path.join(log_dir, os.pardir, 'config.json')), 'w') as outfile:
        json.dump(_run.config, outfile)

    loader = utils.get_loader(loader)

    print("\nTRAINING SET: ")
    train_ds = utils.SleepLearningDataset(data_dir, train_csv, ts['fold'], nclasses,
                                    FeatureExtractor(
                                        feats).get_features(), neighbors,
                                    loader, verbose=True)
    train_loader = utils.get_sampler(train_ds, ts['batch_size_train'],
                               ts['oversample'], ts['cuda'], verbose=True)
    print("\nVAL SET: ")
    val_ds = utils.SleepLearningDataset(data_dir, val_csv, ts['fold'], nclasses,
                                  FeatureExtractor(feats).get_features(),
                                  neighbors, loader,
                                  verbose=True)
    val_loader = utils.get_sampler(val_ds, ts['batch_size_val'], False,
                             ts['cuda'], verbose=True)

    model, criterion, optimizer = utils.get_model(ts, nclasses, train_ds)


    # Fit the model
    clf = Base(model, optimizer, criterion, logger, ts['cuda'])
    clf.fit(train_loader, val_loader, max_epoch=ts['epochs'])

    return clf.best_acc_


if __name__ == '__main__':
    args = sys.argv
    options = '_'.join(
        [x for x in args[2:min(len(args), 6)]])
    args += ['--name', options]
    ex.run_commandline(argv=args)
