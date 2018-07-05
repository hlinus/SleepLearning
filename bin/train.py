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
def train(data_dir, loader, train_csv, val_csv, ts, channels, nbrs, cuda, fold,
          oversample, weighted_loss, batch_size_train, batch_size_val, log_dir,
          seed, _run):
    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if fold is None:
        fold = 0 # pick first and only column in csv files
        log_dir = os.path.join(log_dir, str(_run._id),
                           _run.experiment_info['name'])
    else:
        log_dir = os.path.join(log_dir, 'fold' + str(fold))

    print("log_dir: ", log_dir)
    print("seed: ", seed)

    with LogFileWriter(ex):
        logger = Logger(log_dir, _run)

    cfg_path = os.path.abspath(os.path.join(log_dir, os.pardir, 'config.json'))
    with open(cfg_path, 'w') as outfile:
        json.dump(_run.config, outfile)

    loader = utils.get_loader(loader)

    print("\nTRAINING SET: ")
    train_ds = utils.SleepLearningDataset(data_dir, train_csv, fold,
                                          ts['nclasses'],
                                          FeatureExtractor(
                                              channels).get_features(), nbrs,
                                          loader, verbose=True)
    train_loader = utils.get_sampler(train_ds, batch_size_train,
                               oversample, cuda, verbose=True)
    print("\nVAL SET: ")
    val_ds = utils.SleepLearningDataset(data_dir, val_csv, fold, ts['nclasses'],
                                        FeatureExtractor(
                                            channels).get_features(), nbrs,
                                        loader, verbose=True)
    val_loader = utils.get_sampler(val_ds, batch_size_val, False,
                             cuda, verbose=True)

    model, criterion, optimizer = utils.get_model(ts, weighted_loss, train_ds,
                                                  cuda)


    # Fit the model
    clf = Base(model, optimizer, criterion, logger, cuda)
    clf.fit(train_loader, val_loader, max_epoch=ts['epochs'])

    return clf.best_acc_


if __name__ == '__main__':
    args = sys.argv
    options = '_'.join(
        [x for x in args[2:min(len(args), 6)]])
    args += ['--name', options]
    ex.run_commandline(argv=args)
