import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import torch
import numpy as np
from cfg.config import ex
from sacred.stflow import LogFileWriter
from sleeplearning.lib.feature_extractor import FeatureExtractor
from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
import sleeplearning.lib.utils as utils
import json



@ex.main
def train(ds, arch, ms, cuda, log_dir, seed, _run):
    # fix seed
    print("seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if ds['fold'] is None:
        ds['fold'] = 0  # pick first and only column in csv files
        log_dir = os.path.join(log_dir, str(_run._id),
                           _run.experiment_info['name'])
    else:
        log_dir = os.path.join(log_dir, 'fold' + str(ds['fold']))

    if _run._id is not None:
        with LogFileWriter(ex):
            logger = Logger(log_dir, _run)
            cfg_path = os.path.abspath(os.path.join(log_dir, os.pardir, 'config.json'))
            with open(cfg_path, 'w') as outfile:
                json.dump(_run.config, outfile)
            print("log_dir: ", log_dir)
    else:
        log_dir = None
        logger = None

    # Fit the model
    clf = Base(logger=logger, cuda=cuda, verbose=True)
    clf.fit(arch, ms, **ds)

    return clf.best_acc_


if __name__ == '__main__':
    args = sys.argv
    options = '_'.join(
        [x for x in args[2:] if 'data_dir' not in x and 'train_csv' not in x
         and 'val_csv' not in x])
    args += ['--name', options]
    ex.run_commandline(argv=args)
