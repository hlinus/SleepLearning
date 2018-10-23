import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import torch
import numpy as np
from sacred.stflow import LogFileWriter
from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
from cfg.config import ex
import json


@ex.main
def train(ds, arch, ms, cuda, log_dir, seed, save_model, save_best_only,
          early_stop,_run):
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
        import pathlib
        log_dir = os.path.join(log_dir, 'fold' + str(ds['fold']))
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        cfg_path = os.path.abspath(
            os.path.join(log_dir, os.pardir, 'config.json'))
        with open(cfg_path, 'w') as outfile:
            json.dump(_run.config, outfile)
        print("log_dir JSON: ", log_dir)
        if 'expert_models' in ms.keys():
            ms['expert_models'] = [os.path.join(x, f"fold{ds['fold']}",
                                                "checkpoint.pth.tar")
                                   for x in ms['expert_models']]

    if _run._id is not None:
        # Capture TensorBoard logs with sacred
        with LogFileWriter(ex):
            logger = Logger(log_dir, _run)
    else:
        logger = Logger(log_dir, None)

    # Fit the model
    clf = Base(logger=logger, cuda=cuda, verbose=True)
    clf.fit(arch, ms, **ds, early_stop=early_stop)

    _run.info['modelstr'] = str(clf.model)

    if save_model:
        clf.save_checkpoint_(save_best_only=save_best_only)

    return clf.best_acc_


if __name__ == '__main__':
    args = sys.argv
    options = '_'.join(
        [x for x in args[2:] if 'data_dir' not in x and 'train_csv' not in x
         and 'val_csv' not in x])
    args += ['--name', options]
    ex.run_commandline(argv=args)
