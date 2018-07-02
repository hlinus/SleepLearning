import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from cfg.config import *
import torch
import numpy as np
from sacred.stflow import LogFileWriter
from sleeplearning.lib.models import *
from sleeplearning.lib.feature_extractor import FeatureExtractor
from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
import sleeplearning.lib.utils as utils
from sleeplearning.lib.loaders.physionet_challenge18 import PhysionetChallenge18


def test(clf, loader, test_dir, subject_csv):
    for _, row in pd.read_csv('test_subjects.csv').iterrows():
        subject = row[0]
        #utils.load_data()
    pass

@ex.main
def train(data_dir, train_csv, val_csv, ts, feats, nclasses, neighbors,
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
    train_ds = utils.SleepLearningDataset(data_dir, train_csv, nclasses,
                                    FeatureExtractor(feats).get_features(), neighbors,
                                    PhysionetChallenge18, verbose=True)
    train_loader = utils.get_loader(train_ds, ts['batch_size_train'], ts['oversample'],
                                  ts['cuda'],
                                  verbose=True)
    print("\nVAL SET: ")
    val_ds = utils.SleepLearningDataset(data_dir, val_csv, nclasses,
                                          FeatureExtractor(
                                              feats).get_features(), neighbors,
                                          PhysionetChallenge18, verbose=True)
    val_loader = utils.get_loader(val_ds, ts['batch_size_val'], False, ts['cuda'],
                                    verbose=True)

    # TODO: refactor to utils.get_model(model: str)
    ind = [i for i in range(len(ts['model'])) if str.isupper(ts['model'][i])]
    module_name = ''.join([ts['model'][i]+'_' if (i+1) in ind else str.lower(ts['model'][i]) for i in range(len(ts['model']))])
    model = eval(module_name+'.'+ts['model'])(nclasses, train_ds.dataset_info['input_shape'], ts['dropout'])
    optim_fn, optim_params = utils.get_optimizer(ts['optim'])

    optimizer = optim_fn(model.parameters(), **optim_params)
    # TODO: refactor to utils.get_loss(train_ds, weighted_loss: bool)
    if ts['weighted_loss']:
        # TODO: assure weights are in correct order
        counts = np.fromiter(train_ds.dataset_info['class_distribution'].values(),
                             dtype=float)
        normed_counts = counts / np.min(counts)
        weights = np.reciprocal(normed_counts).astype(np.float32)
    else:
        weights = np.ones(nclasses)
    print("\nCLASS WEIGHTS (LOSS): ", weights)
    weights = torch.from_numpy(weights).type(torch.FloatTensor)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    print('\n', model)
    print('\n')

    if ts['cuda']:
        model.cuda()
        criterion.cuda()
        weights.cuda()

    # Fit the model
    clf = Base(model, optimizer, criterion, logger, ts['cuda'])
    clf.fit(train_loader, val_loader, max_epoch=ts['epochs'])

    return clf.best_acc_

if __name__ == '__main__':
    args = sys.argv
    options = '_'.join(
        [x for x in args[2:] if 'data_dir' not in x and 'train_csv' not in x
         and 'val_csv' not in x])
    args += ['--name', options]
    ex.run_commandline(argv=args)
