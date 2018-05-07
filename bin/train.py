import sys
import os
import torch
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.base import SleepLearning
from sacred.observers import FileStorageObserver
from sacred import Experiment, Ingredient

ex = Experiment()
ex.observers.append(FileStorageObserver.create('exp_logs'))

@ex.config
def cfg():
    # feature extraction settings
    neighbors = 4

    # training settings
    ts = {
        'batch-size': 32,
        'test-batch-size': 512,
        'epochs': 10,
        'lr': 5*1e-5,
        'resume': '',
        'cuda': torch.cuda.is_available(),
        'weight-loss': False
    }


@ex.automain
def main(ts):
    SleepLearning().train(ts)


