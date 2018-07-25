import argparse
import glob
import os
import sys

from sklearn.metrics import accuracy_score, confusion_matrix

from sleeplearning.lib.logger import cm_figure_

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from cfg.config import *
import numpy as np
parser = argparse.ArgumentParser(prog='Sleep Learning cross validation')
parser.add_argument("--data_dir", dest="fn", default=os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '..')) + '/logs/ZZ_test5',
                        help="filepath of cv folds")


def evaluate(files):
    accuracies = np.array([])
    targets = np.array([], dtype=int)
    predictions = np.array([], dtype=int)
    for f in files:
        results = np.load(f)
        tar = results['targets'].astype('int')
        pred = results['predictions'].astype('int')
        epoch = results['epoch'].astype('int')
        acc_f = results['acc']
        acc = accuracy_score(tar, pred)
        print("fold: ",f , " epoch: ", epoch, " acc:",acc, "accf:", acc_f)
        accuracies = np.append(accuracies, acc)
        targets = np.append(targets, tar)
        predictions = np.append(predictions, pred)

    cm = confusion_matrix(targets, predictions)
    print("confusion matrix: ")
    print(cm)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print("\n\naccuracies: ", accuracies)
    print("mean: ", mean_acc, "std: ", std_acc, "\n\n")
    fig = cm_figure_(predictions, targets, ["W", "N1", "N2", "N3", "REM"])
    return mean_acc, fig

@ex.main
def cv(ds, arch, ms, cuda, log_dir, seed, _run):
    print("\n\n TRAIN: \n\n")
    train_files = glob.glob(os.path.join(log_dir, '**',
                                               '*train*last.npz'),
                          recursive=True)

    _, f_train = evaluate(train_files)
    f_train.savefig('train_cm.png')

    print("\n\n VALIDATION: \n\n")
    val_files = glob.glob(os.path.join(log_dir, '**', '*val*last.npz'), recursive=True)
    mean_val_acc, f_val = evaluate(val_files)
    f_val.savefig('val_cm.png')
    return mean_val_acc*100


if __name__ == '__main__':
    args = sys.argv
    args += ['--name', args[-1]]
    ex.run_commandline(argv=args)

