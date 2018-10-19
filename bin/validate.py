import os
import shutil
import pandas as pd
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.base import Base
import sleeplearning.lib.utils as utils
import argparse
from cfg.config import *
import numpy as np
import glob


def main(args):
    subjects = pd.read_csv(args.subject_csv, header=None)[
        0].dropna().tolist()
    data_dir = '/cluster/scratch/hlinus/physionet-challenge-train/'

    if os.path.isdir(args.model):
        models = glob.glob(args.model + '/*')
    else:
        models = [args.model]

    for model in models:
        clf = Base(cuda=torch.cuda.is_available(), verbose=True)
        clf.restore(model)
        print("channels: \n", [c[0] for c in clf.ds['channels']])

        if args.channel_drop is not None:
            channel_drop = tuple(map(float, args.channel_drop[1:-1].split(',')))
            suffix = '_'+args.channel_drop
        else:
            channel_drop = None
            suffix = ''

        output_dir = os.path.join(args.output_dir, os.path.basename(
            os.path.normpath(model))+suffix)

        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print("last acc: ", clf.last_acc, "best acc: ", clf.best_acc_)

        channel_accuracies = np.array([])
        for subject in subjects:
            output, metrics = utils.get_model_output(clf, data_dir,
                                                     subject, channel_drop)
            accuracy = metrics['top1'].avg
            savedict = dict(subject=subject, acc=accuracy)
            for k, v in output.items():
                savedict[k] = v
            np.savez(os.path.join(output_dir, subject), **savedict)
            channel_accuracies = np.append(channel_accuracies, accuracy)
            print(f"\n{subject}: {accuracy}\n")
        print("mean acc: ", np.mean(channel_accuracies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SleepLearning Validation')
    parser.add_argument('--model',
                        default='../models/3008_Attnet_ChannelDrop0.1.pth.tar',
                        required=False,
                        help='file or folder of pytorch model (*.pth.tar')
    parser.add_argument('--data_dir',
                        default='/cluster/scratch/hlinus/physionet-challenge-train/',
                        help='folder containing psg files')
    parser.add_argument('--subject_csv',
                        default='../cfg/physionet18/test_rs50_0.csv',
                        help='csv file containing validation/test subjects')
    parser.add_argument('--output_dir',
                        default='/cluster/scratch/hlinus/AttentionNet',
                        help='folder where predictions are saved')
    parser.add_argument('--channel_drop',
                        required=False,
                        help='tuple of channel dropout probabilities')
    args = parser.parse_args()

    main(args)

