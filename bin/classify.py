import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.base import Base
import argparse
from cfg.config import *
import numpy as np


def main(args):
    clf = Base(cuda=torch.cuda.is_available(), verbose=False)
    clf.restore(args.model)
    print("channels: \n", [c[0] for c in clf.ds['channels']])

    prediction = clf.predict(args.subject)
    subject_name = os.path.basename(os.path.normpath(args.subject))
    output_path = os.path.join(args.output_dir, subject_name + '.csv')
    np.savetxt(output_path, prediction, delimiter=",", fmt='%i')
    print(f"Prediction saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SleepLearning Validation')
    parser.add_argument('--model',
                        default='../models/cv_sleepedf_Fz_2D_singlechanexp2_6464962FC_MP/fold0/checkpoint.pth.tar',
                        required=False,
                        help='file or folder of pytorch model (*.pth.tar')
    parser.add_argument('--subject',
                        default='../data/sleepedf/SC4001E0-PSG',
                        help='subject file to predict')
    parser.add_argument('--output_dir',
                        default='',
                        help='folder where predictions are saved')
    args = parser.parse_args()

    main(args)

