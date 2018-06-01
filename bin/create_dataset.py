"""Script for creating Sleep Learning datasets """
###############################################################################
import glob
import shutil

import numpy as np
import os
import sys
import argparse
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
import sleeplearning.lib.utils as utils
from sleeplearning.lib.loaders.physionet_challenge18 import PhysionetChallenge18

# --------------------------------------------------------------------------- #
# -------------------- Command line arguments parsing ----------------------- #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(prog='Sleep Learning Dataset Creation')
subparsers = parser.add_subparsers(help='supported loader', dest="loader")

# create the parser for the "physionet" command
parser_physionet = subparsers.add_parser('physionet',
                                         help='for https://physionet.org/physiobank/database/challenge/2018/#files')
parser_physionet.add_argument("--data_dir", dest="data_dir", default='../data/raw/physionet_challenge/train',
                              help="directory of data files")
parser_physionet.add_argument("--out_dir", dest="out_dir", default='../data/processed/physionet_challenge/train',
                              help="directory to store the dataset")

# create the parser for the "physio" command
parser_b = subparsers.add_parser('sleepedf',
                                 help='url')

# parse some argument lists
args = vars(parser.parse_args())

# --------------------------------------------------------------------------- #
# ----------------- Load file and do basic preprocessing ---------------------#
# --------------------------------------------------------------------------- #
loader = None
files = None
if 'data_dir' not in args.keys():
    print("Please specify data directory.")
elif 'out_dir' not in args.keys():
    print("Please specify output directory.")
elif not os.path.isdir(args['data_dir']):
    print(args['data_dir'] + " DOES NOT EXIST!")
elif args['loader'] == 'physionet':
    loader = PhysionetChallenge18
    files = glob.glob(args['data_dir'] + '/*')
elif args['loader'] == 'sleepedf':
    print("not yet supported")

# --------------------------------------------------------------------------- #
# ----------------- Perform classification procedure ------------------------ #
# --------------------------------------------------------------------------- #
if loader is not None and files is not None:
    # Create out directory and delete if already exist
    if not os.path.exists(args['out_dir']):
        os.makedirs(args['out_dir'])
    else:
        shutil.rmtree(args['out_dir'])
        os.makedirs(args['out_dir'])

    subject_list = []
    for file in files:
        print('loading: ', file)
        subject = loader(file)
        utils.create_dataset([subject], args['out_dir'])

    print('DONE!')