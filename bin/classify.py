"""Script for classifying sleep phases"""
###############################################################################
import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from sleeplearning.lib.loaders.carofile import Carofile

# --------------------------------------------------------------------------- #
# -------------------- Command line arguments parsing ----------------------- #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(prog='Sleep Learning classification')
subparsers = parser.add_subparsers(help='supported file types', dest="filetype")

# create the parser for the "caro" command
parser_mat = subparsers.add_parser('caro',
                                   help='for a single .mat files including all PSGs')
parser_mat.add_argument("--i", dest="fn", default=os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '..')) + '/data/raw/caroline/WESA_D01_20171121ff_MLready.mat',
                        help="filepath of carofile")

# create the parser for the "physio" command
parser_b = subparsers.add_parser('physio',
                                 help='physionet files are not yet supported')

# parse some argument lists
args = vars(parser.parse_args())

# --------------------------------------------------------------------------- #
# ----------------- Load file and do basic preprocessing --------------------- #
# --------------------------------------------------------------------------- #
abs_path = ROOT_DIR + '/' + args['fn']
if not os.path.isfile(abs_path):
    print(abs_path + " DOES NOT EXIST!")
elif args['filetype'] == 'caro':
    sl = Carofile(abs_path, verbose=True)
elif args['filetype'] == 'physio':
    print("not yet supported")

# --------------------------------------------------------------------------- #
# ----------------- Perform classification procedure ------------------------ #
# --------------------------------------------------------------------------- #
