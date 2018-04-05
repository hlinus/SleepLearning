"""Script for classifying sleep phases"""
###############################################################################
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from sleeplearning.lib.base import SleepLearning


# --------------------------------------------------------------------------- #
# -------------------- Command line arguments parsing ----------------------- #
# --------------------------------------------------------------------------- #
import argparse

parser = argparse.ArgumentParser(prog='classify.py')
subparsers = parser.add_subparsers(help='supported file types')

# create the parser for the "a" command
parser_mat = subparsers.add_parser('mat',
                                   help='for a single .mat files including all PSGs')
parser_mat.add_argument("--i", dest="fn", default=os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')) + \
                                                  '/data/caroline/WESA_D01_20171121ff_MLready.mat',
                        help="filepath of mat file")
parser_mat.add_argument("--id", default="-", type=str,
                        help="identifier for subject (useful for plotting)")
parser_mat.add_argument("--fs", default="250", type=int,
                        help="sampling rate (Hz)")
parser_mat.add_argument("--epsize", default="20", type=int,
                        help="epoch size of sleep phase")
parser_mat.add_argument("--hyp", default="sleepStage_score", type=str,
                        help="variable name where the hypnogram is stored")
parser_mat.add_argument('--psgs', metavar='N', type=str, nargs='+', default=(
    'EEG', 'EEG_data_filt', 'EOGR', 'EOGR_data_filt', 'EOGL', 'EOGL_data_filt',
    'EMG', 'EMG_data_filt'),
                        help='list of mappings between the PSG variables to load in the form ID VARIABLE_IN_MAT_FILE e.g.: EEG EEG_data_filt')

# create the parser for the "EDF" command
parser_b = subparsers.add_parser('EDF', help='EDF files are not yet supported')

# parse some argument lists

args = vars(parser.parse_args())

# --------------------------------------------------------------------------- #
# ----------------- Load file and do basic preprocessing --------------------- #
# --------------------------------------------------------------------------- #
psg_list = list(args['psgs'])
psg_dict = {}
for i in range(0, len(psg_list), 2):
    psg_dict[psg_list[i]] = psg_list[i + 1]


sl = SleepLearning._read_mat(args['id'], args['fn'], psg_dict, args['hyp'],
                             epoch_size=args['epsize'],
                             sampling_rate=args['fs'])

# print(ms_exhalome.avg_intensity_)


# --------------------------------------------------------------------------- #
# ----------------- Perform classification procedure ------------------------ #
# --------------------------------------------------------------------------- #
