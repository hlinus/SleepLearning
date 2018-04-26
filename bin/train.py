import sys
import argparse
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.base import SleepLearning

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SleepLearning')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5 * 1e-5, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-weight-loss', action='store_true', default=False,
                    help='disables class weighted loss')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


args = parser.parse_args()
SleepLearning(not args.no_cuda).train(args)


