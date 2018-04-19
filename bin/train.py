import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)

from sleeplearning.lib.utils import SleepLearningDataset
from sleeplearning.lib.model import Net

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SleepLearning')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5 * 1e-5, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = DataLoader(
    SleepLearningDataset('caroline_cut/train'),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(SleepLearningDataset('caroline_cut/test'),
                         batch_size=args.batch_size, shuffle=True, **kwargs)

model = Net()
if args.cuda:
    model.cuda()

# create a stochastic gradient descent optimizer
# optimizer = optim.SGD(model.parameters(), lr=args.lr,
# momentum=args.momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# create a loss function
criterion = F.nll_loss


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: '
                '({:.0f}%)'.format(epoch, batch_idx * len(data),
                                   len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader),
                                   loss.data[0],
                                   100. * correct / args.batch_size))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(
            target).long()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[
            0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


print('Start training ...')
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
