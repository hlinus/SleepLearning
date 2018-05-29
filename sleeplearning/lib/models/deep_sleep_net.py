import torch

from torch import nn
from torch.nn.init import xavier_normal
import torch.nn.functional as F


class Conv1dLayer(nn.Module):
    def __init__(self, input_shape, filter_size, n_filters, stride, wd=0):
        super(Conv1dLayer, self).__init__()
        self.conv1 = nn.Conv1d(input_shape, n_filters, kernel_size=filter_size,
                               stride=stride, bias=False,
                               padding=(filter_size - 1) // 2) # fake 'SAME'
        self.conv1_bn = nn.BatchNorm1d(n_filters)

        self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        return x


class DeepSleepNet(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super(DeepSleepNet, self).__init__()
        # left side
        self.left = nn.Sequential(
            Conv1dLayer(input_shape[0], filter_size=50, n_filters=64, stride=6),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(p=.5),
            Conv1dLayer(64, filter_size=8,
                        n_filters=128, stride=1),
            Conv1dLayer(128, filter_size=8,
                        n_filters=128, stride=1),
            Conv1dLayer(128, filter_size=8,
                        n_filters=128, stride=1),
            nn.MaxPool1d(4, stride=4)
        )

        self.right = nn.Sequential(
            Conv1dLayer(input_shape[0], filter_size=400, n_filters=64, stride=50),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(p=.5),
            Conv1dLayer(64, filter_size=6,
                        n_filters=128, stride=1),
            Conv1dLayer(128, filter_size=6,
                        n_filters=128, stride=1),
            Conv1dLayer(128, filter_size=6,
                        n_filters=128, stride=1),
            nn.MaxPool1d(2, stride=2)
        )

        self.fc1 = nn.Linear(2560, num_classes)

        self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x) #self.right(x)
        y = [left.view(left.size(0), -1), right.view(right.size(0), -1)]
        x = torch.cat(y, 1)

        x = F.dropout(x, p=.5)

        #x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x