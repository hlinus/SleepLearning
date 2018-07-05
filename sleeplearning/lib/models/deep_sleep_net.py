import torch
from torch import nn
from torch.nn.init import xavier_normal
import torch.nn.functional as F


class Conv1dWithBn(nn.Module):
    def __init__(self, input_shape, filter_size, n_filters, stride, wd=0):
        super(Conv1dWithBn, self).__init__()
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


class DeepFeatureNet_(nn.Module):
    def __init__(self, ts: dict):
        dropout = ts['dropout']
        input_shape = ts['input_dim']
        super(DeepFeatureNet_, self).__init__()
        self.dropout = dropout
        # left side
        self.left = nn.Sequential(
            Conv1dWithBn(input_shape[0], filter_size=50, n_filters=64, stride=6),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(p=self.dropout),
            Conv1dWithBn(64, filter_size=8,
                         n_filters=128, stride=1),
            Conv1dWithBn(128, filter_size=8,
                         n_filters=128, stride=1),
            Conv1dWithBn(128, filter_size=8,
                         n_filters=128, stride=1),
            nn.MaxPool1d(4, stride=4)
        )

        self.right = nn.Sequential(
            Conv1dWithBn(input_shape[0], filter_size=400, n_filters=64, stride=50),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(p=self.dropout),
            Conv1dWithBn(64, filter_size=6,
                         n_filters=128, stride=1),
            Conv1dWithBn(128, filter_size=6,
                         n_filters=128, stride=1),
            Conv1dWithBn(128, filter_size=6,
                         n_filters=128, stride=1),
            nn.MaxPool1d(2, stride=2)
        )

        self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        y = [left.view(left.size(0), -1), right.view(right.size(0), -1)]
        x = torch.cat(y, 1)
        x = F.dropout(x, p=self.dropout)
        return x

class DeepFeatureNet(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple, dropout: float):
        super(DeepFeatureNet, self).__init__()
        self.DeepFeatureNet_ = DeepFeatureNet_(input_shape, dropout)
        n_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(n_size, num_classes)
        #self.fc1 = nn.Linear(2560, num_classes)
        self.num_classes = num_classes
        self.weights_init()

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self.DeepFeatureNet_(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        x = self.DeepFeatureNet_(x)
        x = self.fc1(x)
        return x


class DeepSleepNet(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple, dropout: float):
        super(DeepSleepNet, self).__init__()
        self.DeepFeatureNet_ = DeepFeatureNet_(input_shape, dropout)
        self.left = nn.LSTM(2560, 512, 2,
                         batch_first=True, bidirectional=True, dropout=dropout)
        self.right = nn.Linear(2560, 1024)

        self.fc = nn.Linear(1024, 5, bias=False)
        self.num_classes = num_classes
        self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        x = F.dropout(x, p=.5)
        x = self.DeepFeatureNet_(x)
        right = self.right(x) # (batch x seqlen, hidden_size*2)
        # reshape to (batch, seqlen, feature)
        x = torch.reshape(x, (-1, 25, 2560))
        left, _ = self.left(x)   # (batch_size, seq_length, hidden_size*2)
        left = left.reshape((-1, 1024)) # (batch x seqlen, hidden_size*2)

        x = left + right # (batch x seqlen, hidden_size*2)
        x = F.dropout(x, p=.5)
        x = self.fc(x)
        return x