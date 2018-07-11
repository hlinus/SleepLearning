import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from torch.nn.init import xavier_normal
from sleeplearning.lib.models.deep_sleep_net import DeepFeatureNet_, Conv1dWithBn


class MultivariateNet(nn.Module):
    def __init__(self, ts: dict):
        super(MultivariateNet, self).__init__()
        self.dropout = ts['dropout']
        input_shape = ts['input_dim']
        conv_per_channel = nn.Sequential(
            Conv1dWithBn(1, filter_size=50, n_filters=64, stride=6),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(p=self.dropout),
            Conv1dWithBn(64, filter_size=8,
                         n_filters=128, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(p=self.dropout),
            Conv1dWithBn(128, filter_size=8,
                         n_filters=128, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(p=self.dropout),
            Conv1dWithBn(128, filter_size=8,
                         n_filters=128, stride=1),
        )
        self.num_classes = ts['nclasses']
        self.dfns = ModuleList([conv_per_channel for _ in range(input_shape[0])])
        output_dim = self._get_output_dim((1, input_shape[1]))
        self.adaptive_maxp = nn.AdaptiveMaxPool1d(4096)
        self.fcn = nn.ModuleList()
        for k in range(len(ts['fc_d'])-1):
            in_dim = ts['fc_d'][k][0]
            out_dim = ts['fc_d'][k+1][0]
            dropout = ts['fc_d'][k][1]
            self.fcn.append(nn.Linear(in_dim, out_dim,
                              bias=False))
            self.fcn.append(nn.ReLU())
            self.fcn.append(nn.Dropout(p=dropout))
        self.fcn.append(nn.Linear(ts['fc_d'][-1][0], self.num_classes,
                                  bias=False))
        self.fcn.append(nn.ReLU())
        self.fcn.append(nn.Dropout(p=ts['fc_d'][-1][1]))

        self.weights_init()

    # generate input sample and forward to get shape
    def _get_output_dim(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self.dfns[0](input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout)
        x = [dfn(torch.unsqueeze(channel, 1)) for (dfn, channel) in zip(self.dfns, torch.unbind(x, 1))]
        x = [y.view(y.size(0), 1, -1) for y in x]
        x = torch.cat(x, 2)
        x = self.adaptive_maxp(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=.5)
        for i in range(len(self.fcn)):
            x = self.fcn[i](x)
        return x