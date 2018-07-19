import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from torch.nn.init import xavier_normal_ as xavier_normal
from sleeplearning.lib.models.deep_sleep_net import DeepFeatureNet_, \
    Conv1dWithBn


class Conv2dWithBn(nn.Module):
    def __init__(self, input_shape, filter_size, n_filters, stride, wd=0):
        super(Conv2dWithBn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, n_filters, kernel_size=filter_size,
                               stride=stride, bias=False,
                               padding=((filter_size[0] - 1) // 2, (filter_size[
                                                                        1] - 1) // 2))
        # fake 'SAME'
        self.relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(n_filters)
        # self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_bn(x)
        return x


class MultivariateNet2d(nn.Module):
    def __init__(self, ts: dict):
        super(MultivariateNet2d, self).__init__()
        self.dropout = ts['dropout']
        input_shape = ts['input_dim']
        self.init_bn = nn.BatchNorm2d(input_shape[0])
        self.init_drop = nn.Dropout(p=.1)

        def conv_per_channel():
            return nn.Sequential(
                nn.MaxPool2d((2, 1), stride=(2, 1)),
                Conv2dWithBn(1, filter_size=(3, 3), n_filters=32, stride=1),
                nn.MaxPool2d((2, input_shape[2]//150+1), stride=(2, input_shape[2]//150+1)),

                Conv2dWithBn(32, filter_size=(3, 3), n_filters=48, stride=1),
                nn.MaxPool2d((2, input_shape[2]//210+1), stride=(2, input_shape[2]//210+1)),

                Conv2dWithBn(48, filter_size=(3, 3), n_filters=48, stride=1),
                nn.MaxPool2d((2, input_shape[2]//300+2),
                             stride=(2, input_shape[2]//300+2)),
            )

        self.num_classes = ts['nclasses']

        self.dfns = ModuleList([conv_per_channel() for _ in range(input_shape[
                                                                      0])])
        self.Dropout = nn.Dropout(p=self.dropout)

        in_dim = self._get_output_dim((1, *input_shape[1:])) * input_shape[0]
        last_drop = 0
        self.fcn = nn.ModuleList()
        if ts['fc_d']:
            out_dim = ts['fc_d'][0][0]
            self.fcn.append(nn.Linear(in_dim, out_dim,
                                      bias=True))
            self.fcn.append(nn.ReLU())
            self.fcn.append(nn.Dropout(p=self.dropout))
            for k in range(len(ts['fc_d']) - 1):
                in_dim = ts['fc_d'][k][0]
                out_dim = ts['fc_d'][k + 1][0]
                dropout = ts['fc_d'][k][1]
                self.fcn.append(nn.Linear(in_dim, out_dim,
                                          bias=True))
                self.fcn.append(nn.ReLU())
                self.fcn.append(nn.Dropout(p=dropout))
            in_dim = ts['fc_d'][-1][0]
            last_drop = ts['fc_d'][-1][1]

        self.fcn.append(nn.Linear(in_dim, self.num_classes,
                                  bias=True))
        self.fcn.append(nn.Dropout(p=last_drop))

        # self.weights_init()

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
        x = self.init_bn(x)
        x = self.init_drop(x)
        x = [dfn(torch.unsqueeze(channel, 1)) for (dfn, channel) in
             zip(self.dfns, torch.unbind(x, 1))]
        x = [y.view(y.size(0), 1, -1) for y in x]
        x = torch.cat(x, 2)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        for i in range(len(self.fcn)):
            x = self.fcn[i](x)
        return x
