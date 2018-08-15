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
        #self.weights_init()

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


class ConvBlock(nn.Module):
    def __init__(self, input_shape):
        super(ConvBlock, self).__init__()
        self.input_shape = input_shape
        self.block = nn.Sequential(
                nn.MaxPool2d((2, 1), stride=(2, 1)),
                Conv2dWithBn(1, filter_size=(3, 3), n_filters=128, stride=1),
                nn.MaxPool2d((2, self.input_shape[2]//150+1), stride=(2,
                                                                      self.input_shape[2]//150+1)),

                Conv2dWithBn(128, filter_size=(3, 3), n_filters=64, stride=1),
                nn.MaxPool2d((2, self.input_shape[2]//210+1), stride=(2, self.input_shape[2]//210+1)),

                Conv2dWithBn(64, filter_size=(3, 3), n_filters=12, stride=1),
                nn.MaxPool2d((2, self.input_shape[2]//300+2),
                             stride=(2, self.input_shape[2]//300+2)),
            )

    def forward(self, x):
        x = self.block(x)
        return x.view(x.size(0), -1)


class SingleChanExpert(nn.Module):
    def __init__(self, ts: dict):
        super(SingleChanExpert, self).__init__()
        self.dropout = ts['dropout']
        self.input_dim = ts['input_dim']
        self.num_classes = ts['nclasses']

        self.conv_block = ConvBlock(self.input_dim)

        self.Dropout = nn.Dropout(p=self.dropout)

        in_dim = self._get_output_dim((1, *self.input_dim[1:])) * self.input_dim[0]
        last_drop = 0
        self.fcn = nn.Sequential(
            nn.Linear(in_dim, 128, bias=True)
        )
        self.last_fc = nn.Linear(128, self.num_classes, bias=True)

        #self.weights_init()

    # generate input sample and forward to get shape
    def _get_output_dim(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self.conv_block(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        y = self.conv_block(x)
        y = self.Dropout(y)
        y = self.fcn(y)
        y = self.last_fc(y)
        output = {'logits': y}
        return output
