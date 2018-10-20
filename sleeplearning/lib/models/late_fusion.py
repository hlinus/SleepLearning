import copy
import os
import sys

import gridfs
from torch.nn import ModuleList

from sleeplearning.lib.models.single_chan_expert import SingleChanExpert
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from pymongo import MongoClient
from torch import nn
from torch.nn.init import xavier_normal_ as xavier_normal
from cfg.config import mongo_url
import torch
import torch.nn.functional as F
import sleeplearning.lib.base


class Conv2dWithBn(nn.Module):
    def __init__(self, input_shape, filter_size, n_filters, stride, bn=True):
        super(Conv2dWithBn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, n_filters, kernel_size=filter_size,
                               stride=stride, bias=False,
                               padding=((filter_size[0] - 1) // 2, (filter_size[
                                                                        1] - 1) // 2))
        # fake 'SAME'
        self.relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(n_filters) if bn else None
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
        if self.conv1_bn is not None:
            x = self.conv1_bn(x)
        return x

# generate input sample and forward to get shape
def _get_output_dim(net, shape):
    bs = 1
    input = torch.rand(bs, *shape)
    output_feat = net(input)
    return output_feat.shape


class FeatureExtractorCurrentEpoch(nn.Module):
    def __init__(self, input_shape, bn=True):
        super(FeatureExtractorCurrentEpoch, self).__init__()
        self.input_shape = input_shape
        neighbors = input_shape[2] // 30 -1
        self.block = nn.Sequential(
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                Conv2dWithBn(input_shape[0], filter_size=(3, 3), n_filters=64,
                             stride=1),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                #
                Conv2dWithBn(64, filter_size=(3, 3), n_filters=64, stride=1),
                nn.MaxPool2d((3, 3), stride=(3, 3)),
                #
                Conv2dWithBn(64, filter_size=(3, 3), n_filters=96, stride=1),
                nn.MaxPool2d((3, 2), stride=(3, 2)),
            )

        outdim = _get_output_dim(self.block, input_shape)
        self.fc = nn.Linear(outdim[1]*outdim[2]*outdim[3], 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        #self.bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FECurrentEpochChannel(nn.Module):
    def __init__(self, input_shape, bn=True):
        super(FECurrentEpochChannel, self).__init__()
        self.input_shape = input_shape
        neighbors = input_shape[2] // 30 -1
        def get_conv_block():
            return nn.Sequential(
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                Conv2dWithBn(1, filter_size=(3, 3), n_filters=64,
                             stride=1),
                nn.MaxPool2d((2, 2), stride=(2, 2)),
                #
                Conv2dWithBn(64, filter_size=(3, 3), n_filters=64, stride=1),
                nn.MaxPool2d((3, 3), stride=(3, 3)),
                #
                Conv2dWithBn(64, filter_size=(3, 3), n_filters=96, stride=1),
                nn.MaxPool2d((3, 2), stride=(3, 2)),
            )
        feature_extractor = [get_conv_block() for _ in range(input_shape[0])]
        self.dfns = ModuleList(feature_extractor)

        outdim = _get_output_dim(get_conv_block(), (1, input_shape[1], input_shape[
            2]))
        self.fc = nn.Linear(outdim[1]*outdim[2]*outdim[3]*input_shape[0], 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        z = [dfn(torch.unsqueeze(channel, 1)) for (dfn, channel) in
             zip(self.dfns, torch.unbind(x, 1))]
        z = [y.view(y.size(0), 1, -1) for y in z]

        x = torch.cat(z, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LateFusion(nn.Module):
    def __init__(self, ms: dict):
        super(LateFusion, self).__init__()
        self.dropout = ms['dropout']
        num_classes = ms['nclasses']
        self.expert_channels = []
        self.train_emb = ms['train_emb']
        self.attention = True if 'attention' not in ms.keys() or ms[
            'attention'] else False

        if 'expert_models' in ms.keys():
            experts = []
            for path in ms['expert_models']:
                clf = sleeplearning.lib.base.Base()
                clf.restore(path)
                self.expert_channels.append(clf.ds['channels'][0][0])
                for param in clf.model.parameters():
                    param.requires_grad = False
                sorted(zip(self.expert_channels, experts),
                       key=lambda pair: pair[0])
                experts.append(clf)
            self.expert_channels, experts = list(zip(*sorted(zip(self.expert_channels, experts),
                                    key=lambda pair: pair[0])))

        self.experts_emb = [copy.deepcopy(e.model) for e in experts]
        self.experts_emb = [nn.Sequential(*list(expert.children())[:-1]) for
                            expert in self.experts_emb]
        self.experts_emb = nn.ModuleList(self.experts_emb)
        if self.train_emb:
            for emb in self.experts_emb:
                for param in emb.parameters():
                    param.requires_grad = True

        C, F = ms['input_dim'][0], ms['input_dim'][1]

        num_channels = ms['input_dim'][0]
        self.fes = [FeatureExtractorCurrentEpoch((1, F, 30)) for _ in
                    range(num_channels)]
        self.fes = nn.ModuleList(self.fes)

        emb_dim = list(experts[0].model.children())[-1].in_features
        assert (num_channels == len(self.experts_emb))

        input_dim_classifier = emb_dim*(num_channels+1)

        self.classifier = nn.Sequential(
             nn.Dropout(p=self.dropout),
             nn.Linear(input_dim_classifier, input_dim_classifier // 2),
             nn.ReLU(),
             nn.Dropout(p=self.dropout),
             nn.Linear(input_dim_classifier // 2, num_classes),
        )

    def train(self, mode=True):
        super(LateFusion, self).train(mode=mode)

    def forward(self, x):
        emb = [exp(torch.unsqueeze(channel, 1).contiguous())
             for (exp, channel) in zip(self.experts_emb, torch.unbind(x, 1))]

        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3)//30, -1).permute(0,3,1,2,4)
        x = x[:, x.size(1)//2, :]
        x = [torch.unsqueeze(channel, 1) for channel in
             torch.unbind(x, 1)]
        h = [fe(channel) for (fe, channel) in zip(self.fes,x)]
        h = torch.stack(h, dim=1)
        h = torch.sum(h, dim=1)
        emb = torch.cat(emb, 1)
        concat = torch.cat((emb, h), 1)

        logits = self.classifier(concat)

        result = {'logits': logits}
        return result
