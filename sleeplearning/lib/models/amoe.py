import copy
import os
import sys

import gridfs

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
    def __init__(self, input_shape, filter_size, n_filters, stride, xavier):
        super(Conv2dWithBn, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, n_filters, kernel_size=filter_size,
                               stride=stride, bias=False,
                               padding=((filter_size[0] - 1) // 2, (filter_size[
                                                                        1] - 1) // 2))
        # fake 'SAME'
        self.relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(n_filters)
        if xavier:
            self.weights_init()

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


class Amoe(nn.Module):
    def __init__(self, ms: dict):
        super(Amoe, self).__init__()
        self.dropout = ms['dropout']
        num_classes = ms['nclasses']
        self.expert_channels = []
        self.train_emb = ms['train_emb']

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

        self.expert_y = [copy.deepcopy(e.model) for e in experts]
        self.expert_y = nn.ModuleList(self.expert_y)

        num_channels = ms['input_dim'][0]
        emb_dim = list(experts[0].model.children())[-1].in_features
        assert (num_channels == len(self.experts_emb))

        self.attention_net = nn.Sequential(
             nn.Dropout(p=self.dropout),
             nn.Linear(num_channels * (emb_dim+num_classes), num_channels *
                       emb_dim //
                                       4),
             nn.ReLU(),
             #nn.Dropout(p=self.dropout),
             nn.Linear(num_channels * emb_dim // 4, num_channels),
             nn.Softmax(dim=1)
        )

    def train(self, mode=True):
        super(Amoe, self).train(mode=mode)
        #if not self.train_emb:
        #    self.experts_emb.eval()  # keep the embedding models in eval mode
        #self.expert_y.eval()

    def forward(self, x):
        emb = [exp(torch.unsqueeze(channel, 1).contiguous())
             for (exp, channel) in zip(self.experts_emb, torch.unbind(x, 1))]

        # [nchannels, bs, nclasses]
        expert_y = [exp(torch.unsqueeze(channel, 1).contiguous())['logits']
             for (exp, channel) in zip(self.expert_y, torch.unbind(x, 1))]
        expert_y = torch.stack([F.softmax(logits, dim=1) for logits in
                                expert_y])
        # [bs, nchannel, embdim ]
        emb = torch.stack(emb, dim=1)

        # [bs, nchannel, (embdim+nclasses) ]
        emb_softmax = torch.cat((emb, expert_y.permute(1, 0, 2)), dim=2)

        # [bs, nchannels * (embdim+nclasses)]
        emb_softmax = emb_softmax.view(emb_softmax.size(0), -1)

        # [bs, nchannels]
        weights = self.attention_net(emb_softmax)

        # [bs, nchannels, nclasses]
        expert_y = expert_y.permute(1, 0, 2)
        # [bs, nchannels, nclasses]
        weighted_y = torch.mul(expert_y, weights.unsqueeze(2))

        # [bs, nclasses]
        y = torch.sum(weighted_y, dim=1)
        logits = y.clamp(min=1e-7).log()#.

        return {'logits': logits, 'y_experts': expert_y, 'a': weights}
