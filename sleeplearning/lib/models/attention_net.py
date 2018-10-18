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
    #n_size = output_feat.data.view(bs, -1).size(1)
    #return n_size
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


class AttentionWeightNet(nn.Module):
    def __init__(self, encoder_dim = 128, decoder_dim = 128):
        super(AttentionWeightNet, self).__init__()
        self.W1 = nn.Parameter(torch.Tensor(encoder_dim, encoder_dim))
        self.W2 = nn.Parameter(torch.Tensor(encoder_dim, decoder_dim))
        self.v = nn.Parameter(torch.Tensor(1, encoder_dim))
        self.tanh = nn.Tanh()
        xavier_normal(self.W1)
        xavier_normal(self.W2)
        xavier_normal(self.v)

    def forward(self, h, e_s):
        act = self.tanh(torch.matmul(e_s.unsqueeze(1), self.W1)
                  + torch.matmul(self.W2, h.unsqueeze(2)).permute(0,2,1))
        x = torch.sum(self.v.squeeze() * act.squeeze(1), dim=1).unsqueeze(1)
        return x


class AttentionVasvani(nn.Module):
    def __init__(self, encoder_dim = 128, decoder_dim = 128):
        super(AttentionVasvani, self).__init__()

    def forward(self, k, q):
        x = torch.sum(k*q, dim=1, keepdim=True)
        x /= torch.sqrt(torch.norm(k, p=1, dim=1, keepdim=True))
        return x


class AttentionWeightNetFC(nn.Module):
    def __init__(self, encoder_dim = 128, decoder_dim = 128):
        super(AttentionWeightNetFC, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)

    def forward(self, h, e_s):
        x = torch.cat((h, e_s), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class AttentionVectorNet(nn.Module):
    def __init__(self, encoder_dim = 128, decoder_dim = 128):
        super(AttentionVectorNet, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)

    def forward(self, h, e_s):
        x = torch.cat((h, e_s), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class AttentionNet(nn.Module):
    def __init__(self, ms: dict):
        super(AttentionNet, self).__init__()
        self.dropout = ms['dropout']
        num_classes = ms['nclasses']
        self.expert_channels = []
        self.train_emb = ms['train_emb']
        self.attention = True if 'attention' not in ms.keys() or ms[
            'attention'] else False
        #self.normalize_context = ms['normalize_context']

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

        C, F = ms['input_dim'][0], ms['input_dim'][1]

        num_channels = ms['input_dim'][0]
        #self.fes = [FeatureExtractorCurrentEpoch((1, F, 30)) for _ in
        #            range(num_channels)]
        #self.fes = nn.ModuleList(self.fes)
        self.ht = FeatureExtractorCurrentEpoch((num_channels, F, 30))

        emb_dim = list(experts[0].model.children())[-1].in_features
        assert (num_channels == len(self.experts_emb))

        self.attention_weights = AttentionWeightNet()

        self.attention_vector = nn.Sequential(
                                    nn.Linear(2 * (emb_dim),
                                    emb_dim),
                                    nn.ReLU(),
                                    nn.Linear((emb_dim),
                                              emb_dim),
                                    nn.Tanh(),
                                    #nn.ReLU()
                                ) if self.attention else None

        self.transnet = [nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(emb_dim, emb_dim),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(emb_dim, emb_dim),
            # #nn.Dropout(p=self.dropout),
            # #nn.Tanh(),
        )
            for _ in range(len(experts))]
        self.transnet = nn.ModuleList(self.transnet) if self.attention \
            else None

        input_dim_classifier = emb_dim if self.attention else emb_dim

        self.classifier = nn.Sequential(
             nn.Dropout(p=self.dropout),
             #nn.Linear(input_dim_classifier, input_dim_classifier),
             #nn.ReLU(),
             #nn.Dropout(p=self.dropout),
             nn.Linear(input_dim_classifier, emb_dim // 2),
             nn.ReLU(),
             nn.Dropout(p=self.dropout),
             nn.Linear(emb_dim // 2, num_classes),
        )

    def train(self, mode=True):
        super(AttentionNet, self).train(mode=mode)
        #if not self.train_emb:
        #    self.experts_emb.eval()  # keep the embedding models in eval mode
        self.expert_y.eval()

    def forward(self, x):
        emb = [exp(torch.unsqueeze(channel, 1).contiguous())
             for (exp, channel) in zip(self.experts_emb, torch.unbind(x, 1))]

        # [nchannels, bs, nclasses]
        expert_y = [exp(torch.unsqueeze(channel, 1).contiguous())['logits']
             for (exp, channel) in zip(self.expert_y, torch.unbind(x, 1))]
        expert_y = torch.stack([F.softmax(logits, dim=1) for logits in
                                expert_y])
        # [bs, nchannels, nclasses]
        expert_y = expert_y.permute(1, 0, 2)

        # [bs, nchannel, embdim ]
        #emb = torch.stack(emb, dim=1)
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3)//30,
         -1).permute(0,3,1,2,4)
        x = x[:, x.size(1)//2, :]
        #x = [torch.unsqueeze(channel, 1) for channel in
        #     torch.unbind(x, 1)]
        h = self.ht(x)

        # [bs, nchannels]
        if self.attention:
            # [bs, nchannels * (embdim)]
            temb = [t(e) for t, e in zip(self.transnet, emb)]
            a = [self.attention_weights(h, t_s) for t_s in temb]
            a = torch.stack(a, dim=1)
            a = F.softmax(a, dim=1)

            #a = torch.ones(a.shape).cuda() / 10 # uniform attention weights
            temb = torch.stack(temb, dim=1)
            # nomalize
            #if self.normalize_context:
            #    temb = temb / torch.sum(temb, dim=1, keepdim=True)
            c = torch.mul(temb, a)
            c = torch.sum(c, dim=1)
            attention_vector = self.attention_vector(torch.cat((c, h),
                                                               dim=1))
        else:
            attention_vector = h

        # [bs, nclasses]
        logits = self.classifier(attention_vector)

        if self.attention:
            result = {'logits': logits, 'y_experts': expert_y, 'a': a}
        else:
            result = {'logits': logits, 'y_experts': expert_y}
        return result
