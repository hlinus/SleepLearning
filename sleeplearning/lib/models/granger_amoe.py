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


client = MongoClient(mongo_url)
db = client.sacred
# db.collection_names(include_system_collections=False)
files = db.fs.files


def restore_clf_from_runid(db, id: int):
    import tempfile
    fs = gridfs.GridFS(db)
    model_object = files.find_one({"filename": "artifact://runs/"+str(
        id)+"/checkpoint.pth.tar"})
    myfile = fs.get(model_object['_id'])
    temp_path = os.environ.get('TMPDIR') if 'TMPDIR' in os.environ else \
        tempfile.mkdtemp()
    model_path = os.path.join(temp_path, 'tmp_model')
    with open(model_path, 'wb') as f:
        f.write(myfile.read())
    clf = sleeplearning.lib.base.Base()
    clf.restore(model_path)
    return clf


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


class GrangerAmoe(nn.Module):
    def __init__(self, ms: dict):
        super(GrangerAmoe, self).__init__()
        self.dropout = ms['dropout']
        self.num_classes = ms['nclasses']

        if 'expert_ids' in ms.keys():
            experts = []
            for id in ms['expert_ids']:
                clf = restore_clf_from_runid(db, id)
                for param in clf.model.parameters():
                    param.requires_grad = False

                experts.append(clf)
            ms['experts'] = experts

        emb_dim = list(experts[0].model.children())[-1].in_features
        self.nchannels = ms['input_dim'][0]
        assert (self.nchannels == len(experts))

        self.experts_emb = [e.model for e in ms['experts']]
        self.experts_emb = [nn.Sequential(*list(expert.children())[:-1]) for
                            expert in self.experts_emb]
        self.experts_emb = nn.ModuleList(self.experts_emb).eval()

        self.emb_transf_net = [nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=True),
                nn.ReLU(),
            ) for _ in range(len(self.experts_emb))]
        self.emb_transf_net = nn.ModuleList(self.emb_transf_net)

        # transforms
        self.u = nn.ModuleList([nn.Sequential(
                nn.Linear(emb_dim * self.nchannels, emb_dim, bias=True),
                nn.ReLU(),
            ) for _ in range(len(self.experts_emb))])

        # initialize the attention vectors
        self.ue = nn.Parameter(torch.zeros(self.nchannels, emb_dim))
        torch.nn.init.xavier_uniform_(self.ue)

        self.fcn = nn.Sequential(
            nn.Linear(self.nchannels * emb_dim, self.nchannels * emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            # nn.Linear(num_channels * emb_dim // 2, num_channels * emb_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(num_channels * emb_dim // 2, num_channels * emb_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            nn.Linear(self.nchannels * emb_dim // 2, self.num_classes),
        )

        self.paux_i = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim * (self.nchannels - 1), self.num_classes,
                                 bias=True),
            nn.Softmax(dim=1)
        ) for _ in range(len(self.experts_emb))])

        self.indices = [torch.Tensor([i for i in range(self.nchannels) if i !=
                                     j]).long() for j in range(self.nchannels)]

        self.paux_c = nn.Sequential(
            nn.Linear(emb_dim * self.nchannels, self.num_classes, bias=True),
            nn.Softmax(dim=1)
        )

        del ms['experts']

        # self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def train(self, mode=True):
        super(GrangerAmoe, self).train(mode=mode)
        self.experts_emb.eval()  # keep the embedding models in eval mode

    def forward(self, x):
        #print("ue type:", self.ue.dtype)

        emb = [exp(torch.unsqueeze(channel, 1).contiguous())
             for (exp, channel) in zip(self.experts_emb, torch.unbind(x, 1))]

        # hidden states of experts
        # [bs, nchannels, emb_size]
        h = torch.stack(emb, dim=1)
        # [bs, nchannels*embsize]
        h = h.view(h.size(0), -1)

        # transformed embeddings
        u = [ui(h) for ui in self.u]
        # [bs, nchannels, emb_size]
        u = torch.stack(u, dim=1)


        # [bs, nchannels, emb_size]
        transf_emb = [t(emb.contiguous()) for (t, emb) in zip(
            self.emb_transf_net, emb)]
        transf_emb = torch.stack(transf_emb, dim=1)

        # [bs, nchannels, emb_size]
        weights = torch.add(u, self.ue)
        # [bs, nchannels, 1]
        weights = torch.sum(weights, 2)
        # normalize weights
        # [bs, nchannels]
        weights = F.softmax(weights, 1)
        # [bs, emb_size, nchannels]
        weighted_emb = torch.mul(transf_emb.permute(0, 2, 1), weights.unsqueeze(1))
        # [bs, emb_size*nchannels]
        weighted_emb = weighted_emb.view(weighted_emb.size(0), -1)
        # [bs, nclasses] (prediction with weighted hidden states of experts)
        y_att = self.fcn(weighted_emb)
        # [bs, nclasses] (prediction with unweighted hidden states of experts)
        yaux_c = self.paux_c(h)

        # [bs, nchannels, emb_size]
        h = h.view(h.size(0), self.nchannels, -1)
        # List[bs, (nchannels-1), emb_size]
        hc_wo_hi = [torch.index_select(h, 1, ind) for ind in self.indices]
        # List[bs, nchannels]
        yaux_i = [paux_i_(hci.view(hci.size(0), -1)) for (paux_i_, hci) in zip(self.paux_i, hc_wo_hi)]

        return yaux_i, yaux_c, y_att, weights
