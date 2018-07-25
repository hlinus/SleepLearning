import os
import sys

import gridfs

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
    model_object = files.find_one({"filename": "artifact://runs/"+str(id)+"/model_best.pth.tar"})
    myfile = fs.get(model_object['_id'])
    temp_path = tempfile.mkdtemp()
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


class ExpertsAvg(nn.Module):
    def __init__(self, ms: dict):
        super(ExpertsAvg, self).__init__()

        # TODO: FIX UGLY HACK
        if 'expert_ids' in ms.keys():
            experts = []
            for id in ms['expert_ids']:
                clf = restore_clf_from_runid(db, id)
                for param in clf.model.parameters():
                    param.requires_grad = False
                experts.append(clf)
            ms['experts'] = experts

        self.experts = nn.ModuleList([e.model for e in ms['experts']])
        self.exps = ms['experts']
        del ms['experts']

        # self.weights_init()

    def weights_init(m):
        for _, mi in m._modules.items():
            if isinstance(mi, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_normal(mi.weight.data)
                if mi.bias is not None:
                    xavier_normal(mi.bias.data)

    def forward(self, x):
        x = [F.softmax(exp(torch.unsqueeze(channel, 1).contiguous()), dim=1)
             for (exp, channel) in zip(self.experts, torch.unbind(x, 1))]
        x = torch.stack(x, dim=1)
        x = torch.sum(x, dim=1) / x.shape[1]
        return x
