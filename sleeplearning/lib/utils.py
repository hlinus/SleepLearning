import shutil
import os

import gridfs
import h5py as h5py
import torch
import inspect
import time
import pandas as pd
import re
import numpy as np
from pymongo import MongoClient
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, \
    SequentialSampler

from cfg.config import mongo_url
from sleeplearning.lib.loaders.baseloader import BaseLoader
from sleeplearning.lib.granger_loss import GrangerLoss
from sleeplearning.lib.models import *
from sleeplearning.lib.loaders import *


class SleepLearningDataset(object):
    """Sleep Learning dataset."""

    def __init__(self, data_dir: str, subject_csv: str, cvfold: int,
                 num_labels: int, feature_extractor,neighbors, loader,
                 discard_arts=True, transform=None, verbose=False):
        assert(neighbors % 2 == 0)

        if num_labels == 5:
            class_remapping = {0: 0, 1: 1, 2: 2, 3: 3,
                  4: 3, 5: 4}
        elif num_labels == 3:
            class_remapping = {0: 0, 1: 1, 2: 1,
                               3: 1,
                               4: 1, 5: 2}
        csv_file_prefix = os.path.basename(os.path.normpath(subject_csv))[:-4]
        tmp_dir = os.environ.get('TMPDIR') if 'TMPDIR' in os.environ else \
                      os.path.join(data_dir, 'tmp')
        self.dir = os.path.join(tmp_dir, csv_file_prefix)
        if os.path.exists(self.dir) and os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)
        if verbose:
            print("(", self.dir, ")\n")
        self.transform = transform
        self.X = []
        self.targets = []
        # processed dataset does not yet exist?
        subject_labels = []

        subject_files = pd.read_csv(subject_csv, header=None)[cvfold].dropna().tolist()
        class_distribution = np.zeros(
            len(BaseLoader.sleep_stages_labels.keys()), dtype=int)
        #subject_labels = []

        for subject_file in subject_files:
            start = time.time()
            subject = loader(os.path.join(data_dir, subject_file))
            subject_labels.append(subject.label)
            psgs_reshaped = {}
            num_epochs = len(subject.hypnogram)
            artefact_mask = np.zeros(num_epochs, dtype=bool)
            for k, psgs in subject.psgs.items():
                psgs1 = psgs.reshape(
                    (-1, subject.sampling_rate_ * subject.epoch_length))
                psgs_reshaped[k] = psgs1
                whole_epoch_zero = np.logical_and(psgs1.min(axis=1)==0, psgs1.max(axis=1)==0)
                artefact_mask = np.logical_or(artefact_mask,whole_epoch_zero)
            subject.hypnogram[artefact_mask] = 6
            # if spectogram used:
            # [num_epochs X num_channels X freq_domain X time_domain]
            # else
            # [num_epochs X num_channels X time_domain]
            feature_matrix = feature_extractor.fit_transform(psgs_reshaped)
            del psgs_reshaped
            #num_epochs = feature_matrix.shape[0]

            if neighbors > 0:
                # pad with zeros before and after (additional '#neighbors' epochs)
                pad_width = (
                    (neighbors // 2, neighbors // 2), (0, 0), (0, 0))
                concat_axis = 1
                if feature_matrix.ndim == 4:
                    pad_width += ((0, 0),)
                    concat_axis = 2
                feature_matrix = np.pad(feature_matrix, pad_width,
                                        mode='constant')
                # create samples with neighbors
                feature_matrix = np.array([np.concatenate(
                    feature_matrix[i - neighbors // 2:i + neighbors // 2 + 1],
                    axis=concat_axis) for i
                    in range(neighbors // 2, num_epochs + neighbors // 2)])
            num_artifacts = 0
            for e, (sample, label_int) in enumerate(
                    zip(feature_matrix, subject.hypnogram)):
                label = BaseLoader.sleep_stages_labels[label_int]
                if discard_arts and label == 'Artifact':
                    num_artifacts+=1
                    continue
                class_distribution[label_int] += 1
                id = subject.label + '_epoch_' + '{0:0>5}'.format(
                    e) + '_' + str(neighbors) + 'N_' + label

                label_int = class_remapping[label_int]
                self.targets.append(label_int)
                sample = {'id': id, 'x': sample, 'y': label_int}
                filename = os.path.join(self.dir, id+'.h5')
                #np.savez(filename, **sample)
                with h5py.File(filename, "w") as hf:
                    hf.create_dataset("id", data=sample['id'])
                    hf.create_dataset("x", data=sample['x'])
                    hf.create_dataset("y", data=sample['y'])
                self.X.append(filename)
            if verbose:
                print(f'loaded {subject_file} [{(time.time()-start):.2f}s,'
                      f' {num_artifacts} artifacts removed]')

        self.dataset_info = {}
        class_distribution_dict = {}
        for i in range(len(class_distribution)):
            if class_distribution[i] > 0:
                class_distribution_dict[BaseLoader.sleep_stages_labels[i]] = \
                    int(class_distribution[i])
        self.dataset_info['subjects'] = subject_labels
        self.dataset_info['class_distribution'] = class_distribution_dict
        self.dataset_info['input_shape'] = feature_matrix[0].shape

    def __getitem__(self, index):
        with h5py.File(self.X[index], "r") as hf:
            x = hf["x"].value
            y_ = hf["y"].value

        if self.transform is not None:
            x = self.transform(x)
        x = torch.from_numpy(x).float()

        return x, y_

    def __len__(self):
        return len(self.X)


def restore_model_from_mongodb(model_id: int, best: bool):
    client = MongoClient(mongo_url)
    db = client.sacred
    files = db.fs.files
    fs = gridfs.GridFS(db)
    filename = 'model_best.pth.tar' if best else 'checkpoint.pth.tar'
    model_object = files.find_one({"filename": "artifact://runs/"+str(
        model_id)+"/"+filename})
    model_path = os.path.abspath(os.path.join(os.path.dirname('__file__'),
                                            'models'))
    if model_object is None:
        raise ValueError("model does not exist")
    myfile = fs.get(model_object['_id'])

    model_path = os.path.join(model_path,
                              str(model_id)+'_'+filename)
    with open(model_path, 'wb') as f:
        f.write(myfile.read())


def get_sampler(ds: SleepLearningDataset,
                batch_size: int, oversample: bool, shuffle: bool,
                kwargs: dict, verbose: bool = True):

    assert(not (oversample and shuffle)) # can not oversample but not shuffle

    if oversample:
        class_count = np.fromiter(ds.dataset_info['class_distribution'].values(), dtype=np.float32)
        weight = 1. / class_count
        samples_weight = weight[ds.targets].astype(np.float32)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    elif shuffle:
        sampler = RandomSampler(ds)
    else:
        sampler = SequentialSampler(ds)

    dataloader = DataLoader(ds, batch_size=batch_size, sampler=sampler, **kwargs)

    if verbose:
        print('class distribution:', ds.dataset_info['class_distribution'])
        print('input shape:', ds.dataset_info['input_shape'])
        print('oversample:', oversample)
        print("num workers:", dataloader.num_workers, " pin_mem:", dataloader.pin_memory)

    return dataloader


def get_model_output(clf, data_dir, subject) -> Tuple[Dict, Dict]:
    import tempfile
    subject_path = os.path.join(data_dir, subject)
    subject_path = os.path.normpath(os.path.abspath(subject_path))
    data_dir, subject = os.path.split(subject_path)
    temp_path = tempfile.mkdtemp()
    tmp_csv = os.path.join(temp_path, 'tmp_csv')
    np.savetxt(tmp_csv, np.array([subject]), delimiter=",", fmt='%s')
    test_loader = clf.get_loader_from_csv_(data_dir, tmp_csv, 0, 100,
                                            False)
    output: dict = None
    metrics: dict = None
    clf.model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if clf.cudaEfficient:
                data, target = data.cuda(), target.cuda()
            # compute output
            batch_out, loss, metrics = clf.predict_batch_(data, target,
                                                           metrics)
            if output is None:
                output = {'y_true': [], 'y_probs': []}
                for k in batch_out.keys():
                    output[k] = []

            for k, v in batch_out.items():
                output[k].append(v.data.cpu().numpy())
            output['y_probs'].append(F.softmax(batch_out['logits'], dim=1)
                                     .data.cpu().numpy())
            output['y_true'].append(target.data.cpu().numpy())
    for k, v in output.items():
        output[k] = np.concatenate(output[k], axis=0)

    if hasattr(clf.model, 'expert_channels'):
        output['expert_channels'] = clf.model.expert_channels

    return output, metrics


def get_loader(s):
    ind = [i for i in range(len(s)) if str.isupper(s[i])]
    module_name = ''.join(
        [s[i] + '_' if (i + 1) in ind else str.lower(s[i])
         for i in range(len(s))])
    loader = eval(module_name + '.' + s)
    return loader


def get_model_arch(arch, ms):
    ind = [i for i in range(len(arch)) if str.isupper(arch[i])]
    module_name = ''.join(
        [arch[i] + '_' if (i + 1) in ind else str.lower(arch[i])
         for i in range(len(arch))])
    arch = eval(module_name + '.' + arch)(ms)
    return arch


def get_model(arch, ms, class_dist=None, cuda=True, verbose=False):
    optim_fn, optim_params = get_optimizer(ms['optim'])
    params = [p for p in arch.parameters() if p.requires_grad]

    optimizer = optim_fn(params, **optim_params) if params else None

    # TODO: refactor to get_loss(train_ds, weighted_loss: bool)
    if ms['weighted_loss']:
        # TODO: assure weights are in correct order
        counts = np.array(class_dist,
            dtype=float)
        normed_counts = counts / np.min(counts)
        weights = np.reciprocal(normed_counts).astype(np.float32)
    else:
        weights = np.ones(ms['nclasses'])

    weights = torch.from_numpy(weights).type(torch.FloatTensor)
    if 'loss' not in ms.keys() or ms['loss'] == 'xentropy':
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    elif ms['loss'] == 'granger':
        criterion = GrangerLoss(weights=weights, alpha=1)
    elif ms['loss'] == 'nlle':
        criterion = torch.nn.NLLLoss(weight=weights)
    else:
        raise ValueError(f"loss {ms['loss']} unknown. Please choose "
                         f"'xentropy' or 'granger'.")
    if verbose:
        print("\nCLASS WEIGHTS (LOSS): \n", weights)
        print('MODEL PARAMS:\n', ms)
        print('ARCH: \n', arch)
        print('\n')

    return criterion, optimizer


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count