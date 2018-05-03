import os
import shutil
import sys
import time
from typing import List, Tuple
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from sleeplearning.lib.loaders.subject import Subject
from sleeplearning.lib.model import Net


class SleepLearningDataset(object):
    """Sleep Learning dataset."""

    def __init__(self, foldr: str, class_remapping: dict, feature_extractor,
                 neighbors=4, discard_arts=True, transform=None):
        dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.dir = os.path.join(dir, 'data', 'processed', foldr)
        #self.dir = 'data/processed/' + foldr + '/'
        self.X = []
        self.class_remapping = class_remapping

        self.transform = transform
        subject_files = [filename for filename in
                         os.listdir(self.dir) if
                         filename.endswith(".npz")]
        class_distribution = np.zeros(
            len(Subject.sleep_stages_labels.keys()))
        subject_labels = []
        for subject_file in subject_files:
            subject = np.load(os.path.join(self.dir, subject_file))
            subject_labels.append(subject['subject_label'].item())
            psgs_reshaped = subject['psgs'].item()

            # [num_epochs X num_channels X freq_domain X time_domain]
            feature_matrix = feature_extractor.fit_transform(psgs_reshaped)
            del psgs_reshaped
            num_epochs = feature_matrix.shape[0]
            # pad with zeros before and after (additional '#neighbors' epochs)
            feature_matrix = np.pad(feature_matrix, (
                (neighbors // 2, neighbors // 2), (0, 0), (0, 0), (0, 0)),
                                    mode='constant')
            # create samples with neighbors
            feature_matrix = np.array([np.concatenate(
                feature_matrix[i - neighbors // 2:i + neighbors // 2 + 1],
                axis=2) for i
                in range(neighbors // 2, num_epochs + neighbors // 2)])

            for e, (sample, label_int) in enumerate(
                    zip(feature_matrix, subject['labels'])):
                label = Subject.sleep_stages_labels[label_int]
                if discard_arts and label == 'Artifact':
                    continue
                class_distribution[label_int] += 1
                id = subject[
                         'subject_label'].item() + '_epoch_' + '{0:0>5}'.format(
                    e) + '_' + str(neighbors) + 'N_' + label
                sample = {'id': id, 'x': sample, 'y': label_int}
                self.X.append(sample)

        self.dataset_info = {}
        class_distribution_dict = {}
        for i in range(len(class_distribution)):
            class_distribution_dict[Subject.sleep_stages_labels[i]] = int(
                class_distribution[i])
        self.dataset_info['subjects'] = subject_labels
        self.dataset_info['class_distribution'] = class_distribution_dict
        self.dataset_info['input_shape'] = feature_matrix[0].shape

        remapped_distribution = {}
        for k in class_remapping.values():
            remapped_distribution[k] = 0
        for k, v in self.dataset_info['class_distribution'].items():
            remapped_distribution[class_remapping[k]] += v
        self.dataset_info['class_distribution'] = remapped_distribution

        self.class_int_mapping = {}
        for i, k in enumerate(self.dataset_info['class_distribution'].keys()):
            self.class_int_mapping[k] = i

        self.weights = np.zeros(len(self.dataset_info['class_distribution'].keys()))
        total_classes = 0.0
        for i, (k, v) in enumerate(self.dataset_info['class_distribution'].items()):
            total_classes += v
            self.weights[i] = v+1  # smooth
        self.weights = total_classes/self.weights

    def __getitem__(self, index):
        sample = self.X[index]
        x = sample['x']
        y_str = Subject.sleep_stages_labels[int(sample['y'])]
        y_str_remapped = self.class_remapping[y_str]
        y_ = int(self.class_int_mapping[y_str_remapped])

        if self.transform is not None:
            x = self.transform(x)
        x = torch.from_numpy(x).double()

        return x, y_

    def __len__(self):
        return len(self.X)


def train_epoch(train_loader: DataLoader, model: Net, criterion, optimizer,
                epoch, cuda, log_interval):
    model.train()
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).long()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        (prec1, prec2), _ = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top2.update(prec2[0], data.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            print('Epoch: [{0}]x[{1}/{2}]\t'
                 'Time {batch_time.val:.1f} ({batch_time.avg:.1f})\t'
                  'Loss {loss.val:.1f} ({loss.avg:.1f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@2 {top2.val:.2f} ({top2.avg:.2f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top2=top2))


def validation_epoch(val_loader: DataLoader, model: Net, criterion, cuda: bool) \
        -> Tuple[float, np.array]:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    prediction = np.array([])
    model.eval()
    torch.set_grad_enabled(False)
    end = time.time()
    predictions = []
    for data, target in val_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(
            target).long()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        (prec1, prec2), prediction = accuracy(output, target, topk=(1, 2))
        predictions.append(prediction)
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top2.update(prec2[0], data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test:  [{0}/{0}]\t\t'
          'Time ({batch_time.avg:.1f})\t'
          'Loss ({loss.avg:.1f})\t'
          'Prec@1 ({top1.avg:.2f})\t\t'
          'Prec@2 ({top2.avg:.2f})'.format(
        len(val_loader), batch_time=batch_time,
        loss=losses, top1=top1, top2=top2))
    predictions = torch.cat(predictions)
    return top1.avg, predictions


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)) -> Tuple[List[List[float]], torch.autograd.Variable]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # TODO: keep it as torch Variable and only convert for output if needed
        res.append((correct_k.mul_(100.0 / batch_size)).data.cpu().numpy())
    return res, pred[0]


def test(test_loader: DataLoader, model: Net, cuda) -> np.array:
    model.eval()
    prediction = np.array([])
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float(), Variable(target).long()
        # compute output
        output = model(data)
        _, pred = output.topk(1, 1, True, True)
        prediction = np.append(prediction, pred.data.cpu().numpy())
    return prediction.astype(int)


def save_checkpoint(state, is_best, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best.pth.tar')