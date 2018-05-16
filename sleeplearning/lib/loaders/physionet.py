import math
import os
import numpy as np
import glob
from datetime import datetime
from mne.io import read_raw_edf
from sleeplearning.lib.loaders.subject import Subject

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 5,
    "Sleep stage ?": 6,
    "Movement time": 6
}

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 5
UNKNOWN = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

EPOCH_SEC_SIZE = 30


class Physionet(Subject):
    """
    Loader for https://www.physionet.org/physiobank/database/sleep-edfx/ files.
    """
    def __init__(self, path: str, epoch_length: int = 30):
        super().__init__(path, epoch_length)
        filename_base = os.path.basename(path)
        psg_file = path
        ann_file = glob.glob(os.path.join(os.path.dirname(path),filename_base.split('-')[0][:-2]+'*Hypnogram.edf'))[0]
        select_ch = 'EEG Fpz-Cz'

        self.label = filename_base.split('-')[0]
        self.psgs = {}
        self.sampling_rate_ = 100

        raw = read_raw_edf(psg_file, preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        # Get raw header
        # f = open(psg_file, 'r')
        f = open(psg_file, 'r', encoding='latin1')
        reader_raw = BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.datetime.strptime(h_raw['date_time'],
                                         "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_file, 'r')
        reader_ann = BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.datetime.strptime(h_ann['date_time'],
                                         "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []  # indicies of the data that will be removed
        labels = []  # indicies of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(
                    duration_sec * sampling_rate, dtype=np.int)
                label_idx.append(idx)

                print("Include onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str)
                )
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(
                    duration_sec * sampling_rate, dtype=np.int)
                remove_idx.append(idx)

                print("Remove onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str
                ))
        labels = np.hstack(labels)

        print("before remove unwanted: {}".format(
            np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # Select only the data with labels
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print("after intersect label: {}".format(select_idx.shape))

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print("before remove extra labels: {}, {}".format(select_idx.shape,
                                                              labels.shape))
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                n_label_trims = int(
                    math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                select_idx = select_idx[:-n_trims]
                labels = labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape,
                                                             labels.shape))

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)
        assert len(x) == len(y)


        # TODO: Why cut this off??
        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        self.psgs[select_ch] = x.reshape((-1))
        self.hypnogram = y


#'''
#Reader for EDF+ files.
#TODO:
# - add support for log-transformed channels:
#   http://www.edfplus.info/specs/edffloat.html and test with
#   data generated with
#   http://www.edfplus.info/downloads/software/NeuroLoopGain.zip.
# - check annotations with Schalk's Physiobank data.
#Copyright (c) 2012 Boris Reuderink.
#'''

import re, datetime, operator, logging
from collections import namedtuple

EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)

#class EDFEndOfData: pass


def tal(tal_str):
  '''Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
  stream.
  '''
  exp = b'(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
    b'(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
    b'(\x14(?P<annotation>[^\x00]*))?' + \
    b'(?:\x14\x00)'

  def annotation_to_list(annotation):
    #return unicode(annotation, 'utf-8').split('\x14') if annotation else []
    return str(annotation, 'utf-8').split('\x14') if annotation else []

  def parse(dic):
    return (
      float(dic['onset']),
      float(dic['duration']) if dic['duration'] else 0.,
      annotation_to_list(dic['annotation']))

  return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]


def edf_header(f):
  h = {}
  assert f.tell() == 0  # check file position
  assert f.read(8) == '0       '

  # recording info)
  h['local_subject_id'] = f.read(80).strip()
  h['local_recording_id'] = f.read(80).strip()

  # parse timestamp
  (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
  (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
  h['date_time'] = str(datetime.datetime(year + 2000, month, day,
    hour, minute, sec))

  # misc
  header_nbytes = int(f.read(8))
  subtype = f.read(44)[:5]
  h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
  h['contiguous'] = subtype != 'EDF+D'
  h['n_records'] = int(f.read(8))
  h['record_length'] = float(f.read(8))  # in seconds
  nchannels = h['n_channels'] = int(f.read(4))

  # read channel info
  channels = range(h['n_channels'])
  h['label'] = [f.read(16).strip() for n in channels]
  h['transducer_type'] = [f.read(80).strip() for n in channels]
  h['units'] = [f.read(8).strip() for n in channels]
  h['physical_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['physical_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['prefiltering'] = [f.read(80).strip() for n in channels]
  h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
  f.read(32 * nchannels)  # reserved

  assert f.tell() == header_nbytes
  return h


class BaseEDFReader:
  def __init__(self, file):
    self.file = file


  def read_header(self):
    self.header = h = edf_header(self.file)

    # calculate ranges for rescaling
    self.dig_min = h['digital_min']
    self.phys_min = h['physical_min']
    phys_range = h['physical_max'] - h['physical_min']
    dig_range = h['digital_max'] - h['digital_min']
    assert np.all(phys_range > 0)
    assert np.all(dig_range > 0)
    self.gain = phys_range / dig_range


  def read_raw_record(self):
    '''Read a record with data and return a list containing arrays with raw
    bytes.
    '''
    result = []
    for nsamp in self.header['n_samples_per_record']:
      samples = self.file.read(nsamp * 2)
      if len(samples) != nsamp * 2:
        #raise EDFEndOfData
        raise EOFError
      result.append(samples)
    return result


  def convert_record(self, raw_record):
    '''Convert a raw record to a (time, signals, events) tuple based on
    information in the header.
    '''
    h = self.header
    dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
    time = float('nan')
    signals = []
    events = []
    for (i, samples) in enumerate(raw_record):
      if h['label'][i] == EVENT_CHANNEL:
        #ann = tal(samples)
        ann = tal(str.encode(samples))
        time = ann[0][0]
        events.extend(ann[1:])
        # print(i, samples)
        # exit()
      else:
        # 2-byte little-endian integers
        dig = np.fromstring(samples, '<i2').astype(np.float32)
        phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
        signals.append(phys)

    return time, signals, events


  def read_record(self):
    return self.convert_record(self.read_raw_record())


  def records(self):
    '''
    Record generator.
    '''
    try:
      while True:
        yield self.read_record()
    #except EDFEndOfData:
    except EOFError:
      pass