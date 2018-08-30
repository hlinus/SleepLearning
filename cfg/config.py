import os
import torch
import sys
from sacred import Experiment
from sacred.observers import MongoObserver

root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)

basedir = os.path.join(root_dir, 'sleeplearning', 'lib')
ex = Experiment(base_dir=basedir)
mongo_url = 'mongodb://toor:y0qXDe3qumoawG0rPfnS@cab-e81-31/admin?authMechanism' \
            '=SCRAM-SHA-1'
MONGO_OBSERVER = MongoObserver.create(url=mongo_url, db_name='sacred')
ex.observers.append(MONGO_OBSERVER)


@ex.config
def cfg():
    cmt = ''  # comment for this run
    cuda = torch.cuda.is_available()
    seed = 42  # for reproducibility
    log_dir = '/cluster/scratch/hlinus/logs'
    save_model = False
    save_best_only = False
    early_stop = True

    # default dataset settings
    ds = {
        'channels': None,
        'data_dir': os.path.join('../../../physionet-challenge-train'),
        'train_csv': None,
        'val_csv': None,
        'batch_size_train': 32,
        'batch_size_val': 128,
        'loader': 'Physionet18',
        'nbrs': 2,
        'fold': None,  # only specify for CV
        'oversample': False,
        'nclasses': 5,
    }

@ex.named_config
def MediumAdaptive():
    arch = 'MediumAdaptive'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.00005',
        'weighted_loss': True
    }


@ex.named_config
def paris():
    arch = 'Paris'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.00005',
        'weighted_loss': True
    }

@ex.named_config
def paris2d():
    arch = 'Paris2d'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.00005',
        'weighted_loss': True
    }


@ex.named_config
def multvarnet():
    arch = 'MultivariateNet'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.00005',
        'fc_d': [[4096, 0],[2048, 0]],
        'weighted_loss': True
    }


@ex.named_config
def Mode():
    arch = 'Mode'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'expert_ids': list(range(1452, 1459)),
        'train_emb': True,
        'weighted_loss': True
    }

@ex.named_config
def trainedExpAtt():
    arch = 'TrainedExpertsAtt'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        #'fc_d': [[512, .5],[256, .3]],
        'expert_ids': list(range(1242,1249)),
        #'input_dim': None,  # will be set automatically
        'weighted_loss': True
    }

@ex.named_config
def sleepstage():
    arch = 'SleepStage'

    ms = {
        'epochs': 30,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'weighted_loss': True
    }

@ex.named_config
def EarlyFusion():
    arch = 'EarlyFusion'

    ms = {
        'epochs': 25,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'weighted_loss': True
    }

@ex.named_config
def LateFusion():
    arch = 'LateFusion'

    ms = {
        'epochs': 50,
        'dropout': .5,
        'train_emb': True,
        'optim': 'adam,lr=0.000005',
        'expert_ids': list(range(1452, 1459)),
        'weighted_loss': True
    }

@ex.named_config
def trainedExpAtt2():
    arch = 'TrainedExpertsAtt2'

    ms = {
        'epochs': 15,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'sum_exp': False,
        #'xavier_init': True,
        'expert_ids': list(range(1242,1249)),
        #'input_dim': None,  # will be set automatically
        'weighted_loss': True
    }


@ex.named_config
def GrangerAmoe():
    arch = 'GrangerAmoe'

    ms = {
        'epochs': 15,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'sum_exp': False,
        # 'xavier_init': True,
        'expert_ids': list(range(1242, 1249)),
        # 'input_dim': None,  # will be set automatically
        'weighted_loss': True,
        'loss': 'granger'
    }


@ex.named_config
def SimpleAmoe():
    arch = 'SimpleAmoe'

    ms = {
        'epochs': 15,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'sum_exp': False,
        # 'xavier_init': True,
        'expert_ids': list(range(1242, 1249)),
        # 'input_dim': None,  # will be set automatically
        'weighted_loss': True,
    }

@ex.named_config
def multvar2dnet():
    arch = 'Multivariate2dNet'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.00005',
        'fc_d': [],
        'input_dim': None,  # will be set automatically
        'weighted_loss': True
    }


@ex.named_config
def singlechanexp_bak():
    arch = 'SingleChanExpert'

    ms = {
        'epochs': 25,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'fc_d': [[128,0]],
        'input_dim': None,  # will be set automatically
        'weighted_loss': True
    }

@ex.named_config
def ResNet():
    arch = 'Resnet'

    ms = {
        'epochs': 25,
        'optim': 'adam,lr=0.000005',
        'weighted_loss': True
    }

@ex.named_config
def singlechanexp2_bak():
    arch = 'SingleChanExpert2'

    ms = {
        'epochs': 15,
        'dropout': .5,
        'optim': 'adam,lr=0.00001',
        'weighted_loss': True
    }

@ex.named_config
def singlechanexp():
    arch = 'SingleChanExpert'

    ms = {
        'epochs': 15,
        'dropout': .5,
        'optim': 'adam,lr=0.00001',
        'weighted_loss': True
    }

@ex.named_config
def multvarnet2d():
    arch = 'MultivariateNet2d'

    ms = {
        'epochs': 100,
        'dropout': .5,
        'optim': 'adam,lr=0.000005',
        'fc_d': [[512,0]],
        'input_dim': None,  # will be set automatically
        'weighted_loss': True
    }


@ex.named_config
def exp_avg():
    arch = 'ExpertsAvg'

    ms = {
        'epochs': 1,
        'optim': 'adam,lr=0.00005',
        'expert_ids': list(range(988,995)),
        'weighted_loss': True
    }


@ex.named_config
def ALL_CHAN_2D():
    ds = {

        'channels': [
            ('C3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
            ('C4-M1', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
            ('E1-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
            ('F4-M1', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
            ('O1-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
            ('O2-M1', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             ),
             ]
    }


@ex.named_config
def F3M2_C4M1_E1M2_2D2():
    ds = {
        'channels': [
            ('F3-M2', [
                'BandPass(fs=200, lowpass=30, highpass=.5)',
                'Resample(epoch_len=30, fs=100)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDScaler()'
            ]
             ),
            ('C4-M1', [
                'BandPass(fs=200, lowpass=30, highpass=.5)',
                'Resample(epoch_len=30, fs=100)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDScaler()'
            ]),
            ('E1-M2', [
                'BandPass(fs=200, lowpass=30, highpass=.5)',
                'Resample(epoch_len=30, fs=100)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDScaler()'
            ]),
            ('Chin1-Chin2', [
                'BandPass(fs=200, lowpass=12, highpass=0.5)',
                'Resample(epoch_len=30, fs=100)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_C4M1_E1M2_Chin_2D():
    ds = {
        'channels': [('F3-M2', [
            'Resample(epoch_len=30, fs=100)',
            'BandPass(fs=100, lowpass=45, highpass=.5)',
            'Spectrogram(fs=100, window=150, stride=100)',
            'LogTransform()',
            'TwoDFreqSubjScaler()'
        ]
          ),
         ('C4-M1', [
             'Resample(epoch_len=30, fs=100)',
             'BandPass(fs=100, lowpass=45, highpass=.5)',
             'Spectrogram(fs=100, window=150, stride=100)',
             'LogTransform()',
             'TwoDFreqSubjScaler()'
         ]
          ),
         ('E1-M2', [
             'Resample(epoch_len=30, fs=100)',
             'BandPass(fs=100, lowpass=45, highpass=.5)',
             'Spectrogram(fs=100, window=150, stride=100)',
             'LogTransform()',
             'TwoDFreqSubjScaler()'
         ]
          ),
         ('Chin1-Chin2', [
             'Resample(epoch_len=30, fs=100)',
             'BandPass(fs=100, lowpass=12, highpass=.5)',
             'Spectrogram(fs=100, window=150, stride=100)',
             'LogTransform()',
             'TwoDFreqSubjScaler()'
         ]
          )
         ]
    }


@ex.named_config
def F3M2_C4M1_E1M2_Chin():
    ds = {
        'channels': [('F3-M2', [
            'BandPass(fs=200, lowpass=30, highpass=.5)',
            'Resample(epoch_len=30, fs=128)',
            'OneDScaler()'
        ]
          ),
         ('C4-M1', [
             'BandPass(fs=200, lowpass=30, highpass=.5)',
             'Resample(epoch_len=30, fs=128)',
             'OneDScaler()'
         ]
          ),
         ('E1-M2', [
             'BandPass(fs=200, lowpass=30, highpass=.5)',
             'Resample(epoch_len=30, fs=128)',

             'OneDScaler()'
         ]
          ),
         ('Chin1-Chin2', [
             'BandPass(fs=200, lowpass=12, highpass=.5)',
             'Resample(epoch_len=30, fs=128)',
             'OneDScaler()'
         ]
          )
         ]
    }


@ex.named_config
def F3M2():
    ds = {
        'channels': [
            ('F3-M2', ['BandPass(fs=200, lowpass=30, highpass=.5)',
                       'Resample(epoch_len=30, fs=128)',
                       'OneDScaler()']) # 'ConvToInt16()'
        ]
    }


@ex.named_config
def F3M2_2D():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2DP():
    ds = {
        'channels': [
            ('F3-M2', [
                'ResamplePoly(epoch_len=30, fs=200)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2DP_MULTIT():
    ds = {
        'channels': [
            ('F3-M2', [
                'ResamplePoly(epoch_len=30, fs=200)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'SpectrogramMultiTaper(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_2DPEP():
    ds = {
        'channels': [
            ('F3-M2', [
                'ResamplePoly(epoch_len=30, fs=200)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_2D_30HZ():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }



@ex.named_config
def F3M2_2D_CUT():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_CUTM():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'SpectrogramM(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_2D_CUT_LOG2():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'LogTransform2()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_CUT_EPN():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'LogTransform()',
                'TwoDScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_2D_MF():
    ds = {
        'channels': [
            ('F3-M2', [
                'Spectrogram(fs=200, window=300, stride=200)',
                'CutFrequencies(fs=200, window=300,lower=0, upper=30)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_CUT_NOLOG():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_CUT_NOLOG_EN():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'TwoDScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_NOSC():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'LogTransform()',
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_EPNORM():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_UNNORM():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
            ]
             )
        ]
    }

@ex.named_config
def F3M2_2D_PER_SAMP_SCALE():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'TwoDScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_2D_NOLOG():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }



@ex.named_config
def F3M2_Z3():
    ds = {
        'channels': [
            ('F3-M2', [
                'Z3()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }

@ex.named_config
def F3M2_SleepStage():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'SleepStage()',
            ]
             )
        ]
    }


@ex.named_config
def F3M2_2D_NEW():
    ds = {
        'channels': [
            ('F3-M2', [
                'Resample(epoch_len=30, fs=100)',
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram2(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ]
             )
        ]
    }


@ex.named_config
def F3M2_200Hz_2D():
    ds = {
        'channels': [('F3-M2', [
            'BandPass(fs=200, lowpass=45, highpass=.5)',
            'Spectrogram(fs=200, window=300, stride=200)',
            'CutFrequencies(fs=200, window=300, '
            'lower=0, upper=45)',
            'LogTransform()',
            'TwoDFreqSubjScaler()'
        ]
                      )]
    }


@ex.named_config
def E1M2_2D():
    ds = {
        'channels': [('E1-M2', [
            'Resample(epoch_len=30, fs=100)',
            'BandPass(fs=100, lowpass=45, highpass=.5)',
            'Spectrogram(fs=100, window=150, stride=100)',
            'LogTransform()',
            'TwoDFreqSubjScaler()'
        ])]
    }


@ex.named_config
def O2M1_2D():
    ds = {
        'channels': [('O2-M1', [
             'Resample(epoch_len=30, fs=100)',
             'BandPass(fs=100, lowpass=45, highpass=.5)',
             'Spectrogram(fs=100, window=150, stride=100)',
             'LogTransform()',
             'TwoDFreqSubjScaler()'
        ])]
    }


@ex.named_config
def C4M1_2D():
    ds = {
        'channels': [('C4-M1', [
             'Resample(epoch_len=30, fs=100)',
             'BandPass(fs=100, lowpass=45, highpass=.5)',
             'Spectrogram(fs=100, window=150, stride=100)',
             'LogTransform()',
             'TwoDFreqSubjScaler()'
        ])]
    }


@ex.named_config
def C3M2_2D():
    ds = {
        'channels': [('C3-M2', [
            'Resample(epoch_len=30, fs=100)',
            'BandPass(fs=100, lowpass=45, highpass=.5)',
            'Spectrogram(fs=100, window=150, stride=100)',
            'LogTransform()',
            'TwoDFreqSubjScaler()'
        ])]
    }


@ex.named_config
def F4M1_2D():
    ds = {
        'channels': [('F4-M1', [
            'Resample(epoch_len=30, fs=100)',
            'BandPass(fs=100, lowpass=45, highpass=.5)',
            'Spectrogram(fs=100, window=150, stride=100)',
            'LogTransform()',
            'TwoDFreqSubjScaler()'
        ])]
    }


@ex.named_config
def O1M2_2D():
    ds = {
        'channels': [('O1-M2', [
            'Resample(epoch_len=30, fs=100)',
            'BandPass(fs=100, lowpass=45, highpass=.5)',
            'Spectrogram(fs=100, window=150, stride=100)',
            'LogTransform()',
            'TwoDFreqSubjScaler()'
        ])]
    }


@ex.named_config
def sleepedf():
    loader = 'Sleepedf'

    dchannels = [
        ('EEG-Fpz-Cz', []),
    ]


@ex.named_config
def sleepedf_2D():
    ds = {
        'loader': 'Sleepedf',
        'channels': [
            ('EEG-Fpz-Cz', [
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ])
        ]
    }

@ex.named_config
def sleepedf_Fpz_2D():
    ds = {
        'loader': 'Sleepedf',
        'channels': [
            ('EEG-Fpz-Cz', [
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ])
        ]
    }

@ex.named_config
def sleepedf_Pz_2D():
    ds = {
        'loader': 'Sleepedf',
        'channels': [
            ('EEG-Pz-Oz', [
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ])
        ]
    }

@ex.named_config
def sleepedf_Fpz_2D_CUT():
    ds = {
        'loader': 'Sleepedf',
        'channels': [
            ('EEG-Fpz-Cz', [
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'CutFrequencies(fs=100, window=150,lower=0, upper=30)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ])
        ]
    }


@ex.named_config
def sleepedf_2D_BP30():
    ds = {
        'loader': 'Sleepedf',
        'channels': [
            ('EEG-Fpz-Cz', [
                'BandPass(fs=100, lowpass=30, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'LogTransform()',
                'TwoDFreqSubjScaler()'
            ])
        ]
    }

@ex.named_config
def sleepedf_2D_BAK():
    ds = {
        'loader': 'Sleepedf',
        'channels': [
            ('EEG-Fpz-Cz', [
                'BandPass(fs=100, lowpass=45, highpass=.5)',
                'Spectrogram(fs=100, window=150, stride=100)',
                'TwoDScaler()'
            ])
        ]
    }


@ex.named_config
def three_channels_noscale():
    ds = {
        'channels': [
            ('F3-M2', []),
            ('C4-M1', []),
            ('E1-M2', []),
        ]
    }


@ex.named_config
def three_channels():
    ds = {
        'channels': [
            ('F3-M2', ['OneDScaler()']),
            ('C4-M1', ['OneDScaler()']),
            ('E1-M2', ['OneDScaler()']),
        ]
    }


@ex.named_config
def three_channels_int16():
    ds = {
        'channels': [
            ('F3-M2', ['Resample(epoch_len=30, fs=100)', 'ConvToInt16()']),
            ('C4-M1', ['Resample(epoch_len=30, fs=100)', 'ConvToInt16()']),
            ('E1-M2', ['Resample(epoch_len=30, fs=100)', 'ConvToInt16()']),
        ]
    }



@ex.named_config
def three_channels_filt():
    ds = {
        'channels': [
            ('F3-M2',
             ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
            ('C4-M1',
             ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
            ('E1-M2',
             ['BandPass(fs=100, lowpass=45, highpass=.5)', 'ConvToInt16()']),
        ]
    }


@ex.named_config
def seven_channels():
    ds = {
        'channels': [
            ('F3-M2', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
            ('C4-M1', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
            ('C3-M2', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
            ('E1-M2', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
            ('F4-M1', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
            ('O1-M2', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
            ('O2-M1', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
        ]
    }


@ex.named_config
def one_channel():
    ds = {
        'channels': [
            ('F3-M2', ['Resample(epoch_len=30, fs=100)', 'OneDScaler()']),
        ]
    }
