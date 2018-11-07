# Sleep Learning


## Getting started
 * Download [anaconda](https://docs.anaconda.com/anaconda/install/) for your system
 
    `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
 * Run the installer
 `bash Miniconda3-latest-Linux-x86_64.sh `
 * Create an environment
`conda env create -n sl python=3.6.4 -f environment.yml`
 * Activate the created environment `source activate sl`

## Usage

IMPORTANT: 
* download the sample data first: `cd data && bash download-sample-data.sh`
* [SPACE and TMP data]!!! 
* [Non debug data]

[INCLUDE MODELS IN SAMPLE DATA SCRIPT]


### classify

All preprocessing steps, model architecture and parameters are saved in the 
model file. Just specify a saved model and subject file/directory.

First activate the conda environment and change to the bin directory:

`conda activate sl && cd bin/`

Then specify a model and subject to predict. The prediction is saved to 
<subject>.csv.

E.g. for Sleep-EDF  
`python classify.py --model ../models/cv_sleepedf_Fz_2D_singlechanexp2_6464962FC_MP/fold0/checkpoint.pth.tar --subject ../data/sleepedf/SC4001E0-PSG`

E.g. for Physionet18

`python classify.py --model ../models/3067_LateFusion.pth.tar --subject ../data/physionet-challenge-train/tr11-0064/`

### train on Physionet18 (train/validation split)
First activate the conda environment and change to the bin directory:
`conda activate sl && cd bin/`


#### Single Channel Expert Model
Training the single channel expert model local on the F3M2 channel:

`python train.py with F3M2_2DP singlechanexp save_model=True ds.data_dir=../data/physionet-challenge-train/ ds.train_csv=../cfg/physionet18/physio18_debug.csv ds.val_csv=../cfg/physionet18/physio18_debug_val.csv`

On leonard we request 5 computing nodes with scratch space and a GPU:

`module load python_gpu/3.6.4`

`bsub -n 5 -W 03:59 -R "rusage[mem=6000,scratch=25000,ngpus_excl_p=1]" ~/miniconda3/envs/sl/bin/python train.py with F3M2_2DP singlechanexp save_model=True ds.data_dir=../data/physionet-challenge-train/ ds.train_csv=../cfg/physionet18/physio18_debug.csv ds.val_csv=../cfg/physionet18/physio18_debug_val.csv`

#### Adaptive Mixture-Of-Experts (AMOE) Model

Training the AMOE model local on all channels (6 EEG, 1 EOG, 3 EMG) with 10% 
channel dropout:

`python train.py with PHYSIONET_EEG_EOG_EMG_2D AMOE_RS160 ChannelDropout10 save_model=True ds.data_dir=../data/physionet-challenge-train/ ds.train_csv=../cfg/physionet18/physio18_debug.csv ds.val_csv=../cfg/physionet18/physio18_debug_val.csv`

On leonard we request 5 computing nodes with scratch space and a GPU:

`module load python_gpu/3.6.4`

`bsub -n 5 -W 03:59 -R "rusage[mem=6000,scratch=25000,ngpus_excl_p=1]" ~/miniconda3/envs/sl/bin/python train.py with PHYSIONET_EEG_EOG_EMG_2D AMOE_RS160 ChannelDropout10 save_model=True ds.data_dir=../data/physionet-challenge-train/ ds.train_csv=../cfg/physionet18/physio18_debug.csv ds.val_csv=../cfg/physionet18/physio18_debug_val.csv`


#### Attention Model

#### Late Fusion Model

### train on Sleep-EDF (cross validation)
To reproduce the 20-fold CV results on the Sleep-EDF dataset (Fpz channel) run 
the following bash script in the bin directory:
`bash train-sleepedf-singlechanexp.sh --data_dir=<path to data>`

To run the cross validation on leonhard adjust 
`bash train-sleepedf-singlechanexp.sh --data_dir=<path to data> --leonhard `

### evaluate trained model on a testset

validate.py

### advanced trained model options
All the steps to load a trained model, view its training parameters and how 
to score on single subject is illustrated in a jupyter notebook:

reports/Load-Saved-Model.ipynb




