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

To classify sleep phases of a new subject file refer to the command line instructions
`python bin/classify.py -h`



