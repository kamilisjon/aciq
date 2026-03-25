## Installation
```sh
git clone https://github.com/kamilisjon/aciq.git
cd aciq
conda create -n aciq python=3.11 -y
conda activate aciq
pip install -e .
```

## Usage
1. Download imagenet validation dataset from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview

## Local testing
### Install extra dependencies
```sh
pip install -e '.[pre_commit,linting,testing]'
```
### Run tests
```sh
pre-commit run --all-files
```