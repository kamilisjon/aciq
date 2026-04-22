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
1. ResNet analysis:
    1. `DEV=NV JITBEAM=4 python -m examples.resnet_pipeline --dataset-path /home/kamilis/Downloads/imagenet-object-localization-challenge --model resnet50 --bits 8`
1. MNIST training for evaluating layers outputs distributions shift after quantization:
    1. `DEV=NV JITBEAM=4 python -m examples.mnist_quantization_shift --n-models 100 --epochs 5`

## Local testing
### Install extra dependencies
```sh
pip install -e '.[linting,testing]'
```
### Run tests
```sh
pre-commit run --all-files
```