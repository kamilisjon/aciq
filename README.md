## Installation
```sh
git clone https://github.com/kamilisjon/aciq.git
cd aciq
conda create -n aciq python=3.11 -y
conda activate aciq
pip install -e .
```

## Project structure
- `experiments/` — runnable experiment scripts (MNIST and ResNet quantization
  analysis, pipelines, plotting utilities).
- `aciq/` — the core library: distribution fitting, quantization and clipping,
  bias correction, models, datasets, and plotting helpers.

## ImageNet

Some experiments (e.g. `experiments.resnet_pipeline`) require the ImageNet
validation set. Download it from
<https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview>,
extract it, and pass the extracted root to those scripts via
`--dataset-path <imagenet-root>`.

## Local testing
### Install extra dependencies
```sh
pip install -e '.[linting,testing]'
```
### Format code
```sh
./format.sh
```
### Run tests
```sh
pre-commit run --all-files
```