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
    1. `DEV=NV JITBEAM=4 python -m examples.mnist_quantization_shift --n-models 2 --steps 100`
1. Single MNIST training run (no quantization, useful for sanity-checking the training pipeline in isolation):
    1. `DEV=NV JITBEAM=4 python -m examples.train_mnist --steps 1170 --eval-every 10`
    1. Flags: `--seed` (default 0), `--steps` (default 1170), `--lr` (default 1e-3), `--batch-size` (default 512), `--eval-every` (default 10). Progress bar shows live `loss: X.XX test_accuracy: YY.YY%`; the script prints the final accuracy and last logged train/test losses on exit.

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