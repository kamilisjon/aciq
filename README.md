## Installation
```sh
git clone https://github.com/kamilisjon/aciq.git
cd aciq
conda create -n aciq python=3.11 -y
conda activate aciq
pip install -e .
```

## Usage

The two main experiments are `examples.mnist_quantization_shift` and `examples.resnet_pipeline`. ResNet requires the ImageNet validation set — download it from <https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview> and pass the extracted root via `--dataset-path`.

### Smoke runs

Exercise the full pipeline logic with minimal data; each finishes in seconds. Use after refactors to verify nothing is structurally broken.

```sh
DEV=CUDA DEBUG=2 JITBEAM=1 python -m examples.mnist_quantization_shift --n-models 1 --steps 5
DEV=CUDA DEBUG=2 JITBEAM=1 python -m examples.resnet_pipeline --dataset-path <imagenet-root> --model resnet18 --bits 8 --n-per-class 1
```

### Standalone utilities

Small one-shot scripts independent of the main experiments.

```sh
python -m examples.mnist_save_images
python -m examples.mnist_images_grid
python -m examples.mnist_train
python -m examples.distribution_fits
python -m examples.weights_per_layer --model resnet18 --rows 3 --cols 7 --quantile 99.9
python -m examples.weights_per_layer --model resnet101 --rows 11 --cols 9 --quantile 99.9  # For appendix
python -m examples.weight_statistics --model resnet18  # per-layer mean/variance/skewness/excess-kurtosis
python -m examples.resnet_bn_fusion
python -m examples.weights_per_channel --model resnet18 --layer layer1.0.conv2 --rows 3 --cols 7 --quantile 99  # small kernels
python -m examples.weights_per_channel --model resnet18 --layer layer4.0.conv2 --rows 3 --cols 7 --quantile 99.9  # big kernels
```

### Full experiments

Realistic settings — these produce the thesis-grade numbers and plots.

```sh
DEV=CUDA JITBEAM=4 python -m examples.mnist_quantization_shift --n-models 100 --steps 100
DEV=CUDA JITBEAM=4 python -m examples.resnet_pipeline --dataset-path <imagenet-root> --model resnet50 --bits 8
```

### Replot MNIST from a previous experiment

Re-render `mnist_quantization_shift`'s plots from a prior run's CSVs without retraining. Output goes to a new timestamped dir; the source dir is never written to.

```sh
python -m examples.mnist_quantization_shift --from-dir results/mnist_<timestamp>
```

### Replot ResNet from a previous experiment

Re-render `resnet_pipeline`'s per-layer mean-shift plot from a prior run's `quantization_shift/shifts.csv`. No model load, no ImageNet, no GPU. Other ResNet plots (weight distributions, BN fusion effects) need raw model state and can't be replotted from CSV alone.

```sh
python -m examples.resnet_pipeline --model resnet50 --from-dir results/resnet50_<timestamp>
```

`--model` must match the model from the original run (used for the plot title).

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