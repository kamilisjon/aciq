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
python -m examples.mnist_save_images          # export every MNIST image to results/mnist_images_<timestamp>/
python -m examples.mnist_images_grid          # render results/mnist_grid.png with one sample per class
DEV=CUDA JITBEAM=1 python -m examples.mnist_train  # single MNIST training run (no quantization) — sanity-checks training in isolation
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