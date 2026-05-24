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

Exercise the full pipeline logic with minimal data. Use after refactors to verify nothing is structurally broken.

MNIST (a few seconds): trains one MiniConv at 5 steps and quantizes it at both 4 and 8 bits.

```sh
DEV=CUDA DEBUG=2 JITBEAM=1 python -m experiments.quant_shift_pipeline --n-models 1 --steps 5
```

ResNet (a few minutes on GPU): `--n-per-class 1` is the single global handle — it throttles every stage that touches ImageNet (Stage 3 shift collection, Stage 4 benchmark, Stage 5 CAM cosine analysis) to one image per class (1000 images total). The pipeline always runs both 4- and 8-bit quantization.

```sh
DEV=CUDA DEBUG=2 JITBEAM=1 python -m experiments.resnet_pipeline --dataset-path <imagenet-root> --n-per-class 1
```

Expected artefacts under `results/resnet18_<ts>/`:
- `4bits/`, `8bits/` — per-bit stage outputs (`weight_analysis/`, `quantization_shift/`, `bias_variance_correction/`, `per_image_cosine.csv`)
- `per_image_cosine.csv`, `global_summary.csv`, `cosine_bar_chart.png`, `qualitative_grid.png`, `selected_cams.json`, `selected_image_{A,B,C,D}.png` at the root

### Standalone utilities

Small one-shot scripts independent of the main experiments.

```sh
python -m examples.mnist_save_images
python -m examples.mnist_images_grid
python -m examples.mnist_train
python -m examples.resnet_weights_analysis --model resnet18
python -m examples.resnet_weights_analysis --model resnet101  # For appendix
python -m examples.weights_per_channel --model resnet18 --layer layer1.0.conv2 --rows 3 --cols 7 --quantile 99  # small kernels
python -m examples.weights_per_channel --model resnet18 --layer layer4.0.conv2 --rows 3 --cols 7 --quantile 99.9  # big kernels
DEV=CPU python -m examples.cam_grid --dataset-path /home/kamilis/Downloads/imagenet-object-localization-challenge  --n-images 7 --bits 8 --seed 2
python -m experiments.visualize_crop <image-path>
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