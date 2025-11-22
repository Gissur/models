# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the TensorFlow Model Garden - a collection of state-of-the-art model implementations for TensorFlow 2. The repository contains three main sections:

- **official/**: Production-ready implementations using TF2 high-level APIs, maintained by the TensorFlow team
- **research/**: Research model implementations, maintained by individual researchers
- **orbit/**: A lightweight library for writing custom training loops in TF2
- **label_studio/**: Custom labeled dataset for red die (dice) detection and classification

### Custom Dataset: label_studio/

This directory contains a custom-labeled dataset for detecting and classifying red dice (six-sided dice). The dataset is organized in two standard object detection formats:

**Dataset Structure:**
```
label_studio/
├── 440x440/              # 440x440 pixel images (15 images)
│   ├── COCO/             # COCO format annotations
│   │   ├── images/
│   │   └── result.json
│   └── YOLO/             # YOLO format annotations
│       ├── images/
│       ├── labels/       # .txt files with YOLO format bounding boxes
│       └── classes.txt
└── 600x600/              # 600x600 pixel images (142 images)
    ├── COCO/             # COCO format annotations
    │   ├── images/
    │   └── result.json
    └── YOLO/             # YOLO format annotations
        ├── images/
        ├── labels/
        └── classes.txt
```

**Classes (600x600 dataset):**
- `red_die_invalid` (class 0) - Invalid die state
- `red_die_pip1` through `red_die_pip6` (classes 1-6) - Die showing 1-6 pips

**Classes (440x440 dataset):**
- `red_d6_pip1` through `red_d6_pip6` (classes 0-5) - Die showing 1-6 pips

**Data Format:**
- YOLO format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
- COCO format: JSON with image annotations and bounding boxes
- Source images: Captured with Windows camera (WIN_*.jpg/png files)

## Installation & Setup

### Method 1: Using pip package
```bash
pip3 install tf-models-official
# For nightly builds with latest changes:
pip3 install tf-models-nightly
```

### Method 2: From source (recommended for development)
```bash
git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:/path/to/models
pip3 install --user -r official/requirements.txt

# For NLP packages, also install:
pip3 install tensorflow-text-nightly
```

## Development Commands

### Running Tests

Tests use TensorFlow's testing framework (based on `absl.testing`). To run tests:

```bash
# Run a single test file
python3 -m official.core.train_lib_test

# Run all tests in a directory
python3 -m pytest official/vision/  # if pytest is available

# Or use Python's unittest discovery
python3 -m unittest discover -s official/vision -p "*_test.py"
```

### Linting

Pylint is used for code quality checks:

```bash
# Check only files changed since master (incremental)
bash ./.github/scripts/pylint.sh --incremental

# Check all Python files (slow)
bash ./.github/scripts/pylint.sh
```

The pylint configuration is downloaded from the main TensorFlow repository during execution.

### Training Models

The `official/` models use a common training driver with YAML configuration files:

```bash
# General training command structure
python3 official/nlp/train.py \
  --experiment=<experiment_name> \
  --mode=train_and_eval \
  --model_dir=<output_directory> \
  --config_file=<path_to_yaml_config>

# Example: Train ResNet-50 on ImageNet
python3 official/vision/train.py \
  --experiment=resnet_imagenet \
  --mode=train_and_eval \
  --model_dir=/tmp/resnet50 \
  --config_file=official/vision/configs/experiments/image_classification/imagenet_resnet50_tpu.yaml
```

Training configuration files are located in:
- Vision: `official/vision/configs/experiments/`
- NLP: `official/nlp/configs/experiments/`
- Recommendation: `official/recommendation/ranking/configs/yaml/`

## Architecture Overview

### Core Training Infrastructure (official/core/)

The `official/core/` directory contains the foundational training infrastructure used across all official models:

- **train_lib.py**: Main training orchestration, handles the train/eval loop
- **train_utils.py**: Utilities for distributed training, checkpointing, and model export
- **base_task.py**: Abstract base class for tasks (datasets + model + loss)
- **base_trainer.py**: Base trainer that uses Orbit for custom training loops
- **config_definitions.py**: Configuration schema definitions using dataclasses
- **exp_factory.py** & **task_factory.py**: Factories for creating experiments and tasks via registration

### Task-Based Architecture

Models are organized around "Tasks" which encapsulate:
1. **Data pipeline** (input_reader.py)
2. **Model definition** (using modeling library)
3. **Loss functions**
4. **Metrics and evaluation**

Each domain (vision, nlp, recommendation) implements domain-specific tasks that inherit from `base_task.Task`.

### Configuration System

Uses YAML files + Python dataclasses for configuration:
- Configurations defined in `<domain>/configs/` directories
- Experiments registered via `exp_factory` decorator
- Override configs via command-line flags or additional YAML files
- Gin-config support also available

### Orbit Training Loop Library

Orbit (`orbit/` directory) provides a flexible abstraction over TensorFlow's training loops:
- **Controller**: Orchestrates training and evaluation
- **StandardRunner**: Implements common train/eval patterns
- **Actions**: Hooks for checkpointing, summaries, evaluation
- Seamless integration with `tf.distribute` for multi-GPU/TPU training

Models in `official/` typically use Orbit through `base_trainer.py`.

### Modeling Building Blocks

#### Vision (official/vision/)
- **backbones/**: Reusable backbone networks (ResNet, EfficientNet, ViT, SpineNet, etc.)
- **decoders/**: Feature pyramid networks (FPN, ASPP, NASFPN)
- **heads/**: Task-specific heads (detection, segmentation, classification)
- **modeling/**: Complete model implementations
- **tasks/**: Vision-specific task implementations

#### NLP (official/nlp/)
- **modeling/layers/**: Transformer layers, attention mechanisms, embeddings
- **modeling/networks/**: Complete encoder networks (BERT, ALBERT, XLNet, etc.)
- **modeling/models/**: Pre-built trainable models
- **tasks/**: NLP-specific tasks (pretraining, classification, QA, etc.)
- **data/**: Tokenization and data preprocessing utilities

#### Projects (official/projects/)

Contains experimental or specialized model implementations:
- Each project is self-contained with its own configs, tasks, and models
- Examples: MoViNet, YOLOv7, DETR, SimCLR, ViT variants
- Often implement recent research papers

### Distribution Strategy

All official models support distributed training:
- **TPU**: Use `distribution_strategy: 'tpu'` in runtime config
- **Multi-GPU**: Use `distribution_strategy: 'mirrored'` or `'multi_worker_mirrored'`
- Mixed precision training via `mixed_precision_dtype: 'bfloat16'` or `'float16'`

## Key Files for Common Tasks

### Adding a new vision model
1. Define model in `official/vision/modeling/`
2. Create task in `official/vision/tasks/`
3. Register task using `@task_factory.register_task_cls`
4. Add config in `official/vision/configs/`
5. Create experiment YAML in `official/vision/configs/experiments/`

### Adding a new NLP model
1. Define layers/network in `official/nlp/modeling/`
2. Create task in `official/nlp/tasks/`
3. Register task and experiment
4. Add config and experiment YAML

### Running on TPU
Ensure config includes:
```yaml
runtime:
  distribution_strategy: 'tpu'
  mixed_precision_dtype: 'bfloat16'
```

### Exporting models
Use `export_base.py` utilities or add export logic to task's `build_model()` method.

### Training on the Custom Die Dataset

To train an object detection model on the custom die dataset in `label_studio/`:

**Using YOLO format with YOLOv7:**
```bash
# The label_studio/ dataset is already in YOLO format
# Use the official/projects/yolo/ implementation

python3 official/projects/yolo/train.py \
  --mode=train_and_eval \
  --model_dir=/tmp/die_detector \
  --config_file=<path_to_custom_yolo_config>
```

**Using COCO format with RetinaNet or Mask R-CNN:**
```bash
# Convert the dataset path in your config to point to label_studio/600x600/COCO/

python3 official/vision/train.py \
  --experiment=retinanet_resnetfpn_coco \
  --mode=train_and_eval \
  --model_dir=/tmp/die_detector \
  --config_file=<path_to_custom_config>
```

**Key configuration adjustments needed:**
- Set `num_classes` to 7 (for 600x600 dataset) or 6 (for 440x440 dataset)
- Update `train_data.input_path` to point to `label_studio/600x600/COCO/result.json` or similar
- Adjust `input_size` to match your chosen resolution (440 or 600)
- Update `classes` list in config to match the die classes
- Consider using transfer learning from ImageNet or COCO pretrained weights

## Testing Conventions

- Test files are named `*_test.py`
- Tests use `absl.testing.parameterized.TestCase` for parameterized tests
- Use `tf.test.TestCase` for TensorFlow-specific testing
- Mock tasks available in `official/utils/testing/mock_task.py`
- Tests should be runnable individually: `python3 -m path.to.test_file`

## Code Style

- Follow TensorFlow's pylint configuration (downloaded automatically)
- Use type hints for function signatures
- Imports: TensorFlow 2.x uses `import tensorflow as tf, tf_keras`
- Configurations use dataclasses from `official.core.config_definitions`

## Research Models

Research models in `research/` are independently maintained:
- Each has its own README with specific setup/training instructions
- May use TF1 or TF2
- Not guaranteed to be maintained long-term
- Examples: object_detection, deeplab, slim
