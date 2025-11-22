# Installation Guide

## Prerequisites Check

Run the verification script to check your setup:

```bash
python3 scripts/verify_setup.py
```

## Install Dependencies

### On Your Training Machine (PC/Laptop/Server)

```bash
# Install TensorFlow and required packages
pip3 install tensorflow>=2.12.0
pip3 install pillow numpy

# Or install all official requirements
pip3 install -r official/requirements.txt

# Verify installation
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### On Raspberry Pi 4

```bash
# Install TFLite Runtime (much lighter than full TensorFlow)
pip3 install tflite-runtime

# Install image processing libraries
pip3 install pillow numpy

# Verify installation
python3 -c "import tflite_runtime; print('TFLite runtime installed successfully')"
```

## Dataset Status

Your dataset is already in place:
- ✓ 142 images (600x600 pixels)
- ✓ 601 annotations
- ✓ 7 classes (red_die_invalid, red_die_pip1-6)
- ✓ COCO format JSON
- ✓ Located in `label_studio/600x600/COCO/`

## Ready to Train?

After installing dependencies, verify again:

```bash
python3 scripts/verify_setup.py
```

If all checks pass, start training:

```bash
bash scripts/quick_start.sh
```

## System Requirements

### Minimum (CPU Training)
- Python 3.7+
- 8GB RAM
- 20GB disk space
- CPU with AVX support

### Recommended (GPU Training)
- Python 3.7+
- 16GB RAM
- 50GB disk space
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.2+ and cuDNN 8.1+

### For Raspberry Pi 4
- Raspberry Pi 4B (2GB+ RAM)
- Raspberry Pi OS (Bullseye or newer)
- 8GB SD card minimum
- Optional: Pi Camera Module v2

## GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU:

```bash
# Check GPU is detected
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# If empty, install CUDA and cuDNN:
# https://www.tensorflow.org/install/gpu
```

## Troubleshooting

### TensorFlow Installation Issues

```bash
# If tensorflow install fails, try:
pip3 install --upgrade pip
pip3 install tensorflow-cpu  # CPU-only version
```

### PIL/Pillow Installation

```bash
# Note: Install 'pillow' not 'PIL'
pip3 install pillow
```

### On macOS with M1/M2 Chip

```bash
# Use tensorflow-macos
pip3 install tensorflow-macos
pip3 install tensorflow-metal  # For GPU acceleration
```

## Next Steps

1. Install dependencies (see above)
2. Run verification: `python3 scripts/verify_setup.py`
3. Start training: `bash scripts/quick_start.sh`
4. See `DIE_DETECTION_README.md` for complete guide
