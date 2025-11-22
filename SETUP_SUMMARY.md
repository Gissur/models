# Setup Summary: Die Detection for Raspberry Pi 4

## What Has Been Set Up

I've configured TensorFlow Model Garden to train a lightweight object detection model for your dice dataset, optimized for Raspberry Pi 4 deployment.

## Files Created

### Configuration
- **`configs/die_detection_mobilenet_retinanet.yaml`**
  - MobileNetV2 + RetinaNet configuration
  - Optimized for 600x600 images
  - 7 classes (invalid + pip1-6)
  - Mobile-friendly settings (separable convolutions, ReLU6 activation)

### Scripts
- **`scripts/prepare_die_dataset.py`**
  - Converts COCO JSON to TFRecord format
  - Splits data into train/val (80/20)
  - Creates label map file

- **`scripts/train_die_detector.sh`**
  - Complete training pipeline
  - Automatically prepares data if needed
  - Logs training progress

- **`scripts/export_to_tflite.sh`**
  - Exports trained model to TensorFlow Lite
  - Creates both quantized (recommended) and float32 versions
  - Quantized model is ~600KB

- **`scripts/inference_rpi.py`**
  - Raspberry Pi inference script
  - Works with TFLite runtime
  - Visualizes detections with bounding boxes
  - Optimized for low-memory devices

- **`scripts/quick_start.sh`**
  - Runs entire pipeline: prepare → train → export
  - One-command solution

### Documentation
- **`DIE_DETECTION_README.md`**
  - Complete usage guide
  - Training instructions
  - Deployment to Raspberry Pi
  - Troubleshooting tips
  - Performance benchmarks

## Model Architecture

**MobileNetV2-RetinaNet**
```
Input (600x600x3)
    ↓
MobileNetV2 Backbone (lightweight CNN)
    ↓
Feature Pyramid Network (multi-scale features)
    ↓
RetinaNet Detection Head (4 conv layers)
    ↓
Output: Bounding boxes + class predictions
```

**Key Features:**
- Separable convolutions (lighter than standard conv)
- ReLU6 activation (mobile-optimized)
- Dynamic range quantization support
- ~2.3M parameters (vs ~44M for ResNet50-based models)

## Quick Start

### Step 0: Setup Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
bash scripts/setup_venv.sh
source venv/bin/activate
```

### Option 1: All-in-one
```bash
bash scripts/quick_start.sh
```

### Option 2: Step-by-step
```bash
# 1. Prepare data
python3 scripts/prepare_die_dataset.py

# 2. Train model
bash scripts/train_die_detector.sh

# 3. Export to TFLite
bash scripts/export_to_tflite.sh
```

**Note**: Always activate the virtual environment before training:
```bash
source venv/bin/activate  # Do this first!
```

## Current Dataset

- **Images**: 142 (600x600 pixels)
- **Classes**: 7
  - red_die_invalid
  - red_die_pip1, red_die_pip2, red_die_pip3
  - red_die_pip4, red_die_pip5, red_die_pip6
- **Training**: ~113 images (80%)
- **Validation**: ~29 images (20%)

## Expected Results

### Training Time
- **GPU (NVIDIA GTX 1060 or better)**: 2-4 hours
- **CPU only**: 8-12 hours
- **Cloud TPU**: 30-60 minutes

### Model Performance (estimated)
- **Model size (quantized)**: ~600KB
- **Inference time on RPi 4**: 100-200ms per image
- **Expected mAP@0.5**: 60-80% (depends on data quality)

### Accuracy Expectations
With 142 images:
- Good for proof-of-concept
- May struggle with edge cases
- Recommend collecting 500+ images for production use

## Next Steps

1. **Start Training**
   ```bash
   bash scripts/quick_start.sh
   ```

2. **Monitor with TensorBoard**
   ```bash
   tensorboard --logdir=trained_models/die_detector_mobilenet
   # Open http://localhost:6006
   ```

3. **After Training, Test Locally**
   ```bash
   python3 scripts/inference_rpi.py \
       --model trained_models/die_detector_mobilenet/tflite/die_detector_quantized.tflite \
       --image label_studio/600x600/COCO/images/[any_image].png \
       --labels label_studio/tfrecords/label_map.pbtxt
   ```

4. **Deploy to Raspberry Pi**
   ```bash
   # Copy files
   scp trained_models/die_detector_mobilenet/tflite/die_detector_quantized.tflite pi@raspberrypi:~/
   scp label_studio/tfrecords/label_map.pbtxt pi@raspberrypi:~/
   scp scripts/inference_rpi.py pi@raspberrypi:~/

   # On Raspberry Pi, install dependencies
   pip3 install tflite-runtime pillow numpy

   # Run inference
   python3 inference_rpi.py --model die_detector_quantized.tflite --image test.jpg --labels label_map.pbtxt
   ```

## Customization Options

### Faster Inference (Lower Accuracy)
Edit `configs/die_detection_mobilenet_retinanet.yaml`:
```yaml
backbone:
  mobilenet:
    filter_size_scale: 0.5  # Half the filters = 2x faster
```

### Higher Accuracy (Slower Inference)
```yaml
backbone:
  mobilenet:
    filter_size_scale: 1.4  # More filters = better accuracy
model:
  input_size: [800, 800, 3]  # Larger input = better detection
```

### Different Input Resolution
For 440x440 dataset:
```yaml
task:
  model:
    num_classes: 6  # Only 6 classes in 440x440
    input_size: [440, 440, 3]
  train_data:
    input_path: 'label_studio/440x440/COCO/result.json'
```

## Troubleshooting

### "Out of memory" during training
Reduce batch size in config:
```yaml
global_batch_size: 4  # or even 2
```

### Model not learning
- Check TensorBoard (losses should decrease)
- Ensure data is correctly loaded
- Try higher learning rate: `initial_learning_rate: 0.16`

### Slow on Raspberry Pi
- Use quantized model (not float32)
- Reduce input size to 416x416 or 320x320
- Use filter_size_scale: 0.5 or 0.75

## Hardware Requirements

### Training Machine
- **Minimum**: 8GB RAM, CPU with AVX support
- **Recommended**: 16GB RAM, NVIDIA GPU (4GB+ VRAM)
- **Optimal**: 32GB RAM, NVIDIA GPU (8GB+ VRAM)

### Raspberry Pi 4
- **Model**: Raspberry Pi 4B
- **RAM**: 2GB minimum, 4GB recommended
- **OS**: Raspberry Pi OS (32-bit or 64-bit)
- **Camera** (optional): Pi Camera Module v2 or USB webcam

## Additional Resources

- **TensorFlow Model Garden**: See [official/README.md](official/README.md)
- **Complete Guide**: See [DIE_DETECTION_README.md](DIE_DETECTION_README.md)
- **Configuration**: See [configs/die_detection_mobilenet_retinanet.yaml](configs/die_detection_mobilenet_retinanet.yaml)
- **TFLite Guide**: https://www.tensorflow.org/lite/guide
- **Raspberry Pi Setup**: https://www.raspberrypi.org/documentation/

## Questions?

Check the DIE_DETECTION_README.md for detailed information, or review:
- Training issues: TensorBoard logs at `trained_models/die_detector_mobilenet/`
- Data issues: Verify with `scripts/prepare_die_dataset.py`
- Deployment issues: Test locally first before deploying to RPi

---

**You're all set!** Run `bash scripts/quick_start.sh` to begin training.
