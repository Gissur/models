# Die Detection for Raspberry Pi 4

This guide walks you through training a lightweight object detection model for detecting and classifying dice, optimized for deployment on Raspberry Pi 4.

## Model Architecture

**MobileNetV2 + RetinaNet**
- **Backbone**: MobileNetV2 (lightweight, mobile-optimized)
- **Detector**: RetinaNet with Feature Pyramid Network
- **Input size**: 600x600 pixels
- **Classes**: 7 (red_die_invalid, red_die_pip1-6)
- **Parameters**: ~2.3M (quantized model ~600KB)
- **Expected inference time on RPi 4**: ~100-200ms per image

## Prerequisites

### On Training Machine (PC/Server)

```bash
# Python 3.7+
pip3 install tensorflow>=2.12.0
pip3 install tf-models-official
pip3 install pillow numpy

# Or install from requirements
pip3 install -r official/requirements.txt
```

### On Raspberry Pi 4

```bash
# TensorFlow Lite runtime (much lighter than full TensorFlow)
pip3 install tflite-runtime

# Image processing
pip3 install pillow numpy
```

## Quick Start

### 1. Prepare the Dataset

Convert your Label Studio COCO format data to TFRecords:

```bash
python3 scripts/prepare_die_dataset.py \
    --coco_json label_studio/600x600/COCO/result.json \
    --image_dir label_studio/600x600/COCO/images \
    --output_dir label_studio/tfrecords \
    --train_split 0.8
```

This will create:
- `label_studio/tfrecords/train.tfrecord` (80% of data)
- `label_studio/tfrecords/val.tfrecord` (20% of data)
- `label_studio/tfrecords/label_map.pbtxt` (class definitions)

### 2. Train the Model

Train using the provided configuration:

```bash
bash scripts/train_die_detector.sh
```

Or manually:

```bash
python3 official/vision/train.py \
    --experiment=retinanet_mobile_coco \
    --mode=train_and_eval \
    --model_dir=trained_models/die_detector_mobilenet \
    --config_file=configs/die_detection_mobilenet_retinanet.yaml
```

**Training time**: ~2-4 hours on GPU, longer on CPU

**Monitor training** with TensorBoard:
```bash
tensorboard --logdir=trained_models/die_detector_mobilenet
```

### 3. Export to TensorFlow Lite

Once training is complete, export to TFLite format for Raspberry Pi:

```bash
bash scripts/export_to_tflite.sh
```

This creates two model files:
- `die_detector_quantized.tflite` - **Recommended** for RPi (smaller, faster)
- `die_detector_float32.tflite` - Higher accuracy, larger size

### 4. Deploy to Raspberry Pi 4

#### Copy files to Raspberry Pi:

```bash
# Copy the quantized model
scp trained_models/die_detector_mobilenet/tflite/die_detector_quantized.tflite pi@raspberrypi:~/

# Copy the label map
scp label_studio/tfrecords/label_map.pbtxt pi@raspberrypi:~/

# Copy the inference script
scp scripts/inference_rpi.py pi@raspberrypi:~/
```

#### Run inference on Raspberry Pi:

```bash
# On Raspberry Pi
python3 inference_rpi.py \
    --model die_detector_quantized.tflite \
    --image test_image.jpg \
    --labels label_map.pbtxt \
    --confidence 0.5 \
    --output result.jpg
```

## Advanced Configuration

### Reduce Model Size Further

Edit `configs/die_detection_mobilenet_retinanet.yaml`:

```yaml
backbone:
  mobilenet:
    filter_size_scale: 0.75  # Reduce from 1.0 to 0.75 or 0.5
```

Smaller values = faster inference, but lower accuracy.

### Adjust Training Parameters

For your small dataset (142 images), the configuration is already optimized:

- **Batch size**: 8 (reduce if out of memory)
- **Training steps**: 5000 (~35 epochs)
- **Learning rate**: Cosine decay from 0.08
- **Data augmentation**: Horizontal flip, scale, color jitter

### Use CPU-only Training

If you don't have a GPU:

```yaml
runtime:
  distribution_strategy: 'one_device'  # Instead of 'mirrored'
```

## Dataset Information

### Current Dataset
- **Images**: 142 (600x600 pixels)
- **Classes**: 7 total
  - `red_die_invalid` (class 0)
  - `red_die_pip1` to `red_die_pip6` (classes 1-6)
- **Format**: COCO JSON + YOLO labels
- **Split**: 80% train (113 images), 20% val (29 images)

### Improving the Model

For better results, consider:

1. **Collect more data** (aim for 500-1000 images)
2. **Balance classes** (ensure each pip count has similar number of images)
3. **Add variety** (different lighting, angles, backgrounds)
4. **Data augmentation** (already enabled in config)

## Troubleshooting

### Out of Memory during Training

Reduce batch size in config:
```yaml
train_data:
  global_batch_size: 4  # Reduce from 8
```

### Model not detecting dice

1. **Check confidence threshold**: Try lowering to 0.3
2. **Verify image preprocessing**: Ensure input images are 600x600
3. **Check label map**: Ensure class IDs match
4. **Train longer**: Increase train_steps to 10000

### Slow inference on Raspberry Pi

1. **Use quantized model**: `die_detector_quantized.tflite`
2. **Reduce input size**: Use 416x416 or 320x320 in config
3. **Use smaller backbone**: Set `filter_size_scale: 0.5`
4. **Enable hardware acceleration**: Use Coral USB Accelerator if available

## Real-time Detection on Raspberry Pi

For camera-based real-time detection, modify the inference script to use picamera:

```python
from picamera import PiCamera
import io

camera = PiCamera()
camera.resolution = (600, 600)

while True:
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    image = Image.open(stream)

    # Run detection
    detections, _ = detector.detect_from_image(image)

    # Display results
    # ...
```

## Performance Benchmarks

Expected performance on Raspberry Pi 4 (4GB RAM):

| Model | Size | Inference Time | mAP@0.5 |
|-------|------|----------------|---------|
| MobileNetV2 (1.0) Quantized | ~600KB | ~150ms | TBD* |
| MobileNetV2 (0.75) Quantized | ~400KB | ~100ms | TBD* |
| MobileNetV2 (0.5) Quantized | ~300KB | ~70ms | TBD* |

*TBD: Train model to get actual mAP scores

## File Structure

```
models/
├── configs/
│   └── die_detection_mobilenet_retinanet.yaml  # Training config
├── scripts/
│   ├── prepare_die_dataset.py                  # Dataset preparation
│   ├── train_die_detector.sh                   # Training script
│   ├── export_to_tflite.sh                     # TFLite export
│   └── inference_rpi.py                        # Raspberry Pi inference
├── label_studio/
│   ├── 600x600/COCO/                           # Original dataset
│   └── tfrecords/                              # Processed TFRecords
└── trained_models/
    └── die_detector_mobilenet/                 # Trained model outputs
        ├── checkpoints/
        ├── exported/
        └── tflite/
            ├── die_detector_quantized.tflite
            └── die_detector_float32.tflite
```

## Next Steps

1. **Train the model** with your current 142 images
2. **Evaluate performance** on validation set
3. **Test on Raspberry Pi** with real dice
4. **Collect more data** based on failure cases
5. **Retrain** with expanded dataset
6. **Optimize** model size if needed

## Support

For issues related to:
- **TensorFlow Models**: See [official/README.md](official/README.md)
- **TFLite Runtime**: https://www.tensorflow.org/lite/guide/python
- **Raspberry Pi Setup**: https://www.raspberrypi.org/documentation/

## License

This project uses TensorFlow Model Garden (Apache 2.0 License). Your custom dataset and scripts are yours to license as you wish.
