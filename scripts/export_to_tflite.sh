#!/bin/bash
# Export trained model to TensorFlow Lite for Raspberry Pi 4 deployment
# This creates an optimized .tflite model file

set -e

echo "========================================="
echo "Export Die Detector to TensorFlow Lite"
echo "Target: Raspberry Pi 4"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/trained_models/die_detector_mobilenet"
EXPORT_DIR="${MODEL_DIR}/exported"
TFLITE_DIR="${MODEL_DIR}/tflite"

# Check if model exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory not found: ${MODEL_DIR}"
    echo "Please train the model first: bash scripts/train_die_detector.sh"
    exit 1
fi

# Create export directories
mkdir -p "${EXPORT_DIR}"
mkdir -p "${TFLITE_DIR}"

echo ""
echo "Step 1: Exporting to SavedModel format..."

python3 "${PROJECT_ROOT}/official/vision/serving/export_saved_model.py" \
    --experiment=retinanet_mobile_coco \
    --export_dir="${EXPORT_DIR}" \
    --checkpoint_path="${MODEL_DIR}/ckpt-latest" \
    --config_file="${PROJECT_ROOT}/configs/die_detection_mobilenet_retinanet.yaml" \
    --batch_size=1 \
    --input_image_size=600,600

echo "✓ SavedModel export complete"

echo ""
echo "Step 2: Converting to TensorFlow Lite..."

# Create TFLite conversion script
cat > "${TFLITE_DIR}/convert_to_tflite.py" << 'PYTHON_EOF'
import tensorflow as tf
import sys
from pathlib import Path

def convert_to_tflite(saved_model_dir, output_path, quantize=True):
    """Convert SavedModel to TFLite with optional quantization."""

    # Load the SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        # Enable dynamic range quantization for smaller model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Quantization: Enabled (Dynamic range)")
    else:
        print("Quantization: Disabled")

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Print model size
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    return tflite_model

if __name__ == '__main__':
    saved_model_dir = sys.argv[1]
    output_dir = sys.argv[2]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Converting to TFLite (quantized)...")
    convert_to_tflite(
        saved_model_dir,
        output_dir / 'die_detector_quantized.tflite',
        quantize=True
    )

    print("\nConverting to TFLite (float32)...")
    convert_to_tflite(
        saved_model_dir,
        output_dir / 'die_detector_float32.tflite',
        quantize=False
    )

    print("\n✓ TFLite conversion complete!")
PYTHON_EOF

python3 "${TFLITE_DIR}/convert_to_tflite.py" \
    "${EXPORT_DIR}/saved_model" \
    "${TFLITE_DIR}"

echo ""
echo "========================================="
echo "Export Complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  Quantized (recommended for RPi): ${TFLITE_DIR}/die_detector_quantized.tflite"
echo "  Float32 (higher accuracy):       ${TFLITE_DIR}/die_detector_float32.tflite"
echo "  Label map:                       ${PROJECT_ROOT}/label_studio/tfrecords/label_map.pbtxt"
echo ""
echo "Next steps:"
echo "  1. Copy .tflite file to your Raspberry Pi 4"
echo "  2. Install TFLite runtime: pip3 install tflite-runtime"
echo "  3. Use the inference script: python3 scripts/inference_rpi.py"
