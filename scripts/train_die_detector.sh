#!/bin/bash
# Training script for die detection model optimized for Raspberry Pi 4
# This script trains a MobileNetV2 + RetinaNet model on the custom dice dataset

set -e  # Exit on error

echo "========================================="
echo "Die Detection Model Training"
echo "Target: Raspberry Pi 4 deployment"
echo "========================================="

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/die_detection_mobilenet_retinanet.yaml"
MODEL_DIR="${PROJECT_ROOT}/trained_models/die_detector_mobilenet"
MODE="train_and_eval"

# Check if dataset needs to be prepared
if [ ! -d "${PROJECT_ROOT}/label_studio/tfrecords" ]; then
    echo ""
    echo "Step 1: Preparing dataset (converting COCO to TFRecord)..."
    python3 "${PROJECT_ROOT}/scripts/prepare_die_dataset.py" \
        --coco_json "${PROJECT_ROOT}/label_studio/600x600/COCO/result.json" \
        --image_dir "${PROJECT_ROOT}/label_studio/600x600/COCO/images" \
        --output_dir "${PROJECT_ROOT}/label_studio/tfrecords" \
        --train_split 0.8
    echo "✓ Dataset preparation complete"
else
    echo ""
    echo "Step 1: Dataset already prepared (skipping)"
fi

# Create model directory
mkdir -p "${MODEL_DIR}"

echo ""
echo "Step 2: Starting training..."
echo "  Config: ${CONFIG_FILE}"
echo "  Output: ${MODEL_DIR}"
echo ""

# Train the model
python3 "${PROJECT_ROOT}/official/vision/train.py" \
    --experiment=retinanet_mobile_coco \
    --mode="${MODE}" \
    --model_dir="${MODEL_DIR}" \
    --config_file="${CONFIG_FILE}" \
    2>&1 | tee "${MODEL_DIR}/training.log"

echo ""
echo "✓ Training complete!"
echo ""
echo "Model saved to: ${MODEL_DIR}"
echo "Training log: ${MODEL_DIR}/training.log"
echo ""
echo "Next steps:"
echo "  1. Evaluate: bash scripts/evaluate_die_detector.sh"
echo "  2. Export to TFLite: bash scripts/export_to_tflite.sh"
echo "  3. Deploy to Raspberry Pi 4"
