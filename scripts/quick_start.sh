#!/bin/bash
# Quick start script for die detection training
# This script runs all steps in sequence

set -e

echo "========================================"
echo "Die Detection - Complete Pipeline"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Prepare your dataset"
echo "  2. Train the MobileNetV2 model"
echo "  3. Export to TensorFlow Lite"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Step 1: Prepare dataset
echo ""
echo "========================================="
echo "STEP 1/3: Preparing Dataset"
echo "========================================="
bash scripts/train_die_detector.sh

# Step 2 is included in train_die_detector.sh (training)

# Step 3: Export to TFLite
echo ""
echo "========================================="
echo "STEP 3/3: Exporting to TensorFlow Lite"
echo "========================================="
bash scripts/export_to_tflite.sh

echo ""
echo "========================================="
echo "COMPLETE!"
echo "========================================="
echo ""
echo "Your model is ready for Raspberry Pi 4!"
echo ""
echo "Files to copy to Raspberry Pi:"
echo "  - trained_models/die_detector_mobilenet/tflite/die_detector_quantized.tflite"
echo "  - label_studio/tfrecords/label_map.pbtxt"
echo "  - scripts/inference_rpi.py"
echo ""
echo "See DIE_DETECTION_README.md for deployment instructions"
