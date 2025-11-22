#!/usr/bin/env python3
"""
Inference script for die detection on Raspberry Pi 4 using TensorFlow Lite.
This script loads a .tflite model and performs real-time object detection.
"""

import argparse
import time
import numpy as np
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Please install Pillow: pip3 install Pillow")
    exit(1)


class DieDetector:
    """TFLite-based die detector for Raspberry Pi."""

    def __init__(self, model_path, label_map_path=None, confidence_threshold=0.5):
        """
        Initialize the detector.

        Args:
            model_path: Path to .tflite model file
            label_map_path: Path to label map file (optional)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold

        # Load labels
        self.labels = self._load_labels(label_map_path) if label_map_path else {}

        # Load TFLite model
        print(f"Loading model: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        print(f"Model loaded successfully")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Expected size: {self.input_width}x{self.input_height}")

    def _load_labels(self, label_map_path):
        """Load labels from label map file."""
        labels = {}
        try:
            with open(label_map_path, 'r') as f:
                current_id = None
                for line in f:
                    line = line.strip()
                    if line.startswith('id:'):
                        current_id = int(line.split(':')[1].strip())
                    elif line.startswith('name:'):
                        name = line.split(':')[1].strip().strip("'\"")
                        if current_id is not None:
                            labels[current_id] = name
                            current_id = None
        except Exception as e:
            print(f"Warning: Could not load labels: {e}")
        return labels

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array and original image
        """
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()

        # Resize to model input size
        image = image.resize((self.input_width, self.input_height))

        # Convert to numpy array and normalize
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        # Normalize to [0, 1] or [-1, 1] depending on model
        # For MobileNet, typically [0, 1]
        input_data = input_data / 255.0

        return input_data, original_image

    def detect(self, image_path):
        """
        Perform die detection on an image.

        Args:
            image_path: Path to image file

        Returns:
            List of detections, each containing:
                - bbox: [ymin, xmin, ymax, xmax] (normalized)
                - class_id: Class ID
                - class_name: Class name
                - confidence: Detection confidence
        """
        # Preprocess
        input_data, original_image = self.preprocess_image(image_path)

        # Run inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        # Get outputs
        # Different models may have different output formats
        # This is a typical format for object detection models
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding boxes
        class_ids = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class IDs
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence scores
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])  # Number of detections

        # Filter by confidence threshold
        detections = []
        for i in range(num_detections):
            if scores[i] >= self.confidence_threshold:
                detection = {
                    'bbox': boxes[i],  # [ymin, xmin, ymax, xmax]
                    'class_id': int(class_ids[i]),
                    'class_name': self.labels.get(int(class_ids[i]), f'Class {int(class_ids[i])}'),
                    'confidence': float(scores[i])
                }
                detections.append(detection)

        print(f"Inference time: {inference_time*1000:.1f}ms")
        print(f"Detections: {len(detections)}")

        return detections, original_image

    def visualize_detections(self, image, detections, output_path=None):
        """
        Draw bounding boxes on image.

        Args:
            image: PIL Image
            detections: List of detections from detect()
            output_path: Path to save annotated image (optional)

        Returns:
            Annotated PIL Image
        """
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Color map for different classes
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
            '#FF00FF', '#00FFFF', '#FF8800', '#8800FF'
        ]

        for det in detections:
            ymin, xmin, ymax, xmax = det['bbox']

            # Convert normalized coordinates to pixel coordinates
            left = int(xmin * width)
            right = int(xmax * width)
            top = int(ymin * height)
            bottom = int(ymax * height)

            # Choose color based on class
            color = colors[det['class_id'] % len(colors)]

            # Draw bounding box
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)

            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            draw.text((left, top - 25), label, fill=color, font=font)

        if output_path:
            image.save(output_path)
            print(f"Saved annotated image to: {output_path}")

        return image


def main():
    parser = argparse.ArgumentParser(description='Die detection inference on Raspberry Pi')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to .tflite model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file for inference'
    )
    parser.add_argument(
        '--labels',
        type=str,
        help='Path to label map file'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save annotated image'
    )

    args = parser.parse_args()

    # Create detector
    detector = DieDetector(
        args.model,
        label_map_path=args.labels,
        confidence_threshold=args.confidence
    )

    # Run detection
    detections, image = detector.detect(args.image)

    # Print results
    print("\n" + "="*50)
    print("Detection Results:")
    print("="*50)
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class_name']}")
        print(f"   Confidence: {det['confidence']:.3f}")
        print(f"   BBox: {det['bbox']}")

    if not detections:
        print("No detections found")

    # Visualize
    if args.output or True:  # Always create visualization
        output_path = args.output or 'detection_result.jpg'
        detector.visualize_detections(image, detections, output_path)

    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
