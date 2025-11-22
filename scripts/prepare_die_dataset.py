#!/usr/bin/env python3
"""
Prepare the die detection dataset from Label Studio format to TF Object Detection API format.
This script converts the COCO JSON and creates TFRecord files for efficient training.
"""

import json
import os
import argparse
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np


def create_tf_example(image_path, annotations, image_id, category_id_to_name):
    """
    Create a tf.train.Example from image and annotations.

    Args:
        image_path: Path to the image file
        annotations: List of annotation dicts for this image
        image_id: ID of the image
        category_id_to_name: Mapping from category ID to category name

    Returns:
        tf.train.Example
    """
    # Read image
    with Image.open(image_path) as img:
        width, height = img.size
        img_bytes = img.tobytes()

    # Get image format
    img_format = image_path.suffix[1:].encode('utf8')  # Remove '.' from extension
    filename = image_path.name.encode('utf8')

    # Prepare annotation data
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    class_ids, class_names = [], []

    for ann in annotations:
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        x, y, w, h = bbox

        # Normalize coordinates
        xmins.append(x / width)
        xmaxs.append((x + w) / width)
        ymins.append(y / height)
        ymaxs.append((y + h) / height)

        category_id = ann['category_id']
        class_ids.append(category_id)
        class_names.append(category_id_to_name[category_id].encode('utf8'))

    # Create TensorFlow Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_id).encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_names)),
    }))

    return tf_example


def convert_coco_to_tfrecord(coco_json_path, image_dir, output_path, train_split=0.8):
    """
    Convert COCO format JSON to TFRecord files.

    Args:
        coco_json_path: Path to COCO format JSON file
        image_dir: Directory containing images
        output_path: Directory to save TFRecord files
        train_split: Fraction of data to use for training (rest is validation)
    """
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create category mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    print(f"Found {len(coco_data['categories'])} categories:")
    for cat_id, cat_name in category_id_to_name.items():
        print(f"  {cat_id}: {cat_name}")

    # Group annotations by image
    image_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(ann)

    # Create image ID to filename mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Split into train/val
    image_ids = list(image_to_annotations.keys())
    np.random.seed(42)
    np.random.shuffle(image_ids)

    split_idx = int(len(image_ids) * train_split)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]

    print(f"\nDataset split: {len(train_ids)} train, {len(val_ids)} validation")

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write train TFRecords
    train_writer = tf.io.TFRecordWriter(str(output_path / 'train.tfrecord'))
    for image_id in train_ids:
        img_info = image_id_to_info[image_id]
        image_path = Path(image_dir) / img_info['file_name']
        annotations = image_to_annotations[image_id]

        tf_example = create_tf_example(image_path, annotations, image_id, category_id_to_name)
        train_writer.write(tf_example.SerializeToString())
    train_writer.close()
    print(f"Created train.tfrecord with {len(train_ids)} images")

    # Write validation TFRecords
    val_writer = tf.io.TFRecordWriter(str(output_path / 'val.tfrecord'))
    for image_id in val_ids:
        img_info = image_id_to_info[image_id]
        image_path = Path(image_dir) / img_info['file_name']
        annotations = image_to_annotations[image_id]

        tf_example = create_tf_example(image_path, annotations, image_id, category_id_to_name)
        val_writer.write(tf_example.SerializeToString())
    val_writer.close()
    print(f"Created val.tfrecord with {len(val_ids)} images")

    # Save label map
    label_map_path = output_path / 'label_map.pbtxt'
    with open(label_map_path, 'w') as f:
        for cat_id, cat_name in category_id_to_name.items():
            f.write(f"item {{\n")
            f.write(f"  id: {cat_id}\n")
            f.write(f"  name: '{cat_name}'\n")
            f.write(f"}}\n\n")
    print(f"Created label map at {label_map_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert die detection dataset to TFRecord')
    parser.add_argument(
        '--coco_json',
        type=str,
        default='label_studio/600x600/COCO/result.json',
        help='Path to COCO JSON file'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='label_studio/600x600/COCO/images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='label_studio/tfrecords',
        help='Output directory for TFRecord files'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Fraction of data for training'
    )

    args = parser.parse_args()

    convert_coco_to_tfrecord(
        args.coco_json,
        args.image_dir,
        args.output_dir,
        args.train_split
    )

    print("\n✓ Dataset preparation complete!")
    print(f"  TFRecord files saved to: {args.output_dir}")
    print(f"  You can now train with these files")


if __name__ == '__main__':
    main()
