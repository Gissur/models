#!/usr/bin/env python3
"""
Verify that the die detection training setup is correctly configured.
Checks dependencies, dataset, and configuration files.
"""

import sys
import json
from pathlib import Path
import subprocess


def check_python_version():
    """Check Python version."""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  Python {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print(f"  Python {version.major}.{version.minor} ✗ (Need 3.7+)")
        return False


def check_dependencies():
    """Check required Python packages."""
    print("\n✓ Checking dependencies...")
    required = ['tensorflow', 'PIL', 'numpy']
    missing = []

    for package in required:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"  {package} ✓")
        except ImportError:
            print(f"  {package} ✗ (Not installed)")
            missing.append(package)

    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip3 install {' '.join(missing)}")
        return False
    return True


def check_dataset():
    """Check dataset files."""
    print("\n✓ Checking dataset...")
    project_root = Path(__file__).parent.parent

    # Check COCO JSON
    coco_json = project_root / 'label_studio/600x600/COCO/result.json'
    if not coco_json.exists():
        print(f"  COCO JSON ✗ (Not found: {coco_json})")
        return False

    # Load and validate JSON
    try:
        with open(coco_json, 'r') as f:
            data = json.load(f)

        num_images = len(data.get('images', []))
        num_annotations = len(data.get('annotations', []))
        num_categories = len(data.get('categories', []))

        print(f"  COCO JSON ✓")
        print(f"    Images: {num_images}")
        print(f"    Annotations: {num_annotations}")
        print(f"    Categories: {num_categories}")

        # Check image directory
        image_dir = project_root / 'label_studio/600x600/COCO/images'
        if image_dir.exists():
            num_image_files = len(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
            print(f"  Image directory ✓ ({num_image_files} files)")
        else:
            print(f"  Image directory ✗")
            return False

        # Warnings
        if num_images < 50:
            print(f"  ⚠ Warning: Only {num_images} images. Recommend 500+ for production.")
        if num_categories != 7:
            print(f"  ⚠ Warning: Expected 7 categories, found {num_categories}")

        return True

    except Exception as e:
        print(f"  COCO JSON ✗ (Error loading: {e})")
        return False


def check_config_file():
    """Check training configuration."""
    print("\n✓ Checking configuration...")
    project_root = Path(__file__).parent.parent
    config_file = project_root / 'configs/die_detection_mobilenet_retinanet.yaml'

    if not config_file.exists():
        print(f"  Config file ✗ (Not found: {config_file})")
        return False

    print(f"  Config file ✓")
    return True


def check_scripts():
    """Check training scripts."""
    print("\n✓ Checking scripts...")
    project_root = Path(__file__).parent.parent
    scripts = [
        'scripts/prepare_die_dataset.py',
        'scripts/train_die_detector.sh',
        'scripts/export_to_tflite.sh',
        'scripts/inference_rpi.py',
        'scripts/quick_start.sh'
    ]

    all_exist = True
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            print(f"  {script} ✓")
        else:
            print(f"  {script} ✗")
            all_exist = False

    return all_exist


def check_tfmodels():
    """Check TensorFlow Model Garden structure."""
    print("\n✓ Checking TensorFlow Model Garden...")
    project_root = Path(__file__).parent.parent

    critical_paths = [
        'official/vision/train.py',
        'official/vision/configs/retinanet.py',
        'official/requirements.txt'
    ]

    all_exist = True
    for path in critical_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"  {path} ✓")
        else:
            print(f"  {path} ✗")
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("="*60)
    print("Die Detection Setup Verification")
    print("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Config File", check_config_file),
        ("Training Scripts", check_scripts),
        ("TensorFlow Model Garden", check_tfmodels)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error checking {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")

    print("="*60)

    if all_passed:
        print("\n✓ All checks passed! You're ready to train.")
        print("\nNext step:")
        print("  bash scripts/quick_start.sh")
        print("\nOr step-by-step:")
        print("  python3 scripts/prepare_die_dataset.py")
        print("  bash scripts/train_die_detector.sh")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
