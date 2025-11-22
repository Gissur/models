#!/bin/bash
# Setup virtual environment for die detection training

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================="
echo "Setting up Python Virtual Environment"
echo "========================================="
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists at: venv/"
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old venv..."
        rm -rf venv
    else
        echo "Using existing venv"
        echo ""
        echo "To activate:"
        echo "  source venv/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing TensorFlow and dependencies..."
pip install tensorflow>=2.12.0
pip install pillow numpy

echo ""
echo "Installing TensorFlow Model Garden requirements..."
pip install -r official/requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Virtual environment created at: venv/"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python3 scripts/verify_setup.py"
echo "  3. bash scripts/quick_start.sh"
