#!/bin/bash
# Setup script to ensure Python 3.10 is used for installation

set -e

echo "=========================================="
echo "Stock Forecast Project Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" != "3" ] || [ "$PYTHON_MINOR" != "10" ]; then
    echo "⚠️  Warning: Python version is $PYTHON_VERSION, but Python 3.10 is required"
    echo ""
    echo "For reproducibility, this project requires Python 3.10."
    echo ""
    echo "Options:"
    echo "  1. Use conda (recommended):"
    echo "     conda env create -f environment.yml"
    echo "     conda activate stock-forecast"
    echo ""
    echo "  2. Install Python 3.10 and use it:"
    echo "     python3.10 -m pip install -r requirements.txt"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Python version is $PYTHON_VERSION (correct)"
fi

echo ""
echo "Installing packages from requirements.txt..."
echo ""

python -m pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Verify installation with:"
echo "  python verify_versions.py"
echo ""

