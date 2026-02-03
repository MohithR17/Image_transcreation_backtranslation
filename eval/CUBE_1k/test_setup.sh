#!/bin/bash

# Test script to verify T2I CUBE setup
# This script checks if all necessary files and dependencies are in place

echo "=========================================="
echo "T2I CUBE Setup Verification"
echo "=========================================="
echo ""

# Change to the CUBE_1k directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "✓ Working directory: $(pwd)"
echo ""

# Check if dataset exists
echo "Checking dataset..."
if [ -f "data/cube_1k.json" ]; then
    CUBE_COUNT=$(python -c "import json; print(len(json.load(open('data/cube_1k.json'))))" 2>/dev/null || echo "Error")
    echo "✓ CUBE_1k dataset found ($CUBE_COUNT samples)"
else
    echo "✗ CUBE_1k dataset NOT found at data/cube_1k.json"
fi
echo ""

# Check if T2I models exist
echo "Checking T2I models..."
MODELS=("flux-dev" "qwen-image-2512" "flux-schnell" "sdxl")
for model in "${MODELS[@]}"; do
    MODEL_FILE="../../models/T2I/${model}.py"
    if [ -f "$MODEL_FILE" ]; then
        echo "✓ Model found: $model"
    else
        echo "✗ Model NOT found: $model"
    fi
done
echo ""

# Check if main script exists
echo "Checking main script..."
if [ -f "T2I_cube.py" ]; then
    echo "✓ T2I_cube.py found"
else
    echo "✗ T2I_cube.py NOT found"
fi
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
REQUIRED_PACKAGES=("torch" "diffusers" "transformers" "PIL")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package installed"
    else
        echo "✗ $package NOT installed"
    fi
done
echo ""

# Test if script can be imported
echo "Testing script syntax..."
if python -c "import sys; sys.path.insert(0, '../..'); exec(open('T2I_cube.py').read().replace('if __name__ == \"__main__\":', 'if False:'))" 2>/dev/null; then
    echo "✓ T2I_cube.py syntax is valid"
else
    echo "⚠ T2I_cube.py has syntax issues (may be import-related)"
fi
echo ""

# Check if output directory can be created
echo "Checking output directory permissions..."
if mkdir -p outputs/test_dir && rmdir outputs/test_dir 2>/dev/null; then
    echo "✓ Can create output directories"
else
    echo "✗ Cannot create output directories"
fi
echo ""

# Run help command to verify script works
echo "Testing script execution..."
if python T2I_cube.py --help > /dev/null 2>&1; then
    echo "✓ T2I_cube.py can be executed"
else
    echo "⚠ T2I_cube.py execution has issues (may need dependencies)"
fi
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "To run the evaluation:"
echo "  python T2I_cube.py --model flux-dev --debug"
echo ""
echo "To install missing dependencies:"
echo "  pip install -r requirements.txt"
echo ""
