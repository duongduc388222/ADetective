#!/bin/bash
#
# Setup virtual environment for OpenEvolve preprocessing optimization
#
# Usage:
#   ./setup.sh
#
# This script creates a Python virtual environment and installs
# all required dependencies for the three-phase evolution pipeline.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "OpenEvolve Preprocessing Setup"
echo "========================================"
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Create virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install openevolve
pip install scanpy anndata
pip install torch
pip install scikit-learn
pip install numpy pandas
pip install scipy
pip install openai  # For LLM API calls
pip install pyyaml  # For YAML parsing

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source $SCRIPT_DIR/venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Create proxy dataset:"
echo "     python $PROJECT_ROOT/scripts/create_proxy_dataset.py \\"
echo "       --input-path /path/to/raw/SEAAD.h5ad \\"
echo "       --output-path $SCRIPT_DIR/data/proxy_raw.h5ad"
echo ""
echo "  2. Set API key:"
echo "     export GEMINI_API_KEY='your-key'"
echo ""
echo "  3. Run Phase 0:"
echo "     ./run_phase0.sh"
echo ""
