#!/bin/bash
#
# Run Phase 1: Parameter Evolution
#
# This phase evolves preprocessing PARAMETERS while keeping
# the code structure fixed.
#
# Prerequisites:
#   1. Run setup.sh to create virtual environment
#   2. Run create_proxy_dataset.py to create proxy data
#   3. Set GEMINI_API_KEY environment variable
#   4. (Optional) Run Phase 0 first for optimized prompts
#
# Usage:
#   ./run_phase1.sh [iterations] [output_dir]
#
# Examples:
#   ./run_phase1.sh              # Default: 50 iterations
#   ./run_phase1.sh 100          # 100 iterations
#   ./run_phase1.sh 50 ./my_output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Arguments
ITERATIONS="${1:-50}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/output/phase1_results}"

echo "========================================"
echo "OpenEvolve Phase 1: Parameter Evolution"
echo "========================================"
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Iterations: $ITERATIONS"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found!"
    echo "Run ./setup.sh first."
    exit 1
fi

# Check for proxy dataset
PROXY_PATH="$SCRIPT_DIR/data/proxy_raw.h5ad"
if [ ! -f "$PROXY_PATH" ]; then
    echo "ERROR: Proxy dataset not found at $PROXY_PATH"
    echo ""
    echo "Create it with:"
    echo "  python $PROJECT_ROOT/scripts/create_proxy_dataset.py \\"
    echo "    --input-path /path/to/SEAAD_raw.h5ad \\"
    echo "    --output-path $PROXY_PATH"
    exit 1
fi
echo "Proxy dataset: $PROXY_PATH"

# Check for genemap
GENEMAP_PATH="$PROJECT_ROOT/data/genemap.csv"
if [ ! -f "$GENEMAP_PATH" ]; then
    echo "ERROR: Genemap not found at $GENEMAP_PATH"
    exit 1
fi
echo "Genemap: $GENEMAP_PATH"

# Check for API key
if [ -z "$GEMINI_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: No API key found!"
    echo "Set GEMINI_API_KEY or OPENAI_API_KEY environment variable."
    echo ""
    echo "Example:"
    echo "  export GEMINI_API_KEY='your-key-here'"
    exit 1
fi

# Export paths for evaluator
export PROXY_DATA_PATH="$PROXY_PATH"
export GENEMAP_PATH="$GENEMAP_PATH"
export BACKBONE_PATH="$PROJECT_ROOT/examples/save/cellFM/CellFM_80M_weight.ckpt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting parameter evolution..."
echo "========================================"
echo ""
echo "This phase will:"
echo "  1. Evolve preprocessing parameters (min_cells, n_hvgs, etc.)"
echo "  2. Execute preprocessing on proxy data"
echo "  3. Evaluate using frozen CellFM backbone"
echo ""
echo "Each iteration takes ~1-2 minutes"
echo ""

# Run OpenEvolve
cd "$SCRIPT_DIR"

openevolve \
    candidates/baseline_params.py \
    evaluator.py \
    --config config_phase1.yaml \
    --iterations "$ITERATIONS" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Phase 1 Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Best parameters: $OUTPUT_DIR/best_programs/"
echo "Evolution log: $OUTPUT_DIR/evolution_history.json"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review best parameters:"
echo "     cat $OUTPUT_DIR/best_programs/program_0.py"
echo ""
echo "  2. Copy best result as Phase 2 starting point:"
echo "     cp $OUTPUT_DIR/best_programs/program_0.py candidates/baseline_code.py"
echo ""
echo "  3. Run Phase 2:"
echo "     ./run_phase2.sh"
echo ""
