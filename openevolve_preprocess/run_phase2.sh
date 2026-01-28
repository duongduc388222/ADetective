#!/bin/bash
#
# Run Phase 2: Code Evolution
#
# This phase evolves the FULL PREPROCESSING CODE, allowing
# the LLM to invent new preprocessing strategies.
#
# Prerequisites:
#   1. Run setup.sh to create virtual environment
#   2. Run create_proxy_dataset.py to create proxy data
#   3. Set GEMINI_API_KEY environment variable
#   4. (Recommended) Run Phase 1 first for optimized starting point
#
# Usage:
#   ./run_phase2.sh [iterations] [output_dir]
#
# Examples:
#   ./run_phase2.sh              # Default: 100 iterations
#   ./run_phase2.sh 200          # 200 iterations
#   ./run_phase2.sh 100 ./my_output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Arguments
ITERATIONS="${1:-100}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/output/phase2_results}"

echo "========================================"
echo "OpenEvolve Phase 2: Code Evolution"
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

# Check if Phase 1 results exist and suggest using them
PHASE1_BEST="$SCRIPT_DIR/output/phase1_results/best_programs/program_0.py"
if [ -f "$PHASE1_BEST" ]; then
    echo "Found Phase 1 best result at: $PHASE1_BEST"
    echo "Consider copying it to candidates/baseline_code.py as starting point."
    echo ""
fi

echo ""
echo "Starting code evolution..."
echo "========================================"
echo ""
echo "This phase will:"
echo "  1. Evolve full preprocessing code (can modify any logic)"
echo "  2. Execute preprocessing on proxy data"
echo "  3. Evaluate using frozen CellFM backbone"
echo ""
echo "Each iteration takes ~1-3 minutes"
echo ""

# Run OpenEvolve
cd "$SCRIPT_DIR"

openevolve \
    candidates/baseline_code.py \
    evaluator.py \
    --config config_phase2.yaml \
    --iterations "$ITERATIONS" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Phase 2 Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Best preprocessing code: $OUTPUT_DIR/best_programs/"
echo "Evolution log: $OUTPUT_DIR/evolution_history.json"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review best preprocessing code:"
echo "     cat $OUTPUT_DIR/best_programs/program_0.py"
echo ""
echo "  2. Validate on full dataset:"
echo "     python scripts/train_cellfm.py \\"
echo "       --train-path results/processed/train.h5ad \\"
echo "       --val-path results/processed/val.h5ad \\"
echo "       --preprocessing-script $OUTPUT_DIR/best_programs/program_0.py"
echo ""
echo "  3. Compare top 3 candidates on full training"
echo ""
