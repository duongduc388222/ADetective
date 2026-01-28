#!/bin/bash
#
# Run Phase 0: Meta-Prompt Evolution
#
# This phase evolves PROMPT CONFIGURATIONS that guide code generation.
# The optimized prompts will be used in Phase 1 and Phase 2.
#
# Prerequisites:
#   1. Run setup.sh to create virtual environment
#   2. Run create_proxy_dataset.py to create proxy data
#   3. Set GEMINI_API_KEY environment variable
#
# Usage:
#   ./run_phase0.sh [iterations] [output_dir]
#
# Examples:
#   ./run_phase0.sh              # Default: 30 iterations
#   ./run_phase0.sh 50           # 50 iterations
#   ./run_phase0.sh 30 ./my_output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Arguments
ITERATIONS="${1:-30}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/output/phase0_results}"

echo "========================================"
echo "OpenEvolve Phase 0: Meta-Prompt Evolution"
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
echo "Starting meta-prompt evolution..."
echo "========================================"
echo ""
echo "This phase will:"
echo "  1. Evolve prompt configurations (system_message + template_variations)"
echo "  2. Generate preprocessing code from evolved prompts"
echo "  3. Evaluate code quality using frozen CellFM backbone"
echo ""
echo "Each iteration takes ~2-5 minutes (LLM call + code generation + evaluation)"
echo ""

# Run OpenEvolve
cd "$SCRIPT_DIR"

openevolve \
    prompts/baseline_prompt_config.yaml \
    prompt_evaluator.py \
    --config config_phase0.yaml \
    --iterations "$ITERATIONS" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Phase 0 Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Best prompt configs: $OUTPUT_DIR/best_programs/"
echo "Evolution log: $OUTPUT_DIR/evolution_history.json"
echo ""
echo "Next steps:"
echo ""
echo "  1. Review best prompt config:"
echo "     cat $OUTPUT_DIR/best_programs/program_0.yaml"
echo ""
echo "  2. (Optional) Update config_phase1.yaml with evolved prompt"
echo ""
echo "  3. Run Phase 1:"
echo "     ./run_phase1.sh"
echo ""
