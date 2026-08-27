#!/bin/bash

# Resolve the tree root from this script's own location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANAGENT_ROOT="$(cd "$HERE/.." && pwd)"

# =============================================================================
# Table/Figure Analysis Evaluation (rule-based metrics)
# =============================================================================

# Set threading limits
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$HF_HOME/metrics}"
# PYTHONPATH is not needed: entry points bootstrap sys.path themselves.

# Exit on any error
set -e
GPU_ID="${GPU_ID:-0,1}"
NUM_WORKERS=8  # Set to number of CPU cores you want to use (e.g., 8, 16)

# =============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES AS NEEDED
# =============================================================================

# Required: Path to inference output results (JSON file)
INFER_OUTPUT="${ANAGENT_ROOT}/anagent_output/local/anabench/5_expert_turn/batch/batch_results.json"

# Output directory will be automatically set to the input JSON file's directory
# OUTPUT_DIR will be derived from INFER_OUTPUT path

# Optional: Cache directory for models and NLTK data
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

# Optional: Device (kept for compatibility, not actively used in evaluation)
DEVICE="cuda"

# Additional Python arguments (modify as needed)
EXTRA_ARGS=""

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-output)  INFER_OUTPUT="$2"; shift 2 ;;
        --cache-dir)     CACHE_DIR="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --semantic-encoder) EXTRA_ARGS="$EXTRA_ARGS --semantic-encoder $2"; shift 2 ;;
        --help)
            echo "Table/Figure Analysis Evaluation (rule-based metrics)"
            echo ""
            echo "Usage: bash run_eval.sh [options]"
            echo ""
            echo "  --infer-output <path>       Inference output JSON to evaluate"
            echo "  --semantic-encoder <name>   scibert | bertscore"
            echo "  --cache-dir <dir>           Model/NLTK cache directory"
            echo "  --device <device>           Device (default: $DEVICE)"
            echo "  --num-workers <n>           Parallel workers (default: $NUM_WORKERS)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

echo "============================================================="
echo "Table/Figure Analysis Evaluation (rule-based metrics)"
echo "============================================================="

# Check if required files exist
if [ ! -f "$INFER_OUTPUT" ]; then
    echo "Error: Inference output file not found: $INFER_OUTPUT"
    echo "Please update the INFER_OUTPUT variable with the correct path."
    exit 1
fi

# Automatically set output directory to the same directory as input JSON file
OUTPUT_DIR=$(dirname "$INFER_OUTPUT")
echo "Output directory automatically set to: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "Configuration:"
echo "  - Inference Output: $INFER_OUTPUT"
echo "  - Output Directory: $OUTPUT_DIR"
echo "  - Cache Directory:  $CACHE_DIR"
echo "  - Num Workers:      $NUM_WORKERS"
echo "  - Device:           $DEVICE"
echo ""

# Run the evaluation
echo "Starting evaluation..."
CUDA_VISIBLE_DEVICES=$GPU_ID python "$ANAGENT_ROOT/evaluate/run_eval.py" \
    --infer-output "$INFER_OUTPUT" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    $EXTRA_ARGS

echo ""
echo "============================================================="
echo "Evaluation completed successfully!"
echo "============================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

INFER_BASENAME=$(basename "$INFER_OUTPUT" .json)
EVAL_SUMMARY="$OUTPUT_DIR/${INFER_BASENAME}_eval_summary.json"

if [ -f "$EVAL_SUMMARY" ]; then
    echo "Summary file: $EVAL_SUMMARY"
fi

echo ""