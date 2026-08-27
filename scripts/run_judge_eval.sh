#!/bin/bash

# LLM-as-Judge Evaluation
# Usage: bash run_judge_eval.sh [--json-path <results.json>] [--model <model>]

# Resolve the tree root from this script's own location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANAGENT_ROOT="$(cd "$HERE/.." && pwd)"

# --- API keys: read at runtime from a git-ignored store; never hardcode ---
# Point KEY_DIR at a directory holding one file per credential.
KEY_DIR="${KEY_DIR:-$HOME/.anagent_keys}"
_read_key() { [ -f "$1" ] && tr -d '\n\r' < "$1"; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"


# ============================================================================
# Configuration
# ============================================================================
# Inference output to judge (the batch_results.json produced by run_anagent.sh).
JSON_PATH="${JSON_PATH:-./anagent_output/local/anabench/5_expert_turn/batch/batch_results.json}"

# JUDGE_MODE selects which judge runs:
#   answer            -> grades the final answer only
#   reason_and_answer -> grades reasoning and answer separately
JUDGE_MODE="answer"          # answer | reason_and_answer

# JUDGE_SOURCE selects the judge backend. The judge auto-detects from the model
# string prefix, so switching is purely a config change.
JUDGE_SOURCE="openai"        # openai | azure

if [[ "$JUDGE_SOURCE" == "azure" ]]; then
    MODEL_PATH="azure/gpt-4o"
    API_KEY="$(_read_key "$KEY_DIR/azure_api_key")"
    BASE_URL=""
    API_VERSION="2024-02-01"
else
    MODEL_PATH="openai/gpt-4.1-mini"
    API_KEY="$(_read_key "$KEY_DIR/openai_api_key")"
    BASE_URL=""
    API_VERSION="2024-02-01"
fi

# Judge generation settings. Temperature 0 for reproducible grading.
TEMPERATURE=0.0
TOP_P=1.0
MAX_TOKENS=4096
TIMEOUT=120
MAX_RETRIES=3

OUTPUT_DIR="./eval_outputs/judge"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json-path)   JSON_PATH="$2"; shift 2 ;;
        --model)       MODEL_PATH="$2"; shift 2 ;;
        --judge-source) JUDGE_SOURCE="$2"; shift 2 ;;
        --judge-mode)  JUDGE_MODE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --help)
            echo "LLM-as-Judge Evaluation"
            echo ""
            echo "Usage: bash run_judge_eval.sh [options]"
            echo ""
            echo "  --json-path <path>      Inference output to judge"
            echo "  --model <model>         Judge model (default: $MODEL_PATH)"
            echo "  --judge-source <src>    openai | azure"
            echo "  --judge-mode <mode>     answer | reason_and_answer"
            echo "  --temperature <float>   Judge temperature (default: $TEMPERATURE)"
            echo "  --max-retries <n>       Retries per instance (default: $MAX_RETRIES)"
            echo "  --output-dir <dir>      Output directory"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


# Validation
if [[ "$JUDGE_MODE" != "answer" && "$JUDGE_MODE" != "reason_and_answer" ]]; then
    echo "Error: --judge-mode must be 'answer' or 'reason_and_answer' (got: $JUDGE_MODE)"
    exit 1
fi

if [[ ! -f "$JSON_PATH" ]]; then
    echo "Error: inference output not found: $JSON_PATH"
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    echo "Error: no API key. Put one at \$KEY_DIR/openai_api_key (or azure_api_key)."
    echo "       KEY_DIR is currently: $KEY_DIR"
    exit 1
fi


echo "================================================================================"
echo "LLM-as-Judge Evaluation"
echo "================================================================================"
echo "Judge Mode:         $JUDGE_MODE"
echo "Judge Source:       $JUDGE_SOURCE"
echo "Judge Model:        $MODEL_PATH"
echo "Temperature:        $TEMPERATURE"
echo "Max Retries:        $MAX_RETRIES"
echo "Inference Output:   $JSON_PATH"
echo "Output Directory:   $OUTPUT_DIR"
echo "================================================================================"
echo ""

CMD_ARGS=(
    --model_path "$MODEL_PATH"
    --api_key "$API_KEY"
    --api_version "$API_VERSION"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_tokens "$MAX_TOKENS"
    --timeout "$TIMEOUT"
    --max_retries "$MAX_RETRIES"
    --json_path "$JSON_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
)

[[ -n "$BASE_URL" ]] && CMD_ARGS+=(--base_url "$BASE_URL")

if [[ "$JUDGE_MODE" == "reason_and_answer" ]]; then
    JUDGE_SCRIPT="$ANAGENT_ROOT/evaluate/llm_as_judge/llm_as_judge_reason_and_answer.py"
else
    JUDGE_SCRIPT="$ANAGENT_ROOT/evaluate/llm_as_judge/llm_as_judge_answer.py"
fi

python "$JUDGE_SCRIPT" "${CMD_ARGS[@]}"

echo ""
echo "Judge evaluation complete!"
