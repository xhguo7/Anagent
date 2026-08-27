#!/bin/bash

# Multi-Agent System Execution Script
# Usage: bash run_anagent.sh [options]

# Resolve the tree root from this script's own location, so the launcher works
# from any working directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANAGENT_ROOT="$(cd "$HERE/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
# GPU_ID=0,1,2,3

# Environment setup
# HuggingFace caches. Override these to point at fast/shared storage.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$HF_HOME/metrics}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- API keys: read at runtime from a git-ignored store; never hardcode ---
# Point KEY_DIR at a directory holding one file per credential.
KEY_DIR="${KEY_DIR:-$HOME/.anagent_keys}"
_read_key() { [ -f "$1" ] && tr -d '\n\r' < "$1"; }

# Google Custom Search (used by the `web` tool; optional)
# export HF_TOKEN="$(_read_key "$KEY_DIR/hf_hub_token")"   # for gated HF models
export GOOGLE_API_KEY="$(_read_key "$KEY_DIR/google_api_key")"
export GOOGLE_CSE_ID="$(_read_key "$KEY_DIR/google_cse_id")"


# ============================================================================
# Configuration
# ============================================================================
MODE="batch"                 # single | batch

# --- Models -----------------------------------------------------------------
# MODEL_SOURCE selects where the agents run. The engine layer auto-detects the
# backend from the model string prefix, so switching is purely a config change:
#   api   -> "openai/gpt-4.1-mini", "azure/...", "gemini/...", "vllm/..."
#   local -> a HuggingFace repo id or a local checkpoint directory
MODEL_SOURCE="local"         # local | api

if [[ "$MODEL_SOURCE" == "api" ]]; then
    SHARED_MODEL="openai/gpt-4.1-mini"
    API_KEY="$(_read_key "$KEY_DIR/openai_api_key")"
else
    MODEL_SIZE=4
    SHARED_MODEL="Qwen/Qwen3-VL-${MODEL_SIZE}B-Instruct"
    API_KEY=""
fi

# Per-agent overrides. Default: every agent shares one model.
TOOL_MODEL="${SHARED_MODEL}"
PLANNER_MODEL="${SHARED_MODEL}"
EXPERT_MODEL="${SHARED_MODEL}"
SOLVER_MODEL="${SHARED_MODEL}"
CRITIC_MODEL=""              # "" = no critic; otherwise a model string

BASE_URL=""                  # for Azure / self-hosted endpoints
API_VERSION="2024-08-01-preview"

# --- Generation -------------------------------------------------------------
TEMPERATURE="0.7"
MAX_OUTPUT_TOKENS="4096"
REPETITION_PENALTY=1.05      # 1.0 = disabled
PLANNER_TEMPERATURE=""       # blank = inherit TEMPERATURE
EXPERT_TEMPERATURE=""
SOLVER_TEMPERATURE=""
CRITIC_TEMPERATURE=""
TOOL_TEMPERATURE=""

# --- Single mode ------------------------------------------------------------
QUERY=""
CONTEXT=""
IMAGE=""
ANALYSIS_LEN=2048

# --- Batch mode -------------------------------------------------------------
DATA_SOURCE="local"          # local | hf
HF_DATASET=""
HF_SPLIT="test"
SEED=42                      # random seed
MAX_SAMPLES=""               # blank = no limit

DATA_VERSION="anabench"
TEST_DATA="${ANAGENT_ROOT}/dataset/${DATA_VERSION}/test.json"

# --- Run configuration ------------------------------------------------------
RESUME="true"
GIVE_CONTEXTS="true"
EXPERT_TURNS=5
CRITIC_TURNS=0               # 0 = off; N = N rounds of Solver<->Critic refinement
ENABLE_TOOLS="search,document,knowledge,vision,sandbox"   # or "all"
FEW_SHOT=0
SOLVER_AWARENESS="false"
ENABLE_REASONING="false"     # true = Solver also returns its reasoning
OUTPUT_DIR="./anagent_output/${MODEL_SOURCE}/${DATA_VERSION}/${EXPERT_TURNS}_expert_turn"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --tool-model)
            TOOL_MODEL="$2"
            shift 2
            ;;
        --planner-model)
            PLANNER_MODEL="$2"
            shift 2
            ;;
        --expert-model)
            EXPERT_MODEL="$2"
            shift 2
            ;;
        --solver-model)
            SOLVER_MODEL="$2"
            shift 2
            ;;
        --critic-model)
            CRITIC_MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --repetition-penalty)
            REPETITION_PENALTY="$2"
            shift 2
            ;;
        --max-output-tokens)
            MAX_OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --planner-temperature)
            PLANNER_TEMPERATURE="$2"
            shift 2
            ;;
        --expert-temperature)
            EXPERT_TEMPERATURE="$2"
            shift 2
            ;;
        --solver-temperature)
            SOLVER_TEMPERATURE="$2"
            shift 2
            ;;
        --critic-temperature)
            CRITIC_TEMPERATURE="$2"
            shift 2
            ;;
        --tool-temperature)
            TOOL_TEMPERATURE="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --api-version)
            API_VERSION="$2"
            shift 2
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        --context)
            CONTEXT="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --output-limit)
            ANALYSIS_LEN="$2"
            shift 2
            ;;
        --data-source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --hf-dataset)
            HF_DATASET="$2"
            shift 2
            ;;
        --hf-split)
            HF_SPLIT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --critic-turns)
            CRITIC_TURNS="$2"
            shift 2
            ;;
        --expert-turns)
            EXPERT_TURNS="$2"
            shift 2
            ;;
        --enable-tools)
            ENABLE_TOOLS="$2"
            shift 2
            ;;
        --few-shot)
            FEW_SHOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        --give-contexts)
            GIVE_CONTEXTS="true"
            shift
            ;;
        --self-awareness)
            SOLVER_AWARENESS="true"
            shift
            ;;
        --enable-reasoning)
            ENABLE_REASONING="true"
            shift
            ;;
        --help)
            echo "Multi-Agent System Execution Script"
            echo ""
            echo "Usage: bash run_anagent.sh [options]"
            echo ""
            echo "Mode:"
            echo "  --mode <single|batch>           Inference mode (required)"
            echo ""
            echo "Model Configuration:"
            echo "  --tool-model <model>            Tool model (default: openai/gpt-4.1-mini)"
            echo "  --planner-model <model>         Planner model (default: openai/gpt-4.1-mini)"
            echo "  --expert-model <model>          Expert model (default: openai/gpt-4.1-mini)"
            echo "  --solver-model <model>          Solver model (default: openai/gpt-4o)"
            echo "  --critic-model <model>          Critic model (default: None)"
            echo ""
            echo "Generation Parameters:"
            echo "  --temperature <float>           Sampling temperature for all agents (default: 0.7)"
            echo "  --max-output-tokens <int>       Maximum output tokens (default: 4096)"
            echo "  --planner-temperature <float>   Override temperature for Planner"
            echo "  --expert-temperature <float>    Override temperature for Expert"
            echo "  --solver-temperature <float>    Override temperature for Solver"
            echo "  --critic-temperature <float>    Override temperature for Critic"
            echo "  --tool-temperature <float>      Override temperature for tools"
            echo ""
            echo "API Configuration:"
            echo "  --api-key <key>                 API key"
            echo "  --base-url <url>                API base URL (for Azure)"
            echo "  --api-version <version>         API version"
            echo ""
            echo "Single Mode:"
            echo "  --query <query>                 Input query (required for single mode)"
            echo "  --context <text>                Optional context"
            echo "  --image <path>                  Optional image path"
            echo ""
            echo "Batch Mode:"
            echo "  --data-source <local|hf>        Data source (default: local)"
            echo "  --test-data <path>              Path to test data JSON"
            echo "  --hf-dataset <name>             HuggingFace dataset name"
            echo "  --hf-split <split>              HuggingFace split (default: test)"
            echo "  --seed <int>                    Random seed"
            echo "  --repetition-penalty <float>    Penalty for repeated tokens (default: $REPETITION_PENALTY)"
            echo "  --enable-reasoning              Solver also returns its reasoning"
            echo "  --max-samples <n>               Max sample limit for this run"
            echo ""
            echo "System Configuration:"
            echo "  --expert-turns <n>              Expert retrieval turns (default: 1)"
            echo "  --critic-turns <n>              Solver<->Critic refinement rounds (default: 0 = off)"
            echo "  --enable-tools <tools>          Tools to enable (default: all)"
            echo "  --output-dir <dir>              Output directory"
            echo ""
            echo "Examples:"
            echo "  # Single inference"
            echo "  bash run_anagent.sh --mode single --query \"What is CRISPR?\""
            echo ""
            echo "  # Batch inference"
            echo "  bash run_anagent.sh --mode batch --test-data ./data/test.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validation
if [[ "$MODE" == "single" && -z "$QUERY" ]]; then
    echo "Error: --query is required for single mode"
    exit 1
fi

if [[ "$MODE" == "batch" && -z "$TEST_DATA" && -z "$HF_DATASET" ]]; then
    echo "Error: --test-data or --hf-dataset is required for batch mode"
    exit 1
fi

if [[ "$MODE" == "batch" && "$DATA_SOURCE" == "local" && ! -f "$TEST_DATA" ]]; then
    echo "Error: test data not found: $TEST_DATA"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "Multi-Agent System Configuration"
echo "================================================================================"
echo "Mode:               $MODE"
echo "Tool Model:         $TOOL_MODEL"
echo "Planner Model:      $PLANNER_MODEL"
echo "Expert Model:       $EXPERT_MODEL"
echo "Solver Model:       $SOLVER_MODEL"
if [[ -n "$CRITIC_MODEL" ]]; then
    echo "Critic Model:       $CRITIC_MODEL"
else
    echo "Critic Model:       None"
fi
echo "Temperature:        $TEMPERATURE"
echo "Max Output Tokens:  $MAX_OUTPUT_TOKENS"
[[ -n "$PLANNER_TEMPERATURE" ]] && echo "Planner Temp:       $PLANNER_TEMPERATURE"
[[ -n "$EXPERT_TEMPERATURE" ]] && echo "Expert Temp:        $EXPERT_TEMPERATURE"
[[ -n "$SOLVER_TEMPERATURE" ]] && echo "Solver Temp:        $SOLVER_TEMPERATURE"
[[ -n "$CRITIC_TEMPERATURE" ]] && echo "Critic Temp:        $CRITIC_TEMPERATURE"
[[ -n "$TOOL_TEMPERATURE" ]] && echo "Tool Temp:          $TOOL_TEMPERATURE"
echo "Expert Turns:       $EXPERT_TURNS"
echo "Critic Turns:       $CRITIC_TURNS"
echo "Enabled Tools:      $ENABLE_TOOLS"
echo "K-Shot Inference:   k = $FEW_SHOT"
echo "Self-Awareness:     $SOLVER_AWARENESS"
echo "Reasoning:          $ENABLE_REASONING"
echo "Give Contexts:      $GIVE_CONTEXTS"
echo "Output Directory:   $OUTPUT_DIR"
echo "Resume:             $RESUME"

if [[ "$MODE" == "single" ]]; then
    echo "Query:              $QUERY"
    [[ -n "$CONTEXT" ]] && echo "Context:            ${CONTEXT:0:50}..."
    [[ -n "$IMAGE" ]] && echo "Image:              $IMAGE"
    [[ -n "$ANALYSIS_LEN" ]] && echo "Analysis length:    $ANALYSIS_LEN"
elif [[ "$MODE" == "batch" ]]; then
    echo "Data Source:        $DATA_SOURCE"
    [[ -n "$TEST_DATA" ]] && echo "Test Data:          $TEST_DATA"
    [[ -n "$HF_DATASET" ]] && echo "HF Dataset:         $HF_DATASET"
fi

echo "================================================================================"

# Build Python command
CMD_ARGS=(
    --mode "$MODE"
    --tool-model "$TOOL_MODEL"
    --planner-model "$PLANNER_MODEL"
    --expert-model "$EXPERT_MODEL"
    --solver-model "$SOLVER_MODEL"
    --temperature "$TEMPERATURE"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --expert-turns "$EXPERT_TURNS"
    --critic-turns "$CRITIC_TURNS"
    --output-dir "$OUTPUT_DIR"
    --repetition-penalty "$REPETITION_PENALTY"
)

[[ -n "$API_KEY" ]] && CMD_ARGS+=(--api-key "$API_KEY")
[[ -n "$BASE_URL" ]] && CMD_ARGS+=(--base-url "$BASE_URL")
[[ -n "$API_VERSION" ]] && CMD_ARGS+=(--api-version "$API_VERSION")
[[ -n "$CRITIC_MODEL" ]] && CMD_ARGS+=(--critic-model "$CRITIC_MODEL")
[[ -n "$PLANNER_TEMPERATURE" ]] && CMD_ARGS+=(--planner-temperature "$PLANNER_TEMPERATURE")
[[ -n "$EXPERT_TEMPERATURE" ]] && CMD_ARGS+=(--expert-temperature "$EXPERT_TEMPERATURE")
[[ -n "$SOLVER_TEMPERATURE" ]] && CMD_ARGS+=(--solver-temperature "$SOLVER_TEMPERATURE")
[[ -n "$CRITIC_TEMPERATURE" ]] && CMD_ARGS+=(--critic-temperature "$CRITIC_TEMPERATURE")
[[ -n "$TOOL_TEMPERATURE" ]] && CMD_ARGS+=(--tool-temperature "$TOOL_TEMPERATURE")
[[ -n "$SEED" ]] && CMD_ARGS+=(--seed "$SEED")
[[ -n "$MAX_SAMPLES" ]] && CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
[[ -n "$ENABLE_TOOLS" ]] && CMD_ARGS+=(--enable-tools "$ENABLE_TOOLS")
[[ -n "$FEW_SHOT" ]] && CMD_ARGS+=(--few-shot "$FEW_SHOT")
[[ "$RESUME" == "true" ]] && CMD_ARGS+=(--resume)
[[ "$GIVE_CONTEXTS" == "true" ]] && CMD_ARGS+=(--give-contexts)
[[ "$SOLVER_AWARENESS" == "true" ]] && CMD_ARGS+=(--solver-awareness)
[[ "$ENABLE_REASONING" == "true" ]] && CMD_ARGS+=(--enable-reasoning)

# Mode-specific arguments
if [[ "$MODE" == "single" ]]; then
    CMD_ARGS+=(--query "$QUERY")
    [[ -n "$CONTEXT" ]] && CMD_ARGS+=(--context "$CONTEXT")
    [[ -n "$IMAGE" ]] && CMD_ARGS+=(--image "$IMAGE")
    [[ -n "$ANALYSIS_LEN" ]] && CMD_ARGS+=(--output-limit "$ANALYSIS_LEN")
elif [[ "$MODE" == "batch" ]]; then
    CMD_ARGS+=(--data-source "$DATA_SOURCE")
    [[ -n "$TEST_DATA" ]] && CMD_ARGS+=(--test-data "$TEST_DATA")
    [[ -n "$HF_DATASET" ]] && CMD_ARGS+=(--hf-dataset "$HF_DATASET")
    [[ -n "$HF_SPLIT" ]] && CMD_ARGS+=(--hf-split "$HF_SPLIT")
fi

# Run
echo ""
echo "Running multi-agent system..."
python "$ANAGENT_ROOT/anagent/run_anagent.py" "${CMD_ARGS[@]}"

echo ""
echo "Execution complete!"