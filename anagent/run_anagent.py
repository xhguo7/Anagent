"""
Multi-agent system entry point.
"""

import os
import sys
import json
import copy
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List

_ANAGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ANAGENT_ROOT)
from utils.logger import setup_logger
logger = setup_logger(__name__)

from utils.json_util import read_from_json, save_to_json
from anagent.loader.data_loader import load_test_data
from anagent.agents.orchestrator import MultiAgentOrchestrator


def run_single_inference(args):
    """Run single inference mode."""
    logger.info("="*100)
    logger.info("SINGLE INFERENCE MODE")
    logger.info("="*100)
    
    # Prepare engine kwargs
    engine_kwargs = {}
    if args.api_key:
        engine_kwargs["api_key"] = args.api_key
    if args.base_url:
        engine_kwargs["base_url"] = args.base_url
    if args.api_version:
        engine_kwargs["api_version"] = args.api_version
    engine_kwargs["repetition_penalty"] = args.repetition_penalty
    
    # Google search API
    if args.google_api_key:
        engine_kwargs["google_api_key"] = args.google_api_key
    if args.google_api_key:
        engine_kwargs["google_cse_id"] = args.google_cse_id
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(
        args=args,
        engine_kwargs=engine_kwargs,
        enable_tools=args.enable_tools.split(',') if args.enable_tools else None,
    )
    
    # Run
    single_task = {
        "query": args.query,
        "context": args.context,
        "image": args.image,
        "source": {"url": args.context},
    }
    result = orchestrator.run(
        task=single_task,
        expert_turns=args.expert_turns
    )
    
    # Save result
    output_file = os.path.join(args.output_dir, "single_result.json")
    orchestrator.save_result(result, output_file)
    
    # Print solution
    logger.info("="*100)
    logger.info("FINAL SOLUTION")
    logger.info("="*100)
    logger.info(result["answer"])
    logger.info("="*100)


def resume_inference(file_path: str) -> List:
    """Checks if the given output_file exists."""
    if os.path.exists(file_path):
        try:
            return read_from_json(file_path)
        except json.JSONDecodeError:
            return []
    else:
        return []


def run_batch_inference(args):
    """Run batch inference mode."""
    logger.info("="*100)
    logger.info("BATCH INFERENCE MODE")
    logger.info("="*100)
    
    # Load test data
    test_data = load_test_data(args)    
    
    # Prepare engine kwargs
    engine_kwargs = {}
    if args.api_key:
        engine_kwargs["api_key"] = args.api_key
    if args.base_url:
        engine_kwargs["base_url"] = args.base_url
    if args.api_version:
        engine_kwargs["api_version"] = args.api_version
    engine_kwargs["repetition_penalty"] = args.repetition_penalty
    
    # Initialize orchestrator
    logger.info("Running multi-agent system...")
    orchestrator = MultiAgentOrchestrator(
        args=args,
        engine_kwargs=engine_kwargs,
        enable_tools=args.enable_tools.split(',') if args.enable_tools else None
    )
    
    # Run inference
    if args.resume:
        # Resume from previous
        output_file = os.path.join(args.output_dir, "batch_results.json")
        results = resume_inference(output_file)
        if "results" in results:
            results = copy.deepcopy(results["results"])
        if len(results) > 0:
            existing_ids = [dd["id"] for dd in results]
        else:
            existing_ids = []
    else:
        # Overwrite all
        results = []
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    
    for i, sample in enumerate(tqdm(test_data, desc="Running Inference", total=len(test_data))):
        logger.info(f"\n{'='*70}\nRunning inference on sample {i+1}/{len(test_data)}\nQuery:\n{sample['query']}\n{'='*70}\n")

        # Check if resume
        if args.resume and len(results) > 0 and sample["id"] in existing_ids:
            continue

        # Run inference
        result = orchestrator.run(
            task=sample,
            expert_turns=args.expert_turns
        )
        
        # Add ground truth if available
        if "gt_answer" in sample:
            result["gt_answer"] = sample["gt_answer"]

        # Update and save
        sample["model_infer"] = {
            "answer": result["answer"],
            "reason": result["reason"],
            "confidence": result["confidence"],
            "refined_answer": result.get("refined_answer", None),
            "refined_reason": result.get("refined_reason", None),
            "refined_confidence": result.get("refined_confidence", None),
            "total_duration": result["total_duration"],
            "timestamp": result["timestamp"],
        }
        results.append(sample)

        # Intermediate saving if permits resume
        if args.resume:
            output_file = os.path.join(args.output_dir, "batch_results.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "num_samples": len(test_data),
                    "num_successful": sum(
                        1 for r in results
                        if (r.get("model_infer") or {}).get("answer", "").strip()
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "agents": {
                        "planner": args.planner_model,
                        "expert": args.expert_model,
                        "solver": args.solver_model,
                        "tool": args.tool_model,
                    },
                    "results": results,
                }, f, indent=4, default=str)
    
    # Save results
    output_file = os.path.join(args.output_dir, "batch_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "num_samples": len(test_data),
            "num_successful": sum(
                1 for r in results
                if (r.get("model_infer") or {}).get("answer", "").strip()
            ),
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "planner": args.planner_model,
                "expert": args.expert_model,
                "solver": args.solver_model,
                "tool": args.tool_model,
            },
            "results": results,
        }, f, indent=4, default=str)
    
    logger.info(f"{'='*70}")
    logger.info(f"Batch processing complete!")
    logger.info(f"Processed: {len(results)}/{len(test_data)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent System Execution")
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["single", "batch"], required=True,
                        help="Inference mode")
    parser.add_argument("--few-shot", type=int, choices=[0, 1, 2, 3], required=True, default=0,
                        help="Few-shot inference configuration. By default zero-shot inference.")
    
    # Model configuration
    parser.add_argument("--tool-model", type=str, default="openai/gpt-4o-mini",
                        help="Model for tools")
    parser.add_argument("--planner-model", type=str, default="openai/gpt-4o-mini",
                        help="Model for Planner agent")
    parser.add_argument("--expert-model", type=str, default="openai/gpt-4o-mini",
                        help="Model for Expert agent")
    parser.add_argument("--solver-model", type=str, default="openai/gpt-4o",
                        help="Model for Solver agent")
    parser.add_argument("--critic-model", type=str, default=None,
                        help="Model for Critic agent")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature for all agents")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Maximum output tokens for all agents")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Penalty for repeated tokens; 1.0 disables it")
    parser.add_argument("--output-limit", type=int, default=2048,
                        help="Custom output limit for Solver agent's solution")
    parser.add_argument("--planner-temperature", type=float, default=None,
                        help="Temperature for Planner (overrides --temperature)")
    parser.add_argument("--expert-temperature", type=float, default=None,
                        help="Temperature for Expert (overrides --temperature)")
    parser.add_argument("--solver-temperature", type=float, default=None,
                        help="Temperature for Solver (overrides --temperature)")
    parser.add_argument("--critic-temperature", type=float, default=None,
                        help="Temperature for Critic (overrides --temperature)")
    parser.add_argument("--tool-temperature", type=float, default=None,
                        help="Temperature for tools (overrides --temperature)")
    
    # API configuration
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (for Azure)")
    parser.add_argument("--api-version", type=str, default="2024-08-01-preview",
                        help="API version (for Azure)")
    
    # Google search API configuration
    parser.add_argument("--google-api-key", type=str, default=None,
                        help="Google Search API key")
    parser.add_argument("--google-cse-id", type=str, default=None,
                        help="Google Search Engine ID")
    
    # Single mode arguments
    parser.add_argument("--query", type=str, default=None,
                        help="Query for single mode")
    parser.add_argument("--context", type=str, default=None,
                        help="Context for single mode")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for single mode")
    
    # Batch mode arguments
    parser.add_argument("--data-source", type=str, choices=["local", "hf"], default="local",
                        help="Data source for batch mode")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test data JSON")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset name")
    parser.add_argument("--hf-split", type=str, default="test",
                        help="HuggingFace dataset split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max sample limit for this run")
    parser.add_argument("--give-contexts", action="store_true",
                        help="Whether give contexts to agent")
    
    # System configuration
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Maximum retry attempts")
    parser.add_argument("--expert-turns", type=int, default=1,
                        help="Number of expert retrieval turns")
    parser.add_argument("--critic-turns", type=int, default=0,
                        help="Iterative Solver<->Critic refinement rounds. 0 = off (default).")
    parser.add_argument("--enable-tools", type=str, default=None,
                        help="Comma-separated list of tools to enable (default: all)")
    parser.add_argument("--solver-awareness", action="store_true",
                        help="Whether to enable solver agent's solution generation self-awareness")
    parser.add_argument("--enable-reasoning", action="store_true",
                        help="Ask the Solver to produce its reasoning alongside the analysis")
    parser.add_argument("--output-dir", type=str, default="./anagent_output",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Whether permits resume from previous (Set to false to overwrite all and start from scratch)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run based on mode
    if args.mode == "single":
        if not args.query:
            raise ValueError("--query is required for single mode")
        run_single_inference(args)
    else:
        if not args.test_data and not args.hf_dataset:
            raise ValueError("--test-data or --hf-dataset is required for batch mode")
        run_batch_inference(args)


if __name__ == "__main__":
    main()