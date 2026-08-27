"""
LLM-as-judge evaluation for answers.
"""

import io
import re
import os
import ast
import sys
import copy
import json
import glob
import torch
import base64
import random
import requests
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import random
random.seed(42)

import litellm
from litellm import completion
from openai import AzureOpenAI

import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union, Any

_ANAGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ANAGENT_ROOT)
from utils.json_util import read_from_json, save_to_json
from evaluate.llm_as_judge.grader import parse_model_response


REASON_START = "<think>"
REASON_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"


def print_judge_metric_table(overview: dict):
    """Print the LLM-as-judge metric table."""
    avg = overview.get("average_scores", {}) or {}
    dims = ["accuracy", "completeness", "format", "writing", "faithfulness"]

    def pct(key):
        v = avg.get(key)
        return f"{v * 100:.2f}" if isinstance(v, (int, float)) else "n/a"

    line = "  " + "-" * 68
    print()
    print(f"  LLM-AS-JUDGE EVALUATION          n_scored = {overview.get('valid_evaluations', 0)}")
    print(line)
    print("  " + "".join(f"{d.upper()[:12]:>13}" for d in dims))
    print("  " + "".join(f"{pct(d):>13}" for d in dims))
    print(line)
    print(f"  {'AVERAGE (%)':<20}{pct('avg'):>10}")
    print(line)
    print(f"  excluded: n_empty_answer={overview.get('n_empty_answer', 0)}  "
          f"n_judge_parse_fail={overview.get('n_judge_parse_fail', 0)}  "
          f"n_missing_gt={overview.get('n_missing_gt', 0)}")
    print()


class LLMJudger:
    """A class to handle LLM-as-judge evaluation for table/figure analysis writing."""
    
    def __init__(self, args):
        """Initialize the LLM Judge."""
        self.args = args

        # GPT
        self.model_path = args.model_path
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.api_version = args.api_version
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_tokens
        self.timeout = args.timeout
        self.output_dir = args.output_dir
        self.cache_dir = args.cache_dir

        # Cost
        
        # LiteLLM
        self.litellm = litellm
        self.completion = completion
        
        # Set up LiteLLM environment
        self._setup_litellm()

    def _setup_litellm(self):
        """Setup LiteLLM configuration"""        
        # Set environment variables for Azure
        if self.api_key:
            os.environ["AZURE_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["AZURE_API_BASE"] = self.base_url
        if self.api_version:
            os.environ["AZURE_API_VERSION"] = self.api_version
            
        self.litellm.set_verbose = False
        
        print(f"LiteLLM configured for model: {self.model_path}")

    def _run_vlm_interaction(self, messages: List):
        """Run LLM interaction"""
        try:
            # Azure models, e.g., azure/gpt-4o
            if self.model_path.startswith("azure"):
                response = self.completion(
                    model=self.model_path,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    top_p=self.top_p,
                    api_key=self.api_key,
                    api_base=self.base_url,
                    api_version=self.api_version,
                    timeout=self.timeout,
                )
            # OpenAI models, e.g. openai/gpt-4.1-mini
            elif self.model_path.startswith("openai"):
                if "gpt-4.1-mini" in self.model_path or "gpt-4o-mini" in self.model_path:
                    max_completion_token = 32768
                else:
                    max_completion_token = 128000

                response = self.completion(
                    model=self.model_path,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=max_completion_token,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            # Gemini models, e.g. gemini/gemini-2.5-flash-preview-04-17
            elif self.model_path.startswith("gemini"):
                response = self.completion(
                    model=self.model_path,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=128000,
                    top_p=self.top_p,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )

            # Azure API
            elif self.model_path == "o3":
                response = self._azure_api_call(messages)

            # Extract response content
            model_response = response.choices[0].message.content.strip()


            if self.model_path.startswith("azure/"):
                # Reasoning is in a different place for azure GPT
                try:
                    model_reasoning = response.choices[0].message.reasoning_content.strip()
                    if REASON_START not in model_reasoning:
                        model_reasoning = f"{REASON_START}{model_reasoning}{REASON_END}"
                except Exception as e:
                    model_reasoning = ""

                model_response = model_reasoning + model_response
            
            return model_response
        except Exception as e:
            print(f"Error in LiteLLM completion call: {e}")
            return None

    def _azure_api_call(self, judge_messages: List):
        client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.base_url,
            api_key=self.api_key,
        )

        response = client.chat.completions.create(
            messages=judge_messages,
            max_completion_tokens=40000,
            model=self.model_path,
        )

        return response

    def run_llm_judger(self, data_type: str, gt_analysis: str, model_analysis: str):
        """LLM as a judge for table/figure analysis writing evaluation"""
        judge_prompt = """
You are an expert evaluator for scientific {data_type} analysis writing tasks. Given a {data_type} and its ground-truth analysis, evaluate whether a model-generated analysis is of high quality.

**Ground-truth Analysis**
{gt_analysis}

**Model Analysis**
{model_analysis}

**Evaluation Criteria**
1. **Content Accuracy**: Does the model analysis accurately and correctly reflect the explicitly presented information in the {data_type}, such as reported values, trends, stated findings, and observed patterns, without distortion or omission?
2. **Analytical Completeness**: Does the model analysis sufficiently develop the derived analyses, interpretations, and conclusions implied by the {data_type}, capturing key insights without missing important analytical findings?
3. **Format Correctness**: Are there any format errors in model analysis that violate the task requirements, such as incorrect structure, typographical errors, or improper formatting?
4. **Clarity & Coherence**: Is the model analysis clearly articulated, properly structured, and logically coherent, with sound scientific writing throughout?
5. **Reliability & Faithfulness**: Are there any hallucinated information in model analysis that is never covered, analyzed, and inferred in the ground-truth analysis? NOTE that both referred contents (e.g., referring to other tables/figures/sections/equations/etc., within the paper or referring to other papers) or reasonably inferred contents are considered as grounded (CAREFUL: analysis writing SHOULD have inferred contents, as long as they are reasonable and inferrable from the given information in such as way that are covered by ground-truth analysis)

**Instructions**
- Compare the **Model Analysis** against the **Ground-truth Analysis**
- Focus on the quality of model analysis writing in the five dimensions of **Evaluation Criteria:**
- Consider semantic similarity and information coverage, NOT exact word matching
- Rate the overall quality on a scale from 0-2 (where 2 is excellent)

**Response Format**
<think>your evaluation reasoning</think>
<accuracy>grade 0-2 for **Content Accuracy**: 0 = mostly inaccurate; 1 = partially accurate; 2 = fully accurate</accuracy>
<completeness>grade 0-2 for **Analytical Completeness**: 0 = most key analyses missing; 1 = some key analyses missing; 2 = all key analyses present</completeness>
<format>grade 0-2 for **Format Correctness**: 0 = many format errors; 1 = some format errors; 2 = no format error</format>
<writing>grade 0-2 for **Clarity & Coherence**: 0 = poorly written; 1 = moderately clear; 2 = clear and coherent</writing>
<faithfulness>grade 0-2 for **Reliability & Faithfulness**: 0 = mostly hallucinated; 1 = partially hallucinated; 2 = fully reasonable</faithfulness>
""".strip()

        judge_msg = [
            {
                "role": "user",
                "content": judge_prompt.format(
                    data_type=data_type, 
                    gt_analysis=gt_analysis, 
                    model_analysis=model_analysis
                )
            }
        ]

        response = self._run_vlm_interaction(messages=judge_msg)

        return response

    def run_judge_in_batch(self, json_path: str) -> Dict:
        raw_data = read_from_json(json_path)
        
        # Extract results list
        if "results" in raw_data:
            model_results = raw_data["results"]
        else:
            model_results = raw_data
            
        judge_results = []
        total_score = {'avg': 0.0, 'accuracy': 0.0, 'completeness': 0.0, 'format': 0.0, 'writing': 0.0, 'faithfulness': 0.0}
        max_retries = getattr(self.args, "max_retries", 3)
        valid_evaluations = 0
        n_empty_answer = 0
        n_judge_parse_fail = 0
        n_missing_gt = 0

        for result_item in tqdm(model_results, desc="Running LLM Judge", total=len(model_results)):
            # Extract data type
            data_type = result_item.get("type", "table/figure")
            
            # Extract ground truth analysis.
            gt_raw = result_item.get("ground_truth")
            if not gt_raw:
                gt_raw = result_item.get("gt_answer", "")
            if isinstance(gt_raw, list):
                gt_analysis = " ".join(
                    d["info"].strip() for d in gt_raw
                    if isinstance(d, dict) and "info" in d
                ).strip()
            elif isinstance(gt_raw, str):
                gt_analysis = gt_raw.strip()
            else:
                gt_analysis = ""
            
            # Extract model analysis (prefer answer over reasoning)
            model_infer = result_item.get("model_infer", {})
            model_analysis = model_infer.get("answer", "")
            if not model_analysis:
                model_analysis = model_infer.get("reason", "")
            
            if not gt_analysis:
                # No reference to judge against: not the model's failure, not scored.
                n_missing_gt += 1
                judge_results.append(result_item)
                continue

            if not model_analysis:
                # The model produced nothing. That is a zero, not an exclusion.
                n_empty_answer += 1
                valid_evaluations += 1
                judge_results.append(result_item)
                continue

            # Run judge with retries
            judge_raw_response = None
            judge_grades = None
            
            for retry_idx in range(max_retries):
                judge_raw_response = self.run_llm_judger(data_type, gt_analysis, model_analysis)
                
                if judge_raw_response is None:
                    continue
                    
                judge_grades = parse_model_response(judge_raw_response)
                
                if judge_grades is not None:
                    break

            if judge_grades is None:
                # The judge failed to return a parseable grade after all retries
                n_judge_parse_fail += 1
                judge_results.append(result_item)
                continue

            # Save judge results
            result_copy = copy.deepcopy(result_item)
            result_copy["llm_as_judge"] = {
                "judge_raw_response": judge_raw_response,
                "judge_grades": judge_grades,
            }
            score_list = []
            for metric in judge_grades:
                if metric == "think":
                    continue
                total_score[metric] += judge_grades[metric]
                score_list.append(judge_grades[metric])
            total_score["avg"] += sum(score_list) / len(score_list)
            valid_evaluations += 1

            judge_results.append(result_copy)

        # Create final result dictionary
        judge_dict = {"llm_as_judge_overview": None}
        if "results" in raw_data:
            judge_dict = copy.deepcopy(raw_data)
            judge_dict["results"] = judge_results
        else:
            judge_dict["results"] = judge_results

        # Compute final average scores (guard against zero valid evaluations)
        overall_avg = {}
        denom = valid_evaluations if valid_evaluations else 1
        for metric in total_score:
            overall_avg[metric] = total_score[metric] / denom
            
        judge_dict["llm_as_judge_overview"] = {
            "total_data": len(judge_results),
            "valid_evaluations": valid_evaluations,
            "n_empty_answer": n_empty_answer,
            "n_judge_parse_fail": n_judge_parse_fail,
            "n_missing_gt": n_missing_gt,
            "average_scores": overall_avg,
        }

        print_judge_metric_table(judge_dict["llm_as_judge_overview"])
        
        # Save to json
        save_json_path = os.path.join(os.path.dirname(json_path), "llm_as_judge_result.json")
        save_to_json(judge_dict, save_json_path)

        return judge_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM as Judge for Table/Figure Analysis Evaluation")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                       help="Model path (e.g., 'azure/gpt-4o' or 'openai/gpt-4o')")
    
    # API arguments
    parser.add_argument("--api_key", type=str, required=True,
                       help="API key (Azure or OpenAI)")
    parser.add_argument("--base_url", type=str, default=None,
                       help="Azure OpenAI base URL (required for Azure models)")
    parser.add_argument("--api_version", type=str, default="2024-02-01",
                       help="Azure API version (only for Azure models)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p for generation")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Maximum tokens to generate")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Judge retries per instance before recording a parse failure")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Request timeout in seconds")
    
    # Data and output
    parser.add_argument("--json_path", type=str, required=True,
                       help="Path to the input JSON file with model results")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Cache directory for temporary files")
    
    # Other options
    
    args = parser.parse_args()
    
    # Validation for Azure models
    if args.model_path.startswith("azure/") and not args.base_url:
        raise ValueError("Azure models require --base_url parameter")
    
    # Initialize and run judge
    judge = LLMJudger(args)
    result_dict = judge.run_judge_in_batch(args.json_path)
    
    print(f"\nLLM Judge evaluation completed!")
    print(f"\n\nEvaluation Results:\n{result_dict['llm_as_judge_overview']}\n\n")