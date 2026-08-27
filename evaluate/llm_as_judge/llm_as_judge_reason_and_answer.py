"""
LLM-as-judge evaluation for reasoning and answers.
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


REASON_START = "<reason>"
REASON_END = "</reason>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"


class LLMJudger:
    """A class to handle LLM-as-judge evaluation for table/figure analysis writing (reasoning + answer)."""
    
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

    def run_answer_llm_judger(self, data_type: str, gt_analysis: str, model_analysis: str):
        """LLM as a judge for analysis writing evaluation"""
        judge_prompt = """
You are an expert evaluator for scientific {data_type} analysis writing tasks. Given a {data_type} and its ground-truth analysis, evaluate whether a model-generated analysis is of high quality.

**Evaluation Criteria:**
1. **Content Accuracy**: Does the model analysis accurately capture the key information, findings, trends, and data points from the {data_type}?
2. **Completeness**: Does the model analysis cover the important aspects without missing critical information?
3. **Clarity and Coherence**: Is the model analysis clear, well-structured, and easy to understand?
4. **Conciseness**: Is the model analysis appropriately concise while maintaining completeness?

**Instructions:**
- Compare the Model Analysis against the Ground-truth Analysis
- Focus on the quality of model analysis writing, not exact word matching
- Consider semantic similarity and information coverage
- Rate the overall quality on a scale from 0-10 (where 10 is excellent)

**Ground-truth Analysis:**
{gt_analysis}

**Model Analysis:**
{model_analysis}

**Response Format:**
* Analysis: Provide a detailed analysis comparing the model analysis with the ground-truth analysis across the four criteria above. Please enclose your analysis in: <analysis>your analysis here</analysis>
* Extracted Model Analysis: Extract the core analysis content from the model response, removing any extraneous text or formatting. Please enclose the extracted analysis in: <model>extracted model analysis</model>
* Score: Rate the model analysis quality from 0-10. Please enclose your numerical score in: <score>numerical score</score>
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

    def run_reasoning_llm_judger(self, data_type: str, gt_analysis: str, model_reasoning: str):
        """LLM as a judge for reasoning evaluation"""
        judge_prompt = """
You are an expert evaluator for scientific {data_type} analysis writing reasoning evaluation. Given a {data_type} and its ground-truth analysis, evaluate the quality of the model's reasoning process that leads to its analysis writing output.

**Evaluation Criteria:**
1. **Understanding and Analysis (0-10)**: Does the reasoning demonstrate accurate understanding and analysis of the {data_type} content? Are key elements, patterns, and relationships correctly identified?
2. **Logical Structure and Flow (0-10)**: Is the reasoning logically structured with clear progression from observation to conclusion? Are the steps coherent and well-connected?
3. **Alignment with Ground-Truth (0-10)**: How well does the reasoning process align with the expected analysis approach and insights that should lead to the ground-truth analysis?

**Instructions:**
- Evaluate ONLY the reasoning process, NOT the final analysis
- Focus on the quality of analytical thinking and logical progression
- Consider how well the reasoning demonstrates understanding of the {data_type}
- Each score should be an integer from 0-10
- Provide three separate scores for the three criteria

**Ground-truth Analysis (for reference):**
{gt_analysis}

**Model Reasoning:**
{model_reasoning}

**Response Format:**
* Analysis: Provide a detailed analysis of the reasoning quality across all three criteria. Please enclose your analysis in: <analysis>your analysis here</analysis>
* Scores: Provide three scores (0-10) for: Understanding_and_Analysis, Logical_Structure_and_Flow, Alignment_with_Ground_Truth. Please enclose your scores in: <scores>score1, score2, score3</scores>
""".strip()

        judge_msg = [
            {
                "role": "user",
                "content": judge_prompt.format(
                    data_type=data_type, 
                    gt_analysis=gt_analysis, 
                    model_reasoning=model_reasoning
                )
            }
        ]

        response = self._run_vlm_interaction(messages=judge_msg)
        return response
    
    def _is_reason_score_valid(self, eval_score_list: List) -> bool:
        if isinstance(eval_score_list, List) and len(eval_score_list) == 3:
            for score in eval_score_list:
                if not isinstance(score, int):
                    return False
                if score < 0 or score > 10:
                    return False
            return True
        return False

    def run_judge_in_batch(self, json_path: str) -> Dict:
        raw_data = read_from_json(json_path)
        
        # Extract results list
        if "results" in raw_data:
            model_results = raw_data["results"]
        else:
            model_results = raw_data
            
        judge_results = []
        answer_total_score = 0.0
        reasoning_total_score = 0.0
        reasoning_score_1, reasoning_score_2, reasoning_score_3 = 0.0, 0.0, 0.0
        valid_answer_evaluations = 0
        valid_reasoning_evaluations = 0

        for result_item in tqdm(model_results, desc="Running LLM Judge", total=len(model_results)):
            # Extract data type
            data_type = result_item.get("type", "table/figure")
            
            # Extract ground truth analysis
            gt_analysis_list = result_item.get("ground_truth", [])
            if isinstance(gt_analysis_list, list) and len(gt_analysis_list) > 0:
                gt_analysis_texts = []
                for analysis_dict in gt_analysis_list:
                    if isinstance(analysis_dict, dict) and "info" in analysis_dict:
                        gt_analysis_texts.append(analysis_dict["info"].strip())
                gt_analysis = " ".join(gt_analysis_texts).strip()
            else:
                gt_analysis = ""
            
            # Extract model outputs
            model_infer = result_item.get("model_infer", {})
            model_reasoning = model_infer.get("reason", "")
            model_answer = model_infer.get("answer", "")
            
            if not gt_analysis:
                # Skip if no ground truth
                judge_results.append(result_item)
                continue

            result_copy = copy.deepcopy(result_item)

            ##########  1. Run answer judge  ##########
            if model_answer:
                answer_judge_result, answer_judge_analysis, extracted_model_answer, answer_judge_score = None, None, None, None
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    answer_judge_response = self.run_answer_llm_judger(data_type, gt_analysis, model_answer)
                    
                    if not answer_judge_response:
                        retry_count += 1
                        continue
                    
                    # Parse response
                    if "<analysis>" in answer_judge_response and "</analysis>" in answer_judge_response:
                        answer_judge_analysis = answer_judge_response.split("<analysis>")[1].split("</analysis>")[0].strip()
                    if "<model>" in answer_judge_response and "</model>" in answer_judge_response:
                        extracted_model_answer = answer_judge_response.split("<model>")[1].split("</model>")[0].strip()
                    if "<score>" in answer_judge_response and "</score>" in answer_judge_response:
                        score_str = answer_judge_response.split("<score>")[1].split("</score>")[0].strip()
                        try:
                            answer_judge_score = float(score_str)
                            if 0 <= answer_judge_score <= 10:
                                answer_judge_result = "success"
                            else:
                                answer_judge_score = None
                        except ValueError:
                            answer_judge_score = None

                    if answer_judge_result == "success" and answer_judge_analysis and extracted_model_answer and answer_judge_score is not None:
                        break
                    
                    retry_count += 1

                # Save answer judge results
                if answer_judge_result == "success":
                    result_copy["llm_as_judge"] = {
                        "extracted_model_answer": extracted_model_answer,
                        "judge_score": answer_judge_score,
                        "judge_analysis": answer_judge_analysis,
                        "raw_output": answer_judge_response
                    }
                    answer_total_score += answer_judge_score
                    valid_answer_evaluations += 1
                else:
                    result_copy["llm_as_judge"] = {
                        "extracted_model_answer": model_answer,
                        "judge_score": 0.0,
                        "judge_analysis": "Answer evaluation failed",
                        "raw_output": answer_judge_response or "No response"
                    }

            ##########  2. Run reasoning judge  ##########
            if model_reasoning:
                reasoning_judge_result, reasoning_judge_analysis, reasoning_judge_scores = None, None, None
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    reason_judge_response = self.run_reasoning_llm_judger(data_type, gt_analysis, model_reasoning)
                    
                    if not reason_judge_response:
                        retry_count += 1
                        continue
                    
                    # Parse response
                    if "<analysis>" in reason_judge_response and "</analysis>" in reason_judge_response:
                        reasoning_judge_analysis = reason_judge_response.split("<analysis>")[1].split("</analysis>")[0].strip()
                    if "<scores>" in reason_judge_response and "</scores>" in reason_judge_response:
                        scores_str = reason_judge_response.split("<scores>")[1].split("</scores>")[0].strip()
                        try:
                            reasoning_judge_scores = [int(s.strip()) for s in scores_str.split(",")]
                            if self._is_reason_score_valid(reasoning_judge_scores):
                                reasoning_judge_result = "success"
                            else:
                                reasoning_judge_scores = None
                        except ValueError:
                            reasoning_judge_scores = None

                    if reasoning_judge_result == "success" and reasoning_judge_analysis and reasoning_judge_scores:
                        break
                    
                    retry_count += 1
                # Save reasoning judge results
                if reasoning_judge_result == "success":
                    result_copy["llm_as_judge_reason"] = {
                        "judge_score": sum(reasoning_judge_scores) / len(reasoning_judge_scores),
                        "judge_scores": reasoning_judge_scores,
                        "judge_analysis": reasoning_judge_analysis,
                        "raw_output": reason_judge_response,
                    }
                    reasoning_total_score += sum(reasoning_judge_scores) / len(reasoning_judge_scores)
                    reasoning_score_1 += reasoning_judge_scores[0]
                    reasoning_score_2 += reasoning_judge_scores[1]
                    reasoning_score_3 += reasoning_judge_scores[2]
                    valid_reasoning_evaluations += 1
                else:
                    result_copy["llm_as_judge_reason"] = {
                        "judge_score": 0.0,
                        "judge_scores": [0, 0, 0],
                        "judge_analysis": "Reasoning evaluation failed",
                        "raw_output": reason_judge_response or "No response",
                    }

            judge_results.append(result_copy)

        # Create final result dictionary
        judge_dict = {}
        if "results" in raw_data:
            judge_dict = copy.deepcopy(raw_data)
            judge_dict["results"] = judge_results
        else:
            judge_dict["results"] = judge_results
            
        judge_dict["llm_as_judge_overview"] = {
            "total_data": len(judge_results),
            "valid_answer_evaluations": valid_answer_evaluations,
            "valid_reasoning_evaluations": valid_reasoning_evaluations,
            "answer_average_score": answer_total_score / valid_answer_evaluations if valid_answer_evaluations > 0 else 0.0,
            "reasoning_average_score": reasoning_total_score / valid_reasoning_evaluations if valid_reasoning_evaluations > 0 else 0.0,
            "reasoning_score_1_avg": reasoning_score_1 / valid_reasoning_evaluations if valid_reasoning_evaluations > 0 else 0.0,
            "reasoning_score_2_avg": reasoning_score_2 / valid_reasoning_evaluations if valid_reasoning_evaluations > 0 else 0.0,
            "reasoning_score_3_avg": reasoning_score_3 / valid_reasoning_evaluations if valid_reasoning_evaluations > 0 else 0.0,
            "answer_score_percentage": (answer_total_score / valid_answer_evaluations) * 10 if valid_answer_evaluations > 0 else 0.0,
            "reasoning_score_percentage": (reasoning_total_score / valid_reasoning_evaluations) * 10 if valid_reasoning_evaluations > 0 else 0.0,
        }
        
        # Save to json
        save_json_path = os.path.join(os.path.dirname(json_path), "llm_as_judge_result.json")
        save_to_json(judge_dict, save_json_path)

        return judge_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM as Judge for Table/Figure Analysis Evaluation (Reasoning + Answer)")
    
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