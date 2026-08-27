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
import string
import requests
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict, List, Any

import bert_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer

_ANAGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ANAGENT_ROOT)
from evaluate.scibert import SciBERTScorer

import nltk
_NLTK_DIR = os.environ.get("NLTK_DATA", os.path.join(os.path.expanduser("~"), "nltk_data"))
nltk.download('wordnet', download_dir=_NLTK_DIR)
nltk.download('omw-1.4', download_dir=_NLTK_DIR)

# Tell NLTK where to find the data
nltk.data.path.append(_NLTK_DIR)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from utils.json_util import read_from_json, save_to_json


sys.set_int_max_str_digits(640)  # plenty for your 8668-digit case

REASON_START = "<reason>"
REASON_END = "</reason>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"



def _to_unit_interval(score: float) -> float:
    """Map a cosine-based score from [-1, 1] onto [0, 1]."""
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def print_rule_metric_table(metrics: Dict, n: int, encoder: str = "scibert", prefix: str = "answer_"):
    """Print the rule-based metric table."""
    def pct(key):
        # Values arrive already scaled to percent by the aggregation step.
        v = metrics.get(f"{prefix}{key}")
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    bert_label = "SCIBERT" if encoder == "scibert" else "BERTSCORE"
    line = "  " + "-" * 68
    print()
    print(f"  RULE-BASED EVALUATION            n = {n}    encoder = {encoder}")
    print(line)
    print(f"  {'Semantic Accuracy (%)':<34}{'Lexical Accuracy (%)':<34}")
    print(f"  {'COSINE':>10}{bert_label:>12}{'METEOR':>10}"
          f"{'ROUGE-L':>12}{'BLEU':>10}{'WORD':>10}")
    print(f"  {pct('cosine_similarity'):>10}{pct('bert'):>12}{pct('meteor'):>10}"
          f"{pct('rougeL'):>12}{pct('bleu'):>10}{pct('word_overlap'):>10}")
    print(line)
    print(f"  {'Overall Accuracy (%)':<34}")
    print(f"  {'S_Sem':>10}{'S_Lex':>12}{'S_Avg':>10}")
    print(f"  {pct('semantic_score'):>10}{pct('lexical_score'):>12}{pct('quality'):>10}")
    print(line)
    print()


def numpy_json_encoder(obj):
    """Custom JSON encoder for numpy types and PIL Images"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    # Handle PIL Images
    elif hasattr(obj, 'mode') and hasattr(obj, 'size'):  # PIL Image detection
        return f"<PIL_Image_{obj.mode}_{obj.size[0]}x{obj.size[1]}>"
    elif hasattr(obj, 'items'):
        return {k: numpy_json_encoder(v) for k, v in obj.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        return [numpy_json_encoder(i) for i in obj]
    else:
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def setup_nltk(cache_dir):
    # Set NLTK data directory
    os.environ['NLTK_DATA'] = cache_dir
    nltk.data.path.append(cache_dir)

    # Download required NLTK resources with explicit download_dir
    try:
        # Try to load the punkt tokenizer
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print(f"Downloading NLTK punkt tokenizer to {cache_dir}")
        nltk.download('punkt', download_dir=cache_dir)

    # Sometimes we need to download punkt_tab separately
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print(f"Downloading NLTK punkt_tab to {cache_dir}")
        nltk.download('punkt_tab', download_dir=cache_dir)


class AnalysisEvaluator:
    """Class for evaluating table/figure analysis quality"""

    def __init__(self, args):
        """Initialize evaluation metrics"""
        self.args = args
        self.cache_dir = args.cache_dir

        # Initialize models lazily to avoid loading them if not needed
        self._sentence_transformer = None
        self._rouge_scorer = None
        self._scibert_scorer = None

    @property
    def sentence_transformer(self):
        """Lazy loading of sentence transformer"""
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer(
                'sentence-transformers/all-mpnet-base-v2',
                device='cpu'
                # No need to specify cache_folder - it will use HF_HOME automatically
            )
        return self._sentence_transformer

    @property
    def rouge_scorer(self):
        """Lazy loading of ROUGE scorer"""
        if self._rouge_scorer is None:
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        return self._rouge_scorer

    @property
    def scibert_scorer(self):
        """Lazy loading of SciBERT scorer"""
        if self._scibert_scorer is None:
            import sys
            sys.stdout.flush()

            try:
                self._scibert_scorer = SciBERTScorer()
            except Exception as e:
                print(f"Error loading SciBERT: {e}")
                traceback.print_exc()
                raise
        return self._scibert_scorer

    def _calculate_bleu_score(self, gt_text, pred_text) -> float:
        """Calculate BLEU score between ground truth and prediction"""
        if not pred_text or not gt_text:
            return 0.0

        smoother = SmoothingFunction().method1

        try:
            gt_tokens = nltk.word_tokenize(gt_text.lower())
            pred_tokens = nltk.word_tokenize(pred_text.lower())
        except LookupError:
            gt_tokens = gt_text.lower().split()
            pred_tokens = pred_text.lower().split()

        if len(gt_tokens) > 0 and len(pred_tokens) > 0:
            try:
                bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)
                return bleu
            except Exception:
                # Fallback to word overlap
                common_words = set(gt_tokens).intersection(set(pred_tokens))
                total_words = set(gt_tokens).union(set(pred_tokens))
                return len(common_words) / max(1, len(total_words))

        return 0.0

    def _calculate_rouge_scores(self, gt_text, pred_text) -> Dict:
        """Calculate ROUGE scores"""
        if not pred_text or not gt_text:
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
            }

        try:
            scores = self.rouge_scorer.score(gt_text, pred_text)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }

    def _calculate_meteor_score(self, gt_text, pred_text) -> float:
        """Calculate METEOR score"""
        if not pred_text or not gt_text:
            return 0.0

        try:
            # METEOR requires proper tokenization
            gt_tokens = nltk.word_tokenize(gt_text.lower())
            pred_tokens = nltk.word_tokenize(pred_text.lower())

            if gt_tokens and pred_tokens:
                score = meteor_score([gt_tokens], pred_tokens)
                return score
        except Exception as e:
            print(f"Error calculating METEOR: {e}")
            return 0.0

        return 0.0

    def _calculate_bert_score(self, gt_text, pred_text) -> float:
        """Calculate BERTScore"""
        if not pred_text or not gt_text:
            return 0.0

        try:
            # BERTScore will automatically use TRANSFORMERS_CACHE
            P, R, F1 = bert_score.score(
                [pred_text], 
                [gt_text], 
                lang="en", 
                model_type='bert-base-uncased'
            )
            return F1.mean().item()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return 0.0

    def _calculate_scibert_score(self, gt_text, pred_text) -> float:
        """Calculate BERTScore using SciBERT"""
        if not pred_text or not gt_text:
            return 0.0

        try:
            # Use SciBERT for token-level BERTScore
            score = self.scibert_scorer.compute_similarity_score(
                gt_text, 
                pred_text, 
                score_type='f1'
            )
            return score
        except Exception as e:
            print(f"Error calculating BERTScore with SciBERT: {e}")
            return 0.0

    def _calculate_cosine_similarity(self, gt_text, pred_text):
        """Calculate cosine similarity using sentence transformers"""
        if not pred_text or not gt_text:
            return 0.0

        try:
            gt_embedding = self.sentence_transformer.encode([gt_text])
            pred_embedding = self.sentence_transformer.encode([pred_text])

            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(gt_embedding), 
                torch.tensor(pred_embedding)
            ).item()

            return similarity
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _calculate_word_overlap(self, gt_text, pred_text):
        """Calculate simple word overlap similarity"""
        if not pred_text or not gt_text:
            return 0.0

        gt_words = set(gt_text.lower().split())
        pred_words = set(pred_text.lower().split())

        if gt_words and pred_words:
            overlap = len(gt_words.intersection(pred_words)) / len(gt_words.union(pred_words))
            return overlap
        elif not gt_words and not pred_words:
            return 1.0
        else:
            return 0.0

    def evaluate_text(self, gt_text: str, pred_text: str, metric_prefix: str = "") -> Dict:
        """Evaluate text quality with comprehensive metrics"""
        # Initialize metrics
        metrics = {}

        # Length instrumentation
        metrics[f"{metric_prefix}gt_chars"] = len(gt_text or "")
        metrics[f"{metric_prefix}answer_chars"] = len(pred_text or "")
        metrics[f"{metric_prefix}length_ratio"] = (
            len(pred_text or "") / len(gt_text) if gt_text else 0.0
        )

        # 1. BLEU Score
        metrics[f"{metric_prefix}bleu"] = self._calculate_bleu_score(gt_text, pred_text)

        # 2. ROUGE Scores
        rouge_scores = self._calculate_rouge_scores(gt_text, pred_text)
        for rouge_type, score in rouge_scores.items():
            metrics[f"{metric_prefix}{rouge_type}"] = score

        # 3. METEOR Score
        metrics[f"{metric_prefix}meteor"] = self._calculate_meteor_score(gt_text, pred_text)

        # 4. BERTScore
        if getattr(self.args, "semantic_encoder", "scibert") == "bertscore":
            metrics[f"{metric_prefix}bert"] = _to_unit_interval(
                self._calculate_bert_score(gt_text, pred_text))
        else:
            metrics[f"{metric_prefix}bert"] = _to_unit_interval(
                self._calculate_scibert_score(gt_text, pred_text))

        # 5. Cosine Similarity with Sentence Transformers
        metrics[f"{metric_prefix}cosine_similarity"] = _to_unit_interval(
            self._calculate_cosine_similarity(gt_text, pred_text))

        # 6. Simple word overlap
        metrics[f"{metric_prefix}word_overlap"] = self._calculate_word_overlap(gt_text, pred_text)

        # Calculate overall quality as weighted average of key metrics
        semantic = [
            metrics[f"{metric_prefix}cosine_similarity"],
            metrics[f"{metric_prefix}bert"],
            metrics[f"{metric_prefix}meteor"],
        ]
        lexical = [
            metrics[f"{metric_prefix}rougeL"],
            metrics[f"{metric_prefix}bleu"],
            metrics[f"{metric_prefix}word_overlap"],
        ]
        metrics[f"{metric_prefix}semantic_score"] = sum(semantic) / len(semantic)
        metrics[f"{metric_prefix}lexical_score"] = sum(lexical) / len(lexical)
        metrics[f"{metric_prefix}quality"] = (
            metrics[f"{metric_prefix}semantic_score"]
            + metrics[f"{metric_prefix}lexical_score"]
        ) / 2.0

        return metrics

    def evaluate(self, ground_truth, prediction):
        """Evaluate table/figure analysis quality"""

        # Extract ground truth data
        gt_summary_list = ground_truth.get("ground_truth", ground_truth)

        # Extract prediction data
        pred_reasoning = prediction.get("reason", "")
        pred_answer = prediction.get("answer", "")
        pred_refined_answer = prediction.get("refined_answer", "")

        # Combine ground truth analyses into a single string
        if isinstance(gt_summary_list, list) and len(gt_summary_list) > 0:
            gt_summary_texts = []
            for summary_dict in gt_summary_list:
                if isinstance(summary_dict, dict) and "info" in summary_dict:
                    gt_summary_texts.append(summary_dict["info"].strip())
            gt_summary = " ".join(gt_summary_texts).strip()
        elif isinstance(gt_summary_list, dict):
            gt_summary = gt_summary_list.get("ground_truth")
        elif isinstance(gt_summary_list, str):
            gt_summary = gt_summary_list
        else:
            gt_summary = ""

        all_metrics = {}

        # Evaluate reasoning if present
        if pred_reasoning:
            reasoning_metrics = self.evaluate_text(gt_summary, pred_reasoning, "reasoning_")
            all_metrics.update(reasoning_metrics)
        else:
            # Default reasoning metrics when no reasoning available
            reasoning_metrics = {
                "reasoning_quality": 0.0,
                "reasoning_bleu": 0.0,
                "reasoning_rouge1": 0.0,
                "reasoning_rouge2": 0.0,
                "reasoning_rougeL": 0.0,
                "reasoning_meteor": 0.0,
                "reasoning_bert": 0.0,
                "reasoning_semantic_score": 0.0,
                "reasoning_lexical_score": 0.0,
                "reasoning_cosine_similarity": 0.0,
                "reasoning_word_overlap": 0.0,
                "reasoning_gt_chars": len(gt_summary or ""),
                "reasoning_answer_chars": 0,
                "reasoning_length_ratio": 0.0,
            }
            all_metrics.update(reasoning_metrics)

        # Evaluate answer/summary
        if pred_answer:
            answer_metrics = self.evaluate_text(gt_summary, pred_answer, "answer_")
            all_metrics.update(answer_metrics)
        else:
            # Default answer metrics when no answer available
            answer_metrics = {
                "answer_quality": 0.0,
                "answer_bleu": 0.0,
                "answer_rouge1": 0.0,
                "answer_rouge2": 0.0,
                "answer_rougeL": 0.0,
                "answer_meteor": 0.0,
                "answer_bert": 0.0,
                "answer_semantic_score": 0.0,
                "answer_lexical_score": 0.0,
                "answer_cosine_similarity": 0.0,
                "answer_word_overlap": 0.0,
                "answer_gt_chars": len(gt_summary or ""),
                "answer_answer_chars": 0,
                "answer_length_ratio": 0.0,
            }
            all_metrics.update(answer_metrics)

        # Evaluate refined_answer/summary
        if pred_refined_answer:
            refined_answer_metrics = self.evaluate_text(gt_summary, pred_refined_answer, "refined_answer_")
            all_metrics.update(refined_answer_metrics)
        else:
            # Default refined_answer metrics when no refined_answer available
            refined_answer_metrics = {
                "refined_answer_quality": 0.0,
                "refined_answer_bleu": 0.0,
                "refined_answer_rouge1": 0.0,
                "refined_answer_rouge2": 0.0,
                "refined_answer_rougeL": 0.0,
                "refined_answer_meteor": 0.0,
                "refined_answer_bert": 0.0,
                "refined_answer_semantic_score": 0.0,
                "refined_answer_lexical_score": 0.0,
                "refined_answer_cosine_similarity": 0.0,
                "refined_answer_word_overlap": 0.0,
                "refined_answer_gt_chars": len(gt_summary or ""),
                "refined_answer_answer_chars": 0,
                "refined_answer_length_ratio": 0.0,
            }
            all_metrics.update(refined_answer_metrics)

        # Overall score is always the answer score
        all_metrics["overall_score"] = all_metrics.get("answer_quality", 0.0)

        return all_metrics



##################################################
#                 Main Functions
##################################################
def evaluate_single_item(args_tuple):
    """Worker function to evaluate a single item"""
    idx, result_item, args_dict = args_tuple

    # Recreate args from dict
    class Args:
        pass
    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)

    # Create evaluator for this worker
    evaluator = AnalysisEvaluator(args)

    # Extract ground truth and prediction data
    gt_data = {
        "ground_truth": result_item.get("ground_truth", result_item.get("gt_answer"))
    }

    pred_data = result_item.get("model_infer", {})

    # Compute evaluation metrics
    try:
        eval_metrics = evaluator.evaluate(gt_data, pred_data)
    except Exception as e:
        print(f"\nError evaluating sample {idx}: {e}")
        traceback.print_exc()
        # Set default metrics on error
        eval_metrics = {
            "overall_score": 0.0,
            "reasoning_quality": 0.0,
            "answer_quality": 0.0,
        }

    # Add eval to result item
    result_item["eval"] = eval_metrics

    return idx, result_item, eval_metrics

def run_inference_evaluation(args, infer_results: Dict, eval_summary_save_name: str) -> Dict:
    """Run evaluation on inference results"""
    # Init evaluator
    evaluator = AnalysisEvaluator(args)
    total_tests = len(infer_results)
    all_metrics = {}
    table_metrics = {}
    figure_metrics = {}

    print(f"\nStarting evaluation of {total_tests} samples...")
    print(f"This may take a while depending on the dataset size.\n")

    # (1) Evaluate each inference output with multiprocessing
    num_workers = getattr(args, 'num_workers', 1)

    if num_workers > 1:
        print(f"Using {num_workers} workers for parallel evaluation...")

        # Convert args to dict for pickling
        args_dict = vars(args)

        # Prepare arguments for parallel processing
        eval_args = [(idx, result_item, args_dict) for idx, result_item in enumerate(infer_results)]

        # Use multiprocessing Pool
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(evaluate_single_item, eval_args),
                total=total_tests,
                desc="Evaluating samples"
            ))

        # Sort results by index and update infer_results
        results.sort(key=lambda x: x[0])
        for idx, (_, result_item, eval_metrics) in enumerate(results):
            infer_results[idx] = result_item

            # Accumulate metrics for overall evaluation
            for metric, value in eval_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
    else:
        # Single-process evaluation (original code)
        print("Using single-process evaluation...")
        for idx, result_item in enumerate(tqdm(infer_results, desc="Evaluating samples")):
            # Extract ground truth and prediction data
            gt_data = {
                "ground_truth": result_item.get("ground_truth", result_item.get("gt_answer"))
            }

            pred_data = result_item.get("model_infer", {})

            # Compute evaluation metrics
            try:
                eval_metrics = evaluator.evaluate(gt_data, pred_data)
            except Exception as e:
                print(f"\nError evaluating sample {idx}: {e}")
                traceback.print_exc()
                # Set default metrics on error
                eval_metrics = {
                    "overall_score": 0.0,
                    "reasoning_quality": 0.0,
                    "answer_quality": 0.0,
                }

            # Accumulate metrics for overall evaluation
            for metric, value in eval_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

            # Save to dict
            result_item["eval"] = eval_metrics

    print(f"\n✓ Completed evaluation of {total_tests} samples\n")

    # (2) Separate metrics by data type
    print("Separating metrics by data type...")
    for result_item in tqdm(infer_results, desc="Categorizing metrics"):
        if "eval" not in result_item:
            continue

        data_type = result_item.get("type", "")
        eval_metrics = result_item["eval"]

        # Add to table metrics
        if data_type == "table":
            for metric, value in eval_metrics.items():
                if metric not in table_metrics:
                    table_metrics[metric] = []
                table_metrics[metric].append(value)

        # Add to figure metrics  
        elif data_type == "figure":
            for metric, value in eval_metrics.items():
                if metric not in figure_metrics:
                    figure_metrics[metric] = []
                figure_metrics[metric].append(value)

    # Rest of your code remains the same...
    print("Computing average metrics...")

    # (3) Compute overall metrics (average across all tests)
    def compute_avg_metrics(metrics_dict, name_prefix=""):
        computed_metrics = {}
        for metric, values in metrics_dict.items():
            if not values:  # Skip empty lists
                continue

            if isinstance(values[0], Dict):
                overall_dict = {}
                for scores in values:
                    for kk in scores:
                        if kk not in overall_dict:
                            overall_dict[kk] = scores[kk]
                        else:
                            overall_dict[kk] += scores[kk]
                for kk in overall_dict:
                    overall_dict[kk] = overall_dict[kk] / len(values) * 100
                computed_metrics[metric] = overall_dict
            else:
                computed_metrics[metric] = sum(values) / len(values) * 100 if values else 0.0
        return computed_metrics

    overall_metrics = compute_avg_metrics(all_metrics)
    table_only_metrics = compute_avg_metrics(table_metrics)
    figure_only_metrics = compute_avg_metrics(figure_metrics)

    # Save overall evaluation summary
    print(f"Saving evaluation summary to {eval_summary_save_name}...")
    summary = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "table_tests": len([r for r in infer_results if r.get("type") == "table"]),
        "figure_tests": len([r for r in infer_results if r.get("type") == "figure"]),
        "overall_metrics": overall_metrics,
        "table_only_metrics": table_only_metrics,
        "figure_only_metrics": figure_only_metrics,
        "all_metrics": all_metrics,
    }

    if len(eval_summary_save_name) > 100:
        eval_summary_save_name = eval_summary_save_name[-20:]  # if file name too long

    summary_json = os.path.join(args.output_dir, eval_summary_save_name)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=4, default=numpy_json_encoder)

    print(f"✓ Summary saved successfully\n")

    print_rule_metric_table(
        overall_metrics, total_tests,
        encoder=getattr(args, "semantic_encoder", "scibert"),
    )

    return infer_results, {"overall": overall_metrics, "table_only": table_only_metrics, "figure_only": figure_only_metrics}

def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description="Table/Figure Summarization Evaluation")
    parser.add_argument("--infer-output", type=str, help="Path to inference output results")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers for evaluation (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (kept for compatibility)")

    parser.add_argument("--output-dir", type=str, default="./eval_outputs", help="Directory path for output")
    parser.add_argument("--cache-dir", type=str, default="/tmp", help="Cache directory")
    parser.add_argument("--semantic-encoder", type=str, default="scibert",
                        choices=["scibert", "bertscore"],
                        help="Encoder behind the BERT column of the semantic group")

    args = parser.parse_args()

    # Setup NLTK
    setup_nltk(args.cache_dir)

    # Output file
    base_name = os.path.basename(args.infer_output)    # 'data.json'
    name_without_ext = os.path.splitext(base_name)[0]  # 'data'
    eval_save_name = f"{name_without_ext}_eval.json"
    eval_summary_save_name = f"{name_without_ext}_eval_summary.json"

    # Evaluate inference results
    infer_results = read_from_json(args.infer_output)

    # Extract results list from the loaded data
    if "results" in infer_results:
        results_list = infer_results["results"]
    else:
        # Assume the entire loaded data is the results list
        results_list = infer_results

    # A run that died mid-batch still produces a well-formed file
    declared = infer_results.get("num_samples") if isinstance(infer_results, dict) else None
    if declared and len(results_list) != declared:
        print()
        print("  " + "!" * 68)
        print(f"  WARNING: incomplete run - {len(results_list)} of {declared} instances present.")
        print(f"  Scores below cover {100.0 * len(results_list) / declared:.1f}% of the intended set.")
        print("  " + "!" * 68)
        print()

    eval_results, overall_eval = run_inference_evaluation(
        args,
        results_list,
        eval_summary_save_name,
    )

    # Update the original structure
    if "results" in infer_results:
        infer_results["results"] = eval_results
        infer_results["overall_eval"] = overall_eval["overall"]
        infer_results["table_only_eval"] = overall_eval["table_only"] 
        infer_results["figure_only_eval"] = overall_eval["figure_only"]
        final_results = infer_results
    else:
        final_results = {
            "results": eval_results,
            "overall_eval": overall_eval["overall"],
            "table_only_eval": overall_eval["table_only"],
            "figure_only_eval": overall_eval["figure_only"]
        }

    # Save results to JSON
    if len(eval_save_name) > 120:
        eval_save_name = eval_save_name[-120:]
    output_json = os.path.join(args.output_dir, eval_save_name)
    with open(output_json, "w") as f:
        json.dump(final_results, f, indent=4, default=numpy_json_encoder)

    # Print summary results
    print(
        f"\n\n{'=' * 60} EVALUATION SUMMARY {'=' * 60}"
        f"\n    - Evaluation details saved to: {eval_save_name}"
        f"\n    - Evaluation summary saved to: {eval_summary_save_name}"
        f"\n    - Length of Data: {len(eval_results)}"
        f"\n    - Overall evaluation metrics:"
    )

    print(f"\n    - Overall Metrics (All Data):")
    for metric, value in overall_eval["overall"].items():
        if isinstance(value, (int, float)):
            print(f"        - {metric}: {value:.8f}")
        else:
            print(f"        - {metric}: {value}")

    print(f"\n    - Table-Only Metrics:")
    for metric, value in overall_eval["table_only"].items():
        if isinstance(value, (int, float)):
            print(f"        - {metric}: {value:.8f}")
        else:
            print(f"        - {metric}: {value}")

    print(f"\n    - Figure-Only Metrics:")
    for metric, value in overall_eval["figure_only"].items():
        if isinstance(value, (int, float)):
            print(f"        - {metric}: {value:.8f}")
        else:
            print(f"        - {metric}: {value}")
    print("\n\n")

if __name__ == "__main__":
    main()