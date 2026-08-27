"""
SciBERT-based answer comparison.
"""

import torch
import multiprocessing as _mp
from transformers import AutoTokenizer, AutoModel
import numpy as np


class SciBERTScorer:
    """A scorer that uses SciBERT to compute BERTScore-style semantic similarity."""

    def __init__(self, model_name='allenai/scibert_scivocab_uncased', device=None):
        """Initialize the SciBERT model and tokenizer."""
        if device is not None:
            self.device = torch.device(device)
        elif _mp.current_process().name != "MainProcess":
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def get_token_embeddings(self, text, max_length=510):
        """Generate token-level embeddings using SciBERT, handling long texts with chunking."""
        # Tokenize to check length
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # If text fits in one chunk, process normally
        if len(tokens) <= max_length:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=False
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get token embeddings (exclude [CLS] and [SEP] tokens)
            token_embeddings = outputs.last_hidden_state[0][1:-1]
            return token_embeddings
        
        # Process long text in overlapping chunks
        chunk_size = max_length
        overlap = 50
        all_embeddings = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            inputs = self.tokenizer(
                chunk_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=False
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                chunk_embeddings = outputs.last_hidden_state[0][1:-1]
                all_embeddings.append(chunk_embeddings)
            
            if i + chunk_size >= len(tokens):
                break
        
        # Concatenate all chunk embeddings
        return torch.cat(all_embeddings, dim=0)
    
    def compute_similarity_score(self, gt_answer, model_answer, score_type='f1'):
        """Compare ground-truth answer with model-generated answer using BERTScore with SciBERT."""
        # Handle empty strings
        if not gt_answer.strip() or not model_answer.strip():
            return 0.0
        
        # Get token-level embeddings from SciBERT (now handles long texts)
        ref_embeddings = self.get_token_embeddings(gt_answer)
        cand_embeddings = self.get_token_embeddings(model_answer)
        
        # Normalize embeddings for cosine similarity
        ref_norm = ref_embeddings / ref_embeddings.norm(dim=1, keepdim=True)
        cand_norm = cand_embeddings / cand_embeddings.norm(dim=1, keepdim=True)
        
        # Compute pairwise cosine similarity matrix using SciBERT embeddings
        similarity_matrix = torch.mm(ref_norm, cand_norm.t())
        
        # Precision: For each candidate token, max similarity with reference tokens
        precision = similarity_matrix.max(dim=0)[0].mean().item()
        
        # Recall: For each reference token, max similarity with candidate tokens
        recall = similarity_matrix.max(dim=1)[0].mean().item()
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # Return requested score type
        if score_type == 'precision':
            return float(precision)
        elif score_type == 'recall':
            return float(recall)
        else:  # default to f1
            return float(f1)


def compare_answers(gt_answer, model_answer, scorer=None, score_type='f1'):
    """Compare ground-truth answer with model-generated answer using SciBERT."""
    if scorer is None:
        scorer = SciBERTScorer()
    
    return scorer.compute_similarity_score(gt_answer, model_answer, score_type)
