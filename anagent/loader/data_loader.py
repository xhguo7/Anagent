"""
Data loader for task running.
"""

import os
import sys
import json
import copy
import tqdm
from tqdm import tqdm
from typing import Dict, Any, List
from datasets import load_dataset

import random

from utils.logger import setup_logger
logger = setup_logger(__name__)

from anagent.tools.rag import ArXivRAG, EFetchRAG


IMAGE_TOKEN = "<image>"

OURS_SOURCES = """
**Paper Source**
* Title: {title}
* URL: {url}
""".strip()



OURS_SOURCES_WITH_REFERENCES = """
**Paper Source**
* Title: {title}
* URL: {url}
* **References:**
{references}
"""



OURS_QUERY = """
Help write a scientific analysis paper section for the **Given {data_type}** below.
* This scientific analysis paper section should be in ONE paragraph, consisting of around {len_limit} tokens, in Latex format.
* This section should be in plain text, NO MARKDOWN, ONE paragraph, focusing on in-depth analysis, NOT repeating your reading of the given {data_type}.
* Please ensure the accuracy, precision, profession, and quality of your academic paper writing.
**Given {data_type}:**
{task_data}
"""



def is_valid_image_path(image_path):
    """Check if an image path is valid and points to a valid image file."""
    if not image_path:
        return True  # No image is valid
    
    if not isinstance(image_path, str):
        return False
    
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        return False
    
    # Try to actually open the image to verify it's valid
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False


def load_test_data(args):
    """Load test data based on source."""
    if args.data_source == "local":
        # Load from local JSON file
        if not os.path.exists(args.test_data):
            raise ValueError(f"Test data file not found: {args.test_data}")
        
        with open(args.test_data, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, List):
            test_data = copy.deepcopy(data)
        else:
            test_data = [data]

        # Filter out samples with invalid image paths
        original_count = len(test_data)
        test_data = [item for item in test_data if is_valid_image_path(item.get("image"))]
        filtered_count = len(test_data)
        if filtered_count < original_count:
            logger.warning(f"Skipped {original_count - filtered_count} samples with invalid images ({original_count} -> {filtered_count})")

        # Apply the sample limit.
        max_samples = getattr(args, "max_samples", None)
        if max_samples and len(test_data) > max_samples:
            rng = random.Random(getattr(args, "seed", 42))
            rng.shuffle(test_data)
            test_data = test_data[:max_samples]
            logger.info(f"--max-samples={max_samples}: using {len(test_data)} instances")

    elif args.data_source == "hf":
        # Load from HuggingFace datasets
        dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        test_data = list(dataset)
    
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    # Preprocess dataset
    logger.info(f"Processing samples...\n")
    processed_test_data = []
    for i, item in enumerate(tqdm(test_data, desc="Preprocessing")):
        # Only use data with ground truth
        if len(item["gt_answer"]) == 0:
            continue
        

        # Task data
        task_data = item["data"]

        # Task sources
        item["task_sources"] = OURS_SOURCES.format(
            title=item["source"]["title"],
            url=item["source"]["url"],
            xml_url=item["source"].get("xml_url", "Unavailable"),
        ).strip()
        
        # Task image
        task_image = item["image"]
        if task_image:
            item["task_sources"] = item["task_sources"] + f"\n* Image Path: {task_image}"
            if IMAGE_TOKEN not in task_data:
                task_data = f"{IMAGE_TOKEN}\n{task_data}"

        # Query
        if "query" not in item:
            item["query"] = OURS_QUERY.format(
                task_data=task_data,
                data_type=item["type"],
                len_limit=int(len(item["gt_answer"])/4),
                # len_limit=args.output_limit
            ).strip()

        # Contexts
        if "contexts" not in item or args.give_contexts == False:
            xml_url = item["source"].get("xml_url", None)
            url = item["source"]["url"]
            if xml_url or "pubmed" in url:
                rag_tool = EFetchRAG()
            else:
                rag_tool = ArXivRAG()
            
            # Collect contexts
            item["contexts"] = rag_tool.format_contexts_for_agent(rag_tool.extract_context(task_data, url))

        processed_test_data.append(item)

    return processed_test_data