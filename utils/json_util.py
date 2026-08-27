"""
JSON utilities.
"""

import os
import json
import copy
import numpy as np
from typing import List, Dict, Union


def numpy_json_encoder(obj):
    """Custom JSON encoder function that handles NumPy arrays and other non-serializable types."""
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy scalar types
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                          np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    # Handle numpy scalars
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    # Handle other iterables that might contain numpy types
    elif hasattr(obj, 'items'):  # For dict-like objects
        return {k: numpy_json_encoder(v) for k, v in obj.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        return [numpy_json_encoder(i) for i in obj]
    else:
        # Let the default encoder handle it or throw a TypeError
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

def save_to_json(data: dict, file_path: str, if_sort: bool = False) -> None:
    """Save a dictionary or list of dictionaries to a JSON file at the specified file path."""
    if if_sort:
        data = copy.deepcopy(dict(sorted(data.items())))
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


def save_to_jsonl(data: Union[List[Dict], Dict], file_path: str) -> None:
    """Save a dictionary or list of dictionaries to a JSONL file at the specified file path."""
    # Ensure data is a list of dictionaries
    if isinstance(data, dict):
        sorted_dict = dict(sorted(data.items()))
        data = [sorted_dict]

    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            json_line = json.dumps(entry)
            jsonl_file.write(json_line + '\n')


def read_from_json(file_path: str) -> dict:
    """Reads JSON data from a file and returns it as a dictionary."""
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 0:  # check if the file is not empty
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            return data
        else:
            return {}  # return empty dict if the file is empty
    else:
        return {}  # return empty dict if the file does not exist
    

def read_from_jsonl(file_path: str) -> list:
    """Reads JSONL data from a file and returns it as a list of dictionaries."""
    data = []
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:  # check if the file exists and is not empty
        with open(file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                data.append(json.loads(line.strip()))
    
    return data
