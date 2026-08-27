import re
from typing import Dict, Tuple, Optional


def response_parser(
    response: str, 
    tag_dict: Dict,
    parse_mode: str = "strict",
) -> Tuple[bool, str]:
    """Parse model output to extract content between tags."""
    assert parse_mode in ["strict", "relaxed"]

    # Get tags
    tag_start, tag_end = None, None
    if "start" in tag_dict:
        tag_start = tag_dict["start"]
    if "end" in tag_dict:
        tag_end = tag_dict["end"]

    # If no tags specified, return False
    if not tag_start and not tag_end:
        return False, response.strip()
    
    # If only one tag is specified, treat the other as empty
    if not tag_start:
        tag_start = ""
    if not tag_end:
        tag_end = ""
    
    # Escape special regex characters in tags
    escaped_start = re.escape(tag_start) if tag_start else ""
    escaped_end = re.escape(tag_end) if tag_end else ""
    
    # Case 1: Perfect match - both tags present and properly paired
    if tag_start and tag_end:
        pattern = f"{escaped_start}(.*?){escaped_end}"
        perfect_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if perfect_match:
            return True, perfect_match.group(1).strip()
    
    if parse_mode == "relaxed":
        # Case 2: Only start tag present
        if tag_start and tag_start in response:
            parsed_response = response.split(tag_start)[-1].strip()
            return True, parsed_response
        
        # Case 3: Only end tag present
        if tag_end and tag_end in response:
            parsed_response = response.split(tag_end)[0].strip()
            return True, parsed_response
    
    # No tags found
    return False, response.strip()