import re

def parse_model_response(response: str):
    """Extract <think>, <accuracy>, <completeness>, <format>, <writing>, <faithfulness> from the response string."""
    try:
        # Step 1: Extract tag contents (accept both full and short tag spellings)
        patterns = {
            "think": r"<think>(.*?)</think>",
            "accuracy": r"<accuracy>(.*?)</accuracy>",
            "completeness": r"<(?:completeness|complete)>(.*?)</(?:completeness|complete)>",
            "format": r"<format>(.*?)</format>",
            "writing": r"<(?:writing|write)>(.*?)</(?:writing|write)>",
            "faithfulness": r"<(?:faithfulness|faith)>(.*?)</(?:faithfulness|faith)>",
        }

        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if not match:
                return None  # missing tag
            extracted[key] = match.group(1).strip()

        # Step 2: Extract integer scores (0/1/2) anywhere in the text for the five score tags
        score_tags = ["accuracy", "completeness", "format", "writing", "faithfulness"]
        for tag in score_tags:
            score_match = re.findall(r"\b([0-2])\b", extracted[tag])
            if not score_match or len(score_match) != 1:
                return None  # must have exactly one valid score
            extracted[tag] = int(score_match[0])
        
        # Step 3: Normalize scores to 0 / 0.5 / 1
        for tag in score_tags:
            extracted[tag] = extracted[tag] / 2  # 0->0, 1->0.5, 2->1
        
        return extracted
    
    except Exception:
        return None
