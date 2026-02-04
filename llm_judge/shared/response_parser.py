"""
Response parsing utilities for VLM outputs
"""

import json
import re
from typing import Dict, Any, Optional


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text, handling markdown code blocks.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Text: {text[:200]}...")
        return None


def validate_score(score: Any, min_val: int = 1, max_val: int = 5) -> bool:
    """
    Validate that a score is an integer in the expected range.
    
    Args:
        score: Score to validate
        min_val: Minimum valid score
        max_val: Maximum valid score
        
    Returns:
        True if valid, False otherwise
    """
    try:
        score_int = int(score)
        return min_val <= score_int <= max_val
    except (ValueError, TypeError):
        return False


def parse_json_response(
    text: str,
    expected_keys: list,
    score_range: tuple = (1, 5)
) -> Dict[str, Any]:
    """
    Parse and validate JSON response from VLM.
    
    Args:
        text: Raw response text
        expected_keys: List of expected keys in JSON
        score_range: Tuple of (min, max) for score validation
        
    Returns:
        Dict with 'parsed' (dict or None) and 'is_valid' (bool)
    """
    result = {
        'parsed': None,
        'is_valid': False,
        'error': None
    }
    
    # Extract JSON
    parsed = extract_json(text)
    if parsed is None:
        result['error'] = "Failed to extract JSON"
        return result
    
    # Check expected keys
    missing_keys = [key for key in expected_keys if key not in parsed]
    if missing_keys:
        result['error'] = f"Missing keys: {missing_keys}"
        result['parsed'] = parsed
        return result
    
    # Validate scores if they exist
    for key in expected_keys:
        if isinstance(parsed[key], dict) and 'score' in parsed[key]:
            if not validate_score(parsed[key]['score'], score_range[0], score_range[1]):
                result['error'] = f"Invalid score in {key}: {parsed[key]['score']}"
                result['parsed'] = parsed
                return result
    
    result['parsed'] = parsed
    result['is_valid'] = True
    return result
