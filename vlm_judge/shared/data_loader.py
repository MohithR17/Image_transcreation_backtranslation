"""
Data loading utilities for different metadata formats
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_csv_metadata(
    path: str,
    filter_status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load metadata from CSV file.
    
    Args:
        path: Path to CSV file
        filter_status: Optional status filter (e.g., 'success')
        
    Returns:
        List of metadata dictionaries
    """
    df = pd.read_csv(path)
    
    if filter_status:
        df = df[df['status'] == filter_status]
    
    return df.to_dict('records')


def load_json_metadata(path: str) -> List[Dict[str, Any]]:
    """
    Load metadata from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of metadata dictionaries
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # If dict, convert to list of dicts
        return [data]
    else:
        raise ValueError(f"Unexpected JSON format in {path}")
