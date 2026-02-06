#!/usr/bin/env python3
"""
Main evaluation runner for LLM Judge tasks.

Usage:
    python run_evaluation.py \\
        --config configs/image_transcreation.yaml \\
        --model flux2-klein \\
        --country brazil \\
        --output results/flux2-klein_brazil.json
"""

import argparse
import yaml
import sys
from pathlib import Path

# Import task modules
from tasks import image_transcreation, t2i_cube


# Task registry
TASK_MAP = {
    'image_transcreation': image_transcreation.run_evaluation,
    't2i_cube': t2i_cube.run_evaluation,
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM evaluation for various tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to task configuration YAML file'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model name to evaluate (e.g., flux2-klein, instructpix2pix)'
    )
    parser.add_argument(
        '--country',
        help='Country/culture for evaluation (for image_transcreation task)'
    )
    parser.add_argument(
        '--vlm',
        help='VLM model to use (overrides config). E.g., Qwen/Qwen2-VL-7B-Instruct'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for results JSON file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    task_name = config['task']
    
    # Override VLM if specified
    if args.vlm:
        config['vlm'] = args.vlm
        print(f"Overriding VLM with: {args.vlm}")
    
    # Get task function
    if task_name not in TASK_MAP:
        print(f"Error: Unknown task '{task_name}'")
        print(f"Available tasks: {list(TASK_MAP.keys())}")
        sys.exit(1)
    
    task_fn = TASK_MAP[task_name]
    
    # Run task with appropriate arguments
    if task_name == 'image_transcreation':
        if not args.country:
            print("Error: --country is required for image_transcreation task")
            sys.exit(1)
        
        task_fn(
            config=config,
            model_name=args.model,
            country=args.country,
            output_path=args.output
        )
    else:
        # Generic task runner (for future tasks)
        task_fn(
            config=config,
            model_name=args.model,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
