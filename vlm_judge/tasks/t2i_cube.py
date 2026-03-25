"""
T2I CUBE Evaluation Task

Evaluates text-to-image generation for cultural concepts using VLM.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.data_loader import load_json_metadata
from shared.response_parser import parse_json_response
from vlms import get_vlm_evaluator


# System prompt for the VLM
SYSTEM_PROMPT = """You are an expert multimodal evaluator specializing in cultural reasoning and visual understanding. Your task is to evaluate whether a generated image accurately represents a culturally-specific concept from a particular country."""


# User prompt template
USER_PROMPT_TEMPLATE = """You are evaluating a text-to-image generation task.

**Concept:** {name}
**Country:** {country}
**Domain:** {domain}
**Text Prompt:** "{prompt}"

**Generated Image:** (provided)

Evaluate the generated image using the three criteria below:

**1. Cultural Relevance:** Based solely on the image, does the item depicted belong to {country}? Answer with "Yes", "No", or "Maybe" and provide brief reasoning (1-2 sentences) explaining why.

**2. Faithfulness (1-5 Likert scale):** If the image is from {country}, how well does it match the item in the text description? Score from 1 (very poor match) to 5 (excellent match) and provide reasoning (1-2 sentences). If Cultural Relevance is "No", you may give a low score here.

**3. Realism (1-5 Likert scale):** How realistic does the image look, regardless of faithfulness? Score from 1 (very unrealistic/artificial) to 5 (photorealistic) and provide reasoning (1-2 sentences).

Return JSON only in the exact format:
{{
  "cultural_relevance": {{"answer": "Yes/No/Maybe", "reason": "..."}},
  "faithfulness": {{"score": 1-5, "reason": "..."}},
  "realism": {{"score": 1-5, "reason": "..."}}
}}"""


def build_prompts(metadata_row: Dict[str, Any]) -> tuple:
    """
    Build system and user prompts for evaluation.
    
    Args:
        metadata_row: Metadata dict with name, country, domain, prompt
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        name=metadata_row.get('name', 'Unknown'),
        country=metadata_row.get('country', 'Unknown'),
        domain=metadata_row.get('domain', 'Unknown'),
        prompt=metadata_row.get('prompt', 'Unknown')
    )
    
    return SYSTEM_PROMPT, user_prompt


def parse_response(text: str) -> Dict[str, Any]:
    """
    Parse VLM response for T2I CUBE evaluation.
    
    Args:
        text: Raw VLM response
        
    Returns:
        Dict with parsed response and validity flag
    """
    expected_keys = [
        'cultural_relevance',
        'faithfulness',
        'realism'
    ]
    
    return parse_json_response(text, expected_keys, score_range=(1, 5))


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from evaluation results.
    
    Args:
        results: List of evaluation result dicts
        
    Returns:
        Dict with aggregated metrics
    """
    valid_results = [r for r in results if r.get('is_valid', False)]
    
    if not valid_results:
        return {
            'total': len(results),
            'valid': 0,
            'valid_rate': 0.0
        }
    
    metrics = {
        'total': len(results),
        'valid': len(valid_results),
        'valid_rate': len(valid_results) / len(results)
    }
    
    # Compute Cultural Relevance distribution (Yes/No/Maybe)
    cultural_relevance_answers = []
    for r in valid_results:
        parsed = r.get('parsed_response', {})
        if 'cultural_relevance' in parsed and 'answer' in parsed['cultural_relevance']:
            answer = parsed['cultural_relevance']['answer']
            cultural_relevance_answers.append(answer)
    
    if cultural_relevance_answers:
        total_cr = len(cultural_relevance_answers)
        metrics['cultural_relevance_yes'] = sum(1 for a in cultural_relevance_answers if a.lower() == 'yes') / total_cr
        metrics['cultural_relevance_no'] = sum(1 for a in cultural_relevance_answers if a.lower() == 'no') / total_cr
        metrics['cultural_relevance_maybe'] = sum(1 for a in cultural_relevance_answers if a.lower() == 'maybe') / total_cr
    
    # Compute mean scores for Likert scale metrics
    criteria = [
        'faithfulness',
        'realism'
    ]
    
    for criterion in criteria:
        scores = []
        for r in valid_results:
            parsed = r.get('parsed_response', {})
            if criterion in parsed and 'score' in parsed[criterion]:
                scores.append(parsed[criterion]['score'])
        
        if scores:
            metrics[f'{criterion}_mean'] = sum(scores) / len(scores)
            metrics[f'{criterion}_success_rate'] = sum(1 for s in scores if s >= 4) / len(scores)
    
    return metrics


def run_evaluation(
    config: Dict[str, Any],
    model_name: str,
    output_path: str
):
    """
    Run T2I CUBE evaluation.
    
    Args:
        config: Configuration dict from YAML
        model_name: T2I model name (FLUX.1-dev, sdxl, etc.)
        output_path: Where to save results
    """
    print(f"\n{'='*60}")
    print(f"T2I CUBE Evaluation")
    print(f"{'='*60}")
    print(f"T2I Model: {model_name}")
    print(f"VLM Judge: {config['vlm']}")
    print(f"{'='*60}\n")
    
    # Build metadata path
    metadata_path = config['metadata']['path'].format(model_name=model_name)
    
    print(f"Loading metadata: {metadata_path}")
    
    # Load metadata
    metadata = load_json_metadata(metadata_path)
    
    # Convert relative image paths to absolute paths
    # The metadata JSON contains paths relative to the CUBE_1k directory
    cube_base_dir = Path(__file__).parent.parent.parent / "eval" / "CUBE_1k"
    
    for row in metadata:
        if 'image_path' in row and not row['image_path'].startswith('/'):
            # Remove "outputs/" prefix if present and resolve
            img_path = row['image_path']
            if img_path.startswith('outputs/'):
                img_path = img_path[8:]  # Remove "outputs/"
            row['image_path'] = str(cube_base_dir / "outputs" / img_path)
    
    print(f"✓ Loaded {len(metadata)} samples\n")
    
    # Check if already completed
    output_file = Path(output_path)
    if output_file.exists():
        print(f"⚠️  Output file already exists: {output_path}")
        print("Evaluation already complete. Skipping.")
        print("Delete the file to re-evaluate.\n")
        return
    
    # Load VLM
    vlm_model = config['vlm']
    vlm_short = vlm_model.split('/')[-1].replace('-Instruct', '') if '/' in vlm_model else vlm_model
    
    print(f"Initializing VLM: {vlm_model}")
    evaluator = get_vlm_evaluator(model_name=vlm_model)
    evaluator.load_model()
    
    # Evaluation loop
    results = []
    checkpoint_interval = 10
    
    # Check if checkpoint exists to resume
    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.json"
    
    start_idx = 0
    if checkpoint_file.exists():
        print(f"Found checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get('results', [])
            start_idx = len(results)
        print(f"Resuming from sample {start_idx}\n")
    
    for idx, row in enumerate(tqdm(metadata[start_idx:], desc="Evaluating", initial=start_idx, total=len(metadata))):
        actual_idx = start_idx + idx
        
        # Build prompts
        system_prompt, user_prompt = build_prompts(row)
        
        # Get image path (single image for T2I)
        image_paths = [row['image_path']]
        
        # Call VLM
        try:
            response = evaluator.evaluate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
                max_tokens=config['generation']['max_tokens'],
                temperature=config['generation']['temperature']
            )
            
            # Parse response
            parsed = parse_response(response)
            
            # Store result
            result = {
                'index': actual_idx,
                'metadata': row,
                'raw_response': response,
                'parsed_response': parsed['parsed'],
                'is_valid': parsed['is_valid'],
                'error': parsed.get('error')
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing sample {actual_idx}: {e}")
            results.append({
                'index': actual_idx,
                'metadata': row,
                'error': str(e),
                'is_valid': False
            })
        
        # Save checkpoint every N samples
        if (actual_idx + 1) % checkpoint_interval == 0:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'config': {
                        'task': 't2i_cube',
                        't2i_model': model_name,
                        'vlm': vlm_model,
                        'vlm_short': vlm_short
                    },
                    'results': results,
                    'checkpoint': True
                }, f, indent=2)
            print(f"\n✓ Checkpoint saved ({actual_idx + 1}/{len(metadata)} samples)")
    
    # Remove checkpoint file after completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("✓ Checkpoint removed (evaluation complete)")
    
    # Cleanup
    evaluator.cleanup()
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing metrics...")
    metrics = compute_metrics(results)
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'config': {
            'task': 't2i_cube',
            't2i_model': model_name,
            'vlm': vlm_model,
            'vlm_short': vlm_short
        },
        'metrics': metrics,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total']}")
    print(f"Valid responses: {metrics['valid']} ({metrics['valid_rate']:.1%})")
    
    # Print Cultural Relevance distribution
    if 'cultural_relevance_yes' in metrics:
        print(f"\nCultural Relevance Distribution:")
        print(f"  Yes:   {metrics.get('cultural_relevance_yes', 0):.1%}")
        print(f"  No:    {metrics.get('cultural_relevance_no', 0):.1%}")
        print(f"  Maybe: {metrics.get('cultural_relevance_maybe', 0):.1%}")
    
    # Print mean scores
    print(f"\nMean Scores:")
    if 'faithfulness_mean' in metrics:
        print(f"  Faithfulness: {metrics['faithfulness_mean']:.2f}")
    if 'realism_mean' in metrics:
        print(f"  Realism:      {metrics['realism_mean']:.2f}")
    
    # Print success rates
    print(f"\nSuccess Rates (score >= 4):")
    if 'faithfulness_success_rate' in metrics:
        print(f"  Faithfulness: {metrics['faithfulness_success_rate']:.1%}")
    if 'realism_success_rate' in metrics:
        print(f"  Realism:      {metrics['realism_success_rate']:.1%}")
    print(f"{'='*60}\n")
