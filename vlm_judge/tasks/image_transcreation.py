"""
Image Transcreation Evaluation Task

Evaluates image-to-image transcreation across cultures using VLM.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.data_loader import load_csv_metadata
from shared.response_parser import parse_json_response
from vlms import get_vlm_evaluator


# System prompt for the VLM
SYSTEM_PROMPT = """You are an expert multimodal evaluator specializing in cultural reasoning and visual understanding. Your task is to evaluate image transcreation - the process of adapting images from one culture to another while preserving semantic meaning.

You will evaluate images based on cultural appropriateness, semantic preservation, and visual quality. Provide objective, detailed assessments."""


# User prompt template
USER_PROMPT_TEMPLATE = """You are evaluating an image transcreation task.

**Source Culture:** {src_country}
**Target Culture:** {target_culture}
**Category:** {category}

**Images:**
- First image: Source image (from {src_country})
- Second image: Transcreated image (adapted for {target_culture})

**Evaluation Criteria:**

A. **Source Cultural Appropriateness (1-5):** Does the source image appropriately represent {src_country} culture in the given category?

B. **Adapted Cultural Appropriateness (1-5):** Does the transcreated image appropriately represent {target_culture} culture while maintaining the concept?

C. **Semantic Preservation (1-5):** Does the transcreated image preserve the core semantic meaning and intent of the source image?

D. **Visual Coherence (1-5):** Is the transcreated image visually coherent, realistic, and of good quality?

E. **Category Cultural Adaptation (1-5):** Was the specific concept/item within the category culturally adapted? For example, if category is food, was "pizza" changed to a culturally appropriate food like "dosa" for Indian culture? Score higher when the actual item is transformed to match target culture, not just superficial background changes.

F. **Physical Plausibility (1-5):** How realistic and natural does the image appear? Are objects properly sized relative to each other? Are elements integrated naturally (not artificially pasted)? Are cultural symbols (flags, decorations) placed sensibly and not nonsensically added? Score lower for unrealistic compositions or improper object scales.

**Overall Success (1-5):** Overall, how successful is this transcreation?

**Response Format:**
Return ONLY a JSON object with this exact structure:
{{
  "A_source_cultural_appropriateness": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "B_adapted_cultural_appropriateness": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "C_semantic_preservation": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "D_visual_coherence": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "E_category_cultural_adaptation": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "F_physical_plausibility": {{"score": <1-5>, "reason": "<brief explanation>"}},
  "overall_success": {{"score": <1-5>, "reason": "<brief explanation>"}}
}}

Provide your evaluation now."""


def build_prompts(metadata_row: Dict[str, Any], target_culture: str) -> tuple:
    """
    Build system and user prompts for evaluation.
    
    Args:
        metadata_row: Metadata dict with src_country, src_category, etc.
        target_culture: Target culture name
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        src_country=metadata_row.get('src_country', 'Unknown'),
        target_culture=target_culture,
        category=metadata_row.get('src_category', 'Unknown')
    )
    
    return SYSTEM_PROMPT, user_prompt


def parse_response(text: str) -> Dict[str, Any]:
    """
    Parse VLM response for image transcreation evaluation.
    
    Args:
        text: Raw VLM response
        
    Returns:
        Dict with parsed response and validity flag
    """
    expected_keys = [
        'A_source_cultural_appropriateness',
        'B_adapted_cultural_appropriateness',
        'C_semantic_preservation',
        'D_visual_coherence',
        'E_category_cultural_adaptation',
        'F_physical_plausibility',
        'overall_success'
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
    
    # Compute mean scores
    criteria = [
        'A_source_cultural_appropriateness',
        'B_adapted_cultural_appropriateness',
        'C_semantic_preservation',
        'D_visual_coherence',
        'E_category_cultural_adaptation',
        'F_physical_plausibility',
        'overall_success'
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
    country: str,
    output_path: str
):
    """
    Run Image Transcreation evaluation.
    
    Args:
        config: Configuration dict from YAML
        model_name: I2I model name (flux2-klein, etc.)
        country: Target country
        output_path: Where to save results (can be overridden to include VLM)
    """
    # Extract VLM name for display
    vlm_model = config['vlm']
    vlm_short = vlm_model.split('/')[-1].replace('-Instruct', '') if '/' in vlm_model else vlm_model
    
    print(f"\n{'='*60}")
    print(f"Image Transcreation Evaluation")
    print(f"{'='*60}")
    print(f"I2I Model: {model_name}")
    print(f"Target Culture: {country}")
    print(f"VLM Judge: {vlm_short}")
    print(f"{'='*60}\n")
    
    # Build metadata path
    metadata_path = config['metadata']['path'].format(
        model_name=model_name,
        country=country
    )
    
    print(f"Loading metadata: {metadata_path}")
    
    # Load metadata
    metadata = load_csv_metadata(
        metadata_path,
        filter_status=config['metadata'].get('filter_status')
    )
    
    # Convert relative image paths to absolute paths
    # The metadata CSV contains paths relative to the I2I_Image_transcreation directory
    # e.g., "./outputs/part1/flux2-klein/brazil/..."
    # We need to resolve them relative to ../eval/I2I_Image_transcreation/
    i2i_base_dir = Path(__file__).parent.parent.parent / "eval" / "I2I_Image_transcreation"
    
    for row in metadata:
        # Only convert target image path if it's a relative path (starts with ./)
        if 'tgt_image_path' in row and row['tgt_image_path'].startswith('./'):
            # Remove the "./" prefix and resolve relative to I2I_Image_transcreation directory
            rel_path = row['tgt_image_path'][2:]  # Remove "./"
            row['tgt_image_path'] = str(i2i_base_dir / rel_path)
    
    print(f"✓ Loaded {len(metadata)} samples\n")
    
    # Load VLM using factory
    print(f"Initializing VLM: {config['vlm']}")
    evaluator = get_vlm_evaluator(model_name=config['vlm'])
    evaluator.load_model()
    
    # Evaluation loop
    results = []
    
    for idx, row in enumerate(tqdm(metadata, desc="Evaluating")):
        # Build prompts
        system_prompt, user_prompt = build_prompts(row, country.replace('-', ' ').title())
        
        # Get image paths
        image_paths = [row['src_image_path'], row['tgt_image_path']]
        
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
                'index': idx,
                'metadata': row,
                'raw_response': response,
                'parsed_response': parsed['parsed'],
                'is_valid': parsed['is_valid'],
                'error': parsed.get('error')
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            results.append({
                'index': idx,
                'metadata': row,
                'error': str(e),
                'is_valid': False
            })
    
    # Cleanup
    evaluator.cleanup()
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing metrics...")
    metrics = compute_metrics(results)
    
    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'config': {
            'task': 'image_transcreation',
            'i2i_model': model_name,
            'target_culture': country,
            'vlm': config['vlm'],
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
    print(f"\nMean Scores:")
    for key, value in metrics.items():
        if key.endswith('_mean'):
            print(f"  {key}: {value:.2f}")
    print(f"\nSuccess Rates (score >= 4):")
    for key, value in metrics.items():
        if key.endswith('_success_rate'):
            print(f"  {key}: {value:.1%}")
    print(f"{'='*60}\n")
