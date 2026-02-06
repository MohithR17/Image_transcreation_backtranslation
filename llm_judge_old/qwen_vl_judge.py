"""
Generic LLM-as-Judge framework using Qwen-VL for image evaluation.
Supports customizable evaluation prompts and scoring criteria.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import argparse
import yaml
import logging
from typing import Dict, List, Union, Optional
import os
import re
import requests
from tqdm import tqdm


class QwenVLJudge:
    """
    A configurable LLM-as-Judge using Qwen-VL for image evaluation.
    
    Supports various evaluation tasks:
    - Cultural appropriateness
    - Image quality assessment
    - Instruction following
    - Cultural representation accuracy
    - Custom evaluation criteria
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        """
        Initialize the Qwen-VL judge.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        
        logging.info(f"Loading Qwen-VL model: {model_name}")
        
        # Load model with appropriate settings
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        logging.info("Model loaded successfully")
    
    def create_evaluation_prompt(self, 
                                 task_description: str,
                                 evaluation_criteria: List[str],
                                 scoring_scale: str = "1-5",
                                 additional_context: str = "") -> str:
        """
        Create a structured evaluation prompt.
        
        Args:
            task_description: Description of what to evaluate
            evaluation_criteria: List of criteria to consider
            scoring_scale: Scale for scoring (e.g., "1-5", "1-10", "binary")
            additional_context: Any additional context for the judge
            
        Returns:
            Formatted evaluation prompt
        """
        criteria_text = "\n".join([f"- {criterion}" for criterion in evaluation_criteria])
        
        prompt = f"""You are an expert evaluator. Your task is to {task_description}.

Evaluation Criteria:
{criteria_text}

{additional_context}

Please provide:
1. A score on a scale of {scoring_scale}
2. A brief justification for your score

Format your response as:
Score: [your score]
Justification: [your explanation]
"""
        return prompt
    
    def evaluate_image(self,
                      image_path: str,
                      prompt: str,
                      max_tokens: int = 512,
                      temperature: float = 0.1) -> Dict[str, Union[str, float]]:
        """
        Evaluate a single image using the provided prompt.
        
        Args:
            image_path: Path to the image file or URL
            prompt: Evaluation prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Dictionary with 'raw_response', 'score', and 'justification'
        """
        # Load image
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # Trim to only new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode response
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse response
        parsed = self._parse_response(response)
        
        return {
            "raw_response": response,
            "score": parsed["score"],
            "justification": parsed["justification"]
        }
    
    def _parse_response(self, response: str) -> Dict[str, Union[float, str]]:
        """
        Parse the model's response to extract score and justification.
        
        Args:
            response: Raw model output
            
        Returns:
            Dictionary with 'score' and 'justification'
        """
        score = None
        justification = ""
        
        # Try to extract score
        score_patterns = [
            r"Score:\s*(\d+\.?\d*)",
            r"score:\s*(\d+\.?\d*)",
            r"Rating:\s*(\d+\.?\d*)",
            r"rating:\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*/\s*\d+",  # Format like "4/5"
            r"^(\d+\.?\d*)",  # Score at the beginning
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Try to extract justification
        justification_patterns = [
            r"Justification:\s*(.+?)(?=\n\n|\Z)",
            r"justification:\s*(.+?)(?=\n\n|\Z)",
            r"Explanation:\s*(.+?)(?=\n\n|\Z)",
            r"explanation:\s*(.+?)(?=\n\n|\Z)",
            r"Reasoning:\s*(.+?)(?=\n\n|\Z)",
        ]
        
        for pattern in justification_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                justification = match.group(1).strip()
                break
        
        # If no justification found, use the whole response
        if not justification:
            justification = response.strip()
        
        return {
            "score": score,
            "justification": justification
        }
    
    def batch_evaluate(self,
                      image_paths: List[str],
                      prompts: Union[str, List[str]],
                      save_path: Optional[str] = None,
                      **kwargs) -> List[Dict]:
        """
        Evaluate multiple images.
        
        Args:
            image_paths: List of image paths
            prompts: Single prompt string or list of prompts (one per image)
            save_path: Optional path to save results as JSON
            **kwargs: Additional arguments for evaluate_image
            
        Returns:
            List of evaluation results
        """
        # Handle single prompt for all images
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_paths)
        
        assert len(image_paths) == len(prompts), "Number of images and prompts must match"
        
        results = []
        for img_path, prompt in tqdm(zip(image_paths, prompts), total=len(image_paths)):
            try:
                result = self.evaluate_image(img_path, prompt, **kwargs)
                results.append({
                    "image_path": img_path,
                    "prompt": prompt,
                    **result
                })
            except Exception as e:
                logging.error(f"Error evaluating {img_path}: {e}")
                results.append({
                    "image_path": img_path,
                    "prompt": prompt,
                    "raw_response": "",
                    "score": None,
                    "justification": f"Error: {str(e)}"
                })
        
        # Save results if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {save_path}")
        
        return results


# Predefined evaluation templates
EVALUATION_TEMPLATES = {
    "cultural_appropriateness": {
        "task_description": "evaluate how culturally appropriate this image is for the target culture",
        "criteria": [
            "Cultural symbols and representations are accurate and respectful",
            "Visual elements align with cultural norms and values",
            "No stereotypical or offensive content",
            "Contextually appropriate for the target culture"
        ],
        "scale": "1-5",
        "context": "Consider cultural sensitivity, authenticity, and appropriateness."
    },
    
    "image_quality": {
        "task_description": "assess the technical quality of this image",
        "criteria": [
            "Visual clarity and sharpness",
            "Appropriate lighting and colors",
            "No visible artifacts or distortions",
            "Professional appearance"
        ],
        "scale": "1-5",
        "context": "Focus on technical aspects like resolution, composition, and visual appeal."
    },
    
    "instruction_following": {
        "task_description": "evaluate how well this image follows the given instruction",
        "criteria": [
            "All aspects of the instruction are addressed",
            "Key elements are correctly represented",
            "No contradictions with the instruction",
            "Accurate interpretation of the intent"
        ],
        "scale": "1-5",
        "context": "The instruction was: {instruction}"
    },
    
    "cultural_transcreation": {
        "task_description": "evaluate how well this image has been transcreated from one culture to another",
        "criteria": [
            "Source cultural elements are appropriately adapted",
            "Target cultural context is well represented",
            "Maintains the essence of the original while fitting the new culture",
            "Culturally sensitive and appropriate"
        ],
        "scale": "1-5",
        "context": "Source culture: {source_culture}, Target culture: {target_culture}"
    },
}


def main():
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Qwen-VL LLM-as-Judge for Image Evaluation")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to evaluation config YAML file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                       help="Qwen-VL model name or path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output path for results")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = QwenVLJudge(model_name=args.model, device=args.device)
    
    # Create evaluation prompt
    if "template" in config and config["template"] in EVALUATION_TEMPLATES:
        template = EVALUATION_TEMPLATES[config["template"]]
        # Format context with any variables from config
        context = template["context"].format(**config.get("template_vars", {}))
        prompt = judge.create_evaluation_prompt(
            task_description=template["task_description"],
            evaluation_criteria=template["criteria"],
            scoring_scale=template["scale"],
            additional_context=context
        )
    else:
        # Use custom prompt from config
        prompt = config.get("prompt", config.get("evaluation_prompt"))
    
    logging.info(f"Evaluation prompt:\n{prompt}\n")
    
    # Load images
    image_paths = config["image_paths"]
    if isinstance(image_paths, str):
        # If it's a file, read paths from it
        if os.path.isfile(image_paths):
            with open(image_paths) as f:
                image_paths = [line.strip() for line in f if line.strip()]
    
    # Run evaluation
    results = judge.batch_evaluate(
        image_paths=image_paths,
        prompts=prompt,
        save_path=args.output,
        max_tokens=config.get("max_tokens", 512),
        temperature=config.get("temperature", 0.1)
    )
    
    # Print summary
    scores = [r["score"] for r in results if r["score"] is not None]
    if scores:
        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluation Summary:")
        logging.info(f"  Total images: {len(results)}")
        logging.info(f"  Successfully evaluated: {len(scores)}")
        logging.info(f"  Average score: {sum(scores)/len(scores):.2f}")
        logging.info(f"  Min score: {min(scores):.2f}")
        logging.info(f"  Max score: {max(scores):.2f}")
        logging.info(f"{'='*50}\n")


if __name__ == "__main__":
    main()
