"""
Qwen-VL Model Wrapper for Image Evaluation
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List


class QwenVLEvaluator:
    """Wrapper for Qwen2-VL model to evaluate images."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the Qwen-VL model and processor."""
        print(f"Loading VLM: {self.model_name}")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print(f"✓ Model loaded on {self.device}")
        
    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """
        Evaluate images with given prompts.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query/instruction
            image_paths: List of image file paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Model response as string
        """
        if self.model is None:
            self.load_model()
        
        # Load images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                raise
        
        # Build messages with images
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add user message with images
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})
        
        messages.append({"role": "user", "content": content})
        
        # Process with Qwen-VL
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def cleanup(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
