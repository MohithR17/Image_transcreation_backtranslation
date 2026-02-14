"""
Ovis Model Wrapper for Image Evaluation
Supports AIDC-AI/Ovis2.5-9B and other Ovis models
"""

import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from typing import List
import requests
from io import BytesIO
import os


class OvisEvaluator:
    """Wrapper for Ovis models to evaluate images."""
    
    def __init__(self, model_name: str = "AIDC-AI/Ovis2.5-9B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None
        
    def load_model(self):
        """Load the Ovis model and tokenizers."""
        print(f"Loading VLM: {self.model_name}")
        
        # CRITICAL: Monkey-patch model components BEFORE loading
        # This fixes compatibility with transformers 5.0.0
        
        # Patch 1: Qwen3ForCausalLM (language model)
        try:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            if not hasattr(Qwen3ForCausalLM, 'is_parallelizable'):
                Qwen3ForCausalLM.is_parallelizable = False
                print("  ⚠ Patched Qwen3ForCausalLM.is_parallelizable = False")
            if not hasattr(Qwen3ForCausalLM, '_tied_weights_keys'):
                Qwen3ForCausalLM._tied_weights_keys = []
                print("  ⚠ Patched Qwen3ForCausalLM._tied_weights_keys = []")
        except ImportError:
            print("  ℹ Qwen3ForCausalLM not found in transformers, skipping patch")
        
        # Patch 2: Siglip vision model components (loaded from cache)
        # We need to patch the local Ovis model file's Siglip2NavitModel class
        try:
            import sys
            import importlib
            cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE'))
            
            # The Ovis model loads its own Siglip implementation from cache
            # We need to find and patch it before model initialization
            # Try to import the cached modeling file
            ovis_module_path = None
            if cache_dir:
                import glob
                pattern = f"{cache_dir}/modules/transformers_modules/*/Ovis*/*/modeling_ovis*.py"
                matches = glob.glob(pattern)
                if matches:
                    ovis_module_path = matches[0]
                    module_dir = os.path.dirname(ovis_module_path)
                    if module_dir not in sys.path:
                        sys.path.insert(0, module_dir)
                    
                    # Import and patch
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("modeling_ovis_temp", ovis_module_path)
                    ovis_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ovis_module)
                    
                    # Patch Siglip2NavitModel
                    if hasattr(ovis_module, 'Siglip2NavitModel'):
                        if not hasattr(ovis_module.Siglip2NavitModel, 'is_parallelizable'):
                            ovis_module.Siglip2NavitModel.is_parallelizable = False
                            print("  ⚠ Patched Siglip2NavitModel.is_parallelizable = False")
        except Exception as e:
            print(f"  ⚠ Could not patch Siglip model: {e}")
            print("  ℹ Will try alternative approach...")
        
        # Get cache directory from environment or use default
        cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE'))
        if cache_dir:
            print(f"Using cache directory: {cache_dir}")
        
        # Load Ovis model with multimodal support
        # Note: Using dtype instead of torch_dtype (deprecated warning)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            multimodal_max_length=8192,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )
        
        # Get tokenizers from model (they are attributes, not methods)
        self.text_tokenizer = self.model.text_tokenizer
        self.visual_tokenizer = self.model.visual_tokenizer
        
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
                # Handle both URLs and local file paths
                if path.startswith('http://') or path.startswith('https://'):
                    # Download image from URL
                    response = requests.get(path, timeout=120)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        raise Exception(f"Failed to download image from {path}, status: {response.status_code}")
                else:
                    # Load from local file
                    img = Image.open(path).convert("RGB")
                
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                raise
        
        # Build messages in chat format expected by Ovis
        # Ovis expects: List[Dict] with "role" and "content" keys
        # Content can be a list of dicts with "type" and "image"/"text" keys
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message content with images
        user_content = []
        
        # Add images with "type" field for chat template validation
        for img in images:
            user_content.append({
                "type": "image",
                "image": img  # PIL.Image object
            })
        
        # Add text prompt with "type" field for chat template validation
        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Prepare inputs for the model
        # Returns: (input_ids, pixel_values, grid_thws)
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages
        )
        
        # Move to device
        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)
        
        # Generate response
        with torch.no_grad():
            # Note: Ovis model.generate() creates attention_mask internally, don't pass it
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,  # Required by Ovis model
                max_new_tokens=max_tokens,
                # Don't set max_length - let max_new_tokens handle it
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=True,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id
            )
        
        # Decode output
        # NOTE: When using inputs_embeds (as Ovis does), generate() returns ONLY the generated tokens,
        # not the full sequence including input. So we decode the output directly.
        output_text = self.text_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return output_text.strip()
    
    def cleanup(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.text_tokenizer
            del self.visual_tokenizer
            torch.cuda.empty_cache()
