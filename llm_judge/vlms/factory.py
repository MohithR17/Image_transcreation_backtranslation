"""
VLM Model Factory - Loads the appropriate VLM evaluator based on model name
"""

from .qwen_vl import QwenVLEvaluator


# VLM Registry - maps model names to evaluator classes
VLM_REGISTRY = {
    'qwen': QwenVLEvaluator,
    'qwen2-vl': QwenVLEvaluator,
    'qwen-vl': QwenVLEvaluator,
    # Add more VLMs here:
    # 'gemini': GeminiEvaluator,
    # 'gpt4v': GPT4VEvaluator,
}


def get_vlm_evaluator(model_name: str, **kwargs):
    """
    Get the appropriate VLM evaluator for the given model.
    
    Args:
        model_name: Full model name (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        **kwargs: Additional arguments to pass to evaluator
        
    Returns:
        VLM evaluator instance
        
    Example:
        evaluator = get_vlm_evaluator("Qwen/Qwen2-VL-7B-Instruct")
        evaluator = get_vlm_evaluator("gemini-1.5-pro")
    """
    # Determine VLM type from model name
    model_lower = model_name.lower()
    
    vlm_type = None
    for key in VLM_REGISTRY.keys():
        if key in model_lower:
            vlm_type = key
            break
    
    if vlm_type is None:
        raise ValueError(
            f"Unknown VLM model: {model_name}. "
            f"Supported VLMs: {list(VLM_REGISTRY.keys())}"
        )
    
    # Get evaluator class and instantiate
    evaluator_class = VLM_REGISTRY[vlm_type]
    return evaluator_class(model_name=model_name, **kwargs)
