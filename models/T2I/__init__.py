"""
Text-to-Image (T2I) Models

This package contains text-to-image generation models.

Available models:
- flux-dev: High-quality FLUX.1 Dev model (primary)
- qwen-image-2512: Qwen vision-language T2I model (primary)
- flux-schnell: Fast FLUX.1 Schnell model (optional)
- sdxl: Stable Diffusion XL base model (optional)

Each model module should implement:
    generate_image(prompt, config) -> PIL.Image
"""
