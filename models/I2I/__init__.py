"""
Image-to-Image (I2I) editing models.

All models in this directory must implement the following interface:

def edit_image(image, prompt, config):
    '''
    Edit an image based on a text instruction.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing model parameters
        
    Returns:
        PIL Image: Edited image
    '''
    pass

Available models:
- instructpix2pix: Fast, SD 1.5 based
- sdxl-instructpix2pix: High quality, SDXL based
- cosxl-edit: Alternative SDXL-based model
- flux2-klein: Fast and efficient FLUX.2 4B model
- magicbrush: Enhanced editing (requires setup)
"""
