# Shared Models Directory

This directory contains all models used across different evaluation tasks in the Image Transcreation project.

## Structure

```
models/
├── I2I/                    # Image-to-Image editing models
│   ├── instructpix2pix.py
│   ├── sdxl-instructpix2pix.py
│   ├── cosxl-edit.py
│   ├── magicbrush.py
│   └── README.md
├── T2I/                    # Text-to-Image generation models (future)
└── VLM/                    # Vision-Language models (future)
```

## Model Types

### Image-to-Image (I2I) Models

Located in `models/I2I/`. These models edit existing images based on text instructions.

**Interface:**
```python
def edit_image(image, prompt, config):
    """
    Edit an image based on a text instruction.
    
    Args:
        image: PIL Image to edit
        prompt: Text instruction for editing
        config: Configuration dictionary containing model parameters
        
    Returns:
        PIL Image: Edited image
    """
```

**Available Models:**
- `instructpix2pix` - Fast, SD 1.5 based
- `sdxl-instructpix2pix` - High quality, SDXL based
- `cosxl-edit` - Alternative SDXL-based model
- `magicbrush` - Enhanced editing (requires setup)

See `I2I/README.md` for detailed documentation.

### Text-to-Image (T2I) Models

*Coming soon*

Located in `models/T2I/`. These models generate images from text prompts.

### Vision-Language Models (VLM)

*Coming soon*

Located in `models/VLM/`. These models are used for evaluation and analysis.

## Usage

### From Evaluation Scripts

Evaluation scripts can import models from this shared directory:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import I2I model
from models.I2I import instructpix2pix

# Or dynamically
model_module = __import__(f"models.I2I.{model_name}", fromlist=[''])
model_func = model_module.edit_image
```

### Example: I2I Evaluation

```python
# In eval/I2I_Image_transcreation/I2I_trancreation.py
model_module = __import__(f"models.I2I.{model_name}", fromlist=[''])
edited_image = model_module.edit_image(source_image, prompt, config)
```

## Adding New Models

### For I2I Models:

1. Create a new file in `models/I2I/your_model.py`
2. Implement the `edit_image(image, prompt, config)` function
3. Update `models/I2I/README.md` with model info
4. Use in evaluation: `--model your_model`

### For Other Model Types:

Create the appropriate subdirectory (T2I, VLM, etc.) and follow similar patterns.

## Benefits of Shared Models

✅ **Reusability**: Models can be used across multiple evaluation tasks
✅ **Maintainability**: Single source of truth for model implementations
✅ **Consistency**: Same model behavior across different evaluations
✅ **Organization**: Clear separation between models and evaluation logic
✅ **Extensibility**: Easy to add new model types without duplicating code

## Migration Notes

Models were moved from:
- `eval/I2I_Image_transcreation/models/` → `models/I2I/`

Scripts updated to import from the new location.
