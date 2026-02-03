# T2I Models Added - Summary

## Models Added to `models/T2I/`

### 1. flux-dev.py ✅
- **Model**: `black-forest-labs/FLUX.1-dev`
- **Type**: State-of-the-art text-to-image generation
- **Default Steps**: 50
- **Default Guidance**: 3.5
- **Quality**: State-of-the-art
- **Note**: Requires Hugging Face authentication (gated model)

### 2. qwen-image-2512.py ✅
- **Model**: `Qwen/Qwen-Image-2512`
- **Type**: Vision-language text-to-image model
- **Default Steps**: 50
- **Default Guidance**: 7.5
- **Quality**: Very High
- **Features**: Excellent multilingual and cross-cultural support
- **Special**: Ideal for CUBE evaluation with diverse cultural concepts

### 3. flux-schnell.py (Optional)
- **Model**: `black-forest-labs/FLUX.1-schnell`
- **Type**: Fast distilled text-to-image
- **Default Steps**: 4
- **Default Guidance**: 0.0
- **Quality**: High
- **Speed**: Very fast

### 4. sdxl.py (Optional)
- **Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Type**: High-quality text-to-image baseline
- **Default Steps**: 50
- **Default Guidance**: 7.5
- **Quality**: Very High

## Files Updated/Created

### New Files
1. ✅ `models/T2I/flux-dev.py` - FLUX.1 Dev model
2. ✅ `models/T2I/qwen-image-2512.py` - Qwen-Image-2512 model
3. ✅ `models/T2I/flux-schnell.py` - FLUX.1 Schnell model
4. ✅ `models/T2I/sdxl.py` - SDXL model
5. ✅ `models/T2I/__init__.py` - Package init
6. ✅ `models/T2I/README.md` - Model documentation
7. ✅ `eval/T2I_cube.py` - Main evaluation script
8. ✅ `eval/CUBE_1k/README.md` - Usage documentation
9. ✅ `eval/CUBE_1k/QUICKSTART.md` - Quick start guide
10. ✅ `eval/CUBE_1k/run_all_models.sh` - Script to run all models
11. ✅ `eval/CUBE_1k/config_example.yaml` - Example configuration
12. ✅ `eval/CUBE_1k/requirements.txt` - Python dependencies

### Updated Files
- ✅ All documentation updated to prioritize flux-dev and qwen-image-2512
- ✅ Default model changed to `flux-dev`
- ✅ Run script configured for both primary models

## Usage

### Quick Start
```bash
# Install dependencies
cd eval/CUBE_1k
pip install -r requirements.txt

# Run with FLUX.1 Dev
python T2I_cube.py --model flux-dev

# Run with Qwen-Image-2512
python T2I_cube.py --model qwen-image-2512

# Run all primary models
./run_all_models.sh
```

### Command Line Examples
```bash
# Basic run with flux-dev
python T2I_cube.py

# Run with Qwen model
python T2I_cube.py --model qwen-image-2512

# Debug mode (20 samples only)
python T2I_cube.py --model flux-dev --debug

# Custom parameters
python T2I_cube.py \
  --model flux-dev \
  --num_inference_steps 50 \
  --guidance_scale 3.5 \
  --seed 42 \
  --max_samples 100
```

## Output Structure

```
eval/CUBE_1k/outputs/
├── flux-dev/
│   ├── brazil_cuisine_carne_de_panela.png
│   ├── india_cuisine_biryani.png
│   ├── japan_cuisine_sushi.png
│   ├── ...
│   └── metadata.json
└── qwen-image-2512/
    ├── brazil_cuisine_carne_de_panela.png
    ├── india_cuisine_biryani.png
    ├── japan_cuisine_sushi.png
    ├── ...
    └── metadata.json
```

## Metadata Format

Each `metadata.json` contains:
```json
[
  {
    "name": "carne de panela",
    "country": "Brazil",
    "domain": "cuisine",
    "prompt": "A high resolution image of carne de panela from Brazilian cuisine, realistic",
    "image_path": "outputs/flux-dev/brazil_cuisine_carne_de_panela.png",
    "status": "success"
  }
]
```

## Key Features

✅ **Primary Models**: flux-dev and qwen-image-2512  
✅ **Resume Support**: Skips existing images  
✅ **Incremental Saving**: Metadata saved after each image  
✅ **Error Handling**: CUDA OOM and other errors logged  
✅ **Progress Tracking**: Detailed logs  
✅ **Reproducible**: Seed control  
✅ **Cultural Focus**: Qwen model excels at diverse cultural content  

## Model Comparison

| Model | Steps | Quality | Speed | Cultural Support | Auth Required |
|-------|-------|---------|-------|------------------|---------------|
| **flux-dev** | 50 | ⭐⭐⭐⭐⭐ | Slow | Good | Yes |
| **qwen-image-2512** | 50 | ⭐⭐⭐⭐ | Slow | Excellent | No |
| flux-schnell | 4 | ⭐⭐⭐ | Very Fast | Good | No |
| sdxl | 50 | ⭐⭐⭐⭐ | Slow | Moderate | No |

## Recommended Workflow

1. **Start with Qwen-Image-2512** (no auth required, great cultural support)
   ```bash
   python T2I_cube.py --model qwen-image-2512 --debug
   ```

2. **Then try FLUX.1 Dev** (requires Hugging Face login)
   ```bash
   huggingface-cli login
   python T2I_cube.py --model flux-dev --debug
   ```

3. **Run full evaluation on both**
   ```bash
   ./run_all_models.sh
   ```

## Notes

- **FLUX.1 Dev requires authentication**: Login with `huggingface-cli login`
- **Qwen-Image-2512** is particularly strong for cross-cultural content
- Both models support 1024x1024 generation
- Metadata matches CUBE_1k format: name, country, domain, prompt
- Similar pattern to I2I_transcreation.py but for T2I generation

## Next Steps

After running the models, you can:
1. Evaluate outputs using `llm_judge/` scripts
2. Compare cultural representation across models
3. Analyze per-country and per-domain performance
4. Use metadata.json for downstream evaluation pipelines
