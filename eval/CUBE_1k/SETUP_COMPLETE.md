# T2I CUBE Evaluation - Complete Setup Summary

## ✅ What Has Been Created

I've successfully created a complete T2I (Text-to-Image) evaluation system for the CUBE_1k dataset with your requested models: **FLUX.1-dev** and **Qwen-Image-2512**.

## 📁 File Structure

```
models/T2I/                           # Text-to-Image models
├── flux-dev.py                       # ✅ FLUX.1-dev (black-forest-labs/FLUX.1-dev)
├── qwen-image-2512.py                # ✅ Qwen-Image-2512 (Qwen/Qwen-Image-2512)
├── flux-schnell.py                   # Optional: Fast FLUX variant
├── sdxl.py                           # Optional: SDXL baseline
├── __init__.py                       # Package initialization
└── README.md                         # Model documentation

eval/CUBE_1k/
├── T2I_cube.py                       # ✅ Main evaluation script
├── data/
│   └── cube_1k.json                  # Input: 1000 prompts (existing)
├── outputs/                          # Generated images will be saved here
│   ├── flux-dev/
│   └── qwen-image-2512/
├── README.md                         # ✅ Complete usage guide
├── QUICKSTART.md                     # ✅ Quick start guide
├── SETUP_COMPLETE.md                 # ✅ This file
├── requirements.txt                  # ✅ Python dependencies
├── config_example.yaml               # ✅ Example configuration
├── run_all_models.sh                 # ✅ Script to run both models
└── test_setup.sh                     # ✅ Setup verification script
```

## 🎯 Primary Models (As Requested)

### 1. FLUX.1-dev (`models/T2I/flux-dev.py`)
- **Model ID**: `black-forest-labs/FLUX.1-dev`
- **Quality**: State-of-the-art
- **Steps**: 50 (default)
- **Guidance**: 3.5 (default)
- **Note**: Requires Hugging Face authentication

### 2. Qwen-Image-2512 (`models/T2I/qwen-image-2512.py`)
- **Model ID**: `Qwen/Qwen-Image-2512`
- **Quality**: Very High
- **Steps**: 50 (default)
- **Guidance**: 7.5 (default)
- **Special**: Excellent for multilingual/cultural content

## 🚀 How to Use

### Step 1: Install Dependencies
```bash
cd eval/CUBE_1k
pip install -r requirements.txt
```

### Step 2: Run Individual Model
```bash
# Run FLUX.1-dev (requires HF login first)
huggingface-cli login
python T2I_cube.py --model flux-dev

# Run Qwen-Image-2512
python T2I_cube.py --model qwen-image-2512
```

### Step 3: Run Both Models
```bash
./run_all_models.sh
```

### Step 4: Debug Mode (Test First)
```bash
# Test with 20 samples only
python T2I_cube.py --model qwen-image-2512 --debug
```

## 📊 Output Format

The script generates:

1. **Images**: `outputs/{model_name}/{country}_{domain}_{name}.png`
2. **Metadata**: `outputs/{model_name}/metadata.json`

### Metadata Structure (as requested)
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

## 🔧 Command Line Options

```bash
python T2I_cube.py \
  --model flux-dev \                    # flux-dev or qwen-image-2512
  --cube_data data/cube_1k.json \
  --output_dir outputs \
  --num_inference_steps 50 \            # Generation steps
  --guidance_scale 3.5 \                # Guidance scale
  --seed 42 \                           # Random seed
  --height 1024 \                       # Image height
  --width 1024 \                        # Image width
  --debug \                             # Process only 20 samples
  --max_samples 100                     # Limit to N samples
```

## 📋 Complete Feature List

✅ **Requested Models Implemented**
- FLUX.1-dev (black-forest-labs/FLUX.1-dev)
- Qwen-Image-2512 (Qwen/Qwen-Image-2512)

✅ **Dataset Support**
- Reads `cube_1k.json` with 1000 prompts
- Extracts: name, country, domain, prompt

✅ **Output Management**
- Saves generated images
- Saves metadata.json with all fields
- Incremental saving (resume support)
- Skip existing images

✅ **Error Handling**
- CUDA OOM errors caught
- Download failures logged
- All errors tracked in metadata

✅ **Similar to I2I_transcreation.py**
- Same code structure
- Similar command-line interface
- Same error handling patterns
- Metadata tracking

## 📚 Documentation Files

1. **README.md** - Complete usage documentation
2. **QUICKSTART.md** - Quick start guide with examples
3. **MODELS_ADDED.md** - Detailed model information
4. **models/T2I/README.md** - Model-specific documentation
5. **config_example.yaml** - Configuration template

## 🧪 Testing

Run the verification script:
```bash
cd eval/CUBE_1k
./test_setup.sh
```

## 💡 Recommended Workflow

1. **Test with Qwen first** (no auth required):
   ```bash
   python T2I_cube.py --model qwen-image-2512 --debug
   ```

2. **Then test FLUX.1-dev** (requires login):
   ```bash
   huggingface-cli login
   python T2I_cube.py --model flux-dev --debug
   ```

3. **Run full evaluation**:
   ```bash
   ./run_all_models.sh
   ```

## 🎨 Model Comparison

| Model | Quality | Speed | Cultural Support | Auth Required |
|-------|---------|-------|------------------|---------------|
| FLUX.1-dev | ⭐⭐⭐⭐⭐ | Slow | Good | Yes (HF) |
| Qwen-Image-2512 | ⭐⭐⭐⭐ | Slow | Excellent | No |

**Recommendation**: Start with Qwen-Image-2512 for its excellent cultural understanding and no authentication requirement.

## 🔗 Integration with Existing Code

The T2I evaluation follows the same pattern as your I2I_transcreation.py:

| Feature | I2I_transcreation.py | T2I_cube.py |
|---------|---------------------|-------------|
| Input | Images + prompts | Prompts only |
| Models | `models/I2I/` | `models/T2I/` |
| Dataset | Country JSONs | CUBE_1k JSON |
| Output | CSV metadata | JSON metadata |
| Structure | Same pattern | Same pattern |

## 📝 Next Steps

After generation, you can:
1. Use `llm_judge/` to evaluate outputs
2. Compare model performance across cultures
3. Analyze per-domain results
4. Feed metadata into evaluation pipelines

## ❓ Troubleshooting

**CUDA Out of Memory**:
```bash
python T2I_cube.py --height 512 --width 512
```

**Missing Dependencies**:
```bash
pip install torch diffusers transformers accelerate pillow
```

**FLUX.1-dev Authentication**:
```bash
huggingface-cli login
# Then accept the model license on HuggingFace website
```

---

**All files are ready to use!** Start with `python T2I_cube.py --model qwen-image-2512 --debug` to test the setup.
