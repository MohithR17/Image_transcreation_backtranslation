# VLM Judge Selection Guide

## How VLM Selection Works

The system uses a **factory pattern** to automatically select the right VLM evaluator based on the model name.

### 1. Default VLM (from config)

Set in `configs/image_transcreation.yaml`:
```yaml
vlm: "Qwen/Qwen2-VL-7B-Instruct"
```

### 2. Override via CLI

```bash
python run_evaluation.py \
  --config configs/image_transcreation.yaml \
  --model flux2-klein \
  --country brazil \
  --vlm "Qwen/Qwen2-VL-2B-Instruct" \
  --output results/brazil.json
```

### 3. Batch with VLM Override

```bash
./scripts/batch_image_transcreation.sh flux2-klein "Qwen/Qwen2-VL-2B-Instruct"
```

## VLM Registry

Located in `vlms/factory.py`, the registry maps model name patterns to evaluator classes:

```python
VLM_REGISTRY = {
    'qwen': QwenVLEvaluator,
    'qwen2-vl': QwenVLEvaluator,
    'qwen-vl': QwenVLEvaluator,
    # Add more:
    # 'gemini': GeminiEvaluator,
    # 'gpt4v': GPT4VEvaluator,
}
```

The factory automatically detects the VLM type by checking if any registry key appears in the model name (case-insensitive).

## Supported VLMs

### Qwen2-VL
- `Qwen/Qwen2-VL-2B-Instruct` (smaller, faster)
- `Qwen/Qwen2-VL-7B-Instruct` (default)
- `Qwen/Qwen2-VL-72B-Instruct` (larger, more capable)

### Ovis2.5
- `AIDC-AI/Ovis2.5-9B` (9B parameter multimodal model)
- `AIDC-AI/Ovis2.5-Llama3.2-3B` (smaller variant)

### Future VLMs
To add a new VLM:

1. **Create evaluator** in `vlms/new_vlm.py`:
```python
class NewVLMEvaluator:
    def __init__(self, model_name, device="cuda"):
        ...
    
    def load_model(self):
        ...
    
    def evaluate(self, system_prompt, user_prompt, image_paths, max_tokens, temperature):
        # Return string response
        ...
    
    def cleanup(self):
        ...
```

2. **Register** in `vlms/factory.py`:
```python
from .new_vlm import NewVLMEvaluator

VLM_REGISTRY = {
    'qwen': QwenVLEvaluator,
    'new-vlm': NewVLMEvaluator,  # Add here
}
```

3. **Use it**:
```bash
python run_evaluation.py \
  --vlm "new-vlm-model-name" \
  ...
```

## Example Workflow

```bash
# Evaluate flux2-klein with default VLM (Qwen2-VL-7B)
./scripts/batch_image_transcreation.sh flux2-klein
# Results: results/image_transcreation/Qwen2-VL-7B/flux2-klein/

# Evaluate instructpix2pix with smaller VLM (2B) for faster testing
./scripts/batch_image_transcreation.sh instructpix2pix "Qwen/Qwen2-VL-2B-Instruct"
# Results: results/image_transcreation/Qwen2-VL-2B/instructpix2pix/

# Evaluate magicbrush with Ovis2.5-9B
./scripts/batch_image_transcreation.sh magicbrush "AIDC-AI/Ovis2.5-9B"
# Results: results/image_transcreation/Ovis2.5-9B/magicbrush/

# Evaluate with larger Qwen VLM (72B) for higher quality
./scripts/batch_image_transcreation.sh qwen-image-edit "Qwen/Qwen2-VL-72B-Instruct"
# Results: results/image_transcreation/Qwen2-VL-72B/qwen-image-edit/
```

## Results Directory Structure

```
results/
└── image_transcreation/
    ├── default/                    # Default VLM from config
    │   ├── flux2-klein/
    │   │   ├── brazil.json
    │   │   ├── india.json
    │   │   └── ...
    │   └── instructpix2pix/
    │       └── ...
    ├── Qwen2-VL-2B/               # Smaller VLM
    │   └── flux2-klein/
    │       └── ...
    └── Qwen2-VL-72B/              # Larger VLM
        └── flux2-klein/
            └── ...
```

This structure allows you to:
- Compare same I2I model evaluated by different VLMs
- Compare different I2I models evaluated by same VLM
- Keep results organized by VLM capability tier

## VLM Selection Flow

```
1. Load config → vlm: "Qwen/Qwen2-VL-7B-Instruct"
                 ↓
2. CLI override? → --vlm "Qwen/Qwen2-VL-2B-Instruct"
   (optional)    ↓
3. Factory pattern → Check model name contains "qwen"
                    ↓
4. Get evaluator → QwenVLEvaluator(model_name="Qwen/Qwen2-VL-2B-Instruct")
                   ↓
5. Load & Evaluate → Returns string response
```
