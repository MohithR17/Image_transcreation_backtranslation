# Qwen-VL LLM-as-Judge Framework

A flexible framework for evaluating images using Qwen-VL as an LLM judge. Supports customizable evaluation criteria and prompts.

## Features

- 🎯 **Predefined Templates**: Cultural appropriateness, image quality, instruction following, transcreation quality
- 🔧 **Customizable**: Define your own evaluation criteria and prompts
- 📊 **Batch Processing**: Evaluate multiple images efficiently
- 💾 **Structured Output**: JSON results with scores and justifications
- 🔄 **Flexible Input**: Support for image paths, URLs, or file lists

## Installation

```bash
pip install transformers torch pillow pyyaml requests tqdm qwen-vl-utils
```

For GPU acceleration:
```bash
pip install accelerate
```

## Quick Start

### 1. Basic Usage with Template

```bash
python qwen_vl_judge.py \
    --config configs/cultural_eval.yaml \
    --output results/cultural_scores.json
```

### 2. Custom Evaluation

```bash
python qwen_vl_judge.py \
    --config configs/custom_eval.yaml \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output results/custom_scores.json
```

## Configuration

### Using Predefined Templates

```yaml
# configs/cultural_eval.yaml
template: "cultural_appropriateness"

template_vars:
  target_culture: "Japanese"

image_paths:
  - "./path/to/image1.jpg"
  - "./path/to/image2.jpg"

max_tokens: 512
temperature: 0.1
```

### Available Templates

1. **`cultural_appropriateness`**: Evaluates cultural sensitivity and accuracy
2. **`image_quality`**: Assesses technical quality
3. **`instruction_following`**: Checks instruction adherence
4. **`cultural_transcreation`**: Evaluates cross-cultural adaptation

### Custom Prompts

```yaml
# configs/custom_eval.yaml
prompt: |
  Evaluate this image based on:
  1. Criterion A
  2. Criterion B
  3. Criterion C
  
  Provide:
  Score: [1-10]
  Justification: [explanation]

image_paths: "./image_list.txt"
```

## Programmatic Usage

```python
from qwen_vl_judge import QwenVLJudge

# Initialize judge
judge = QwenVLJudge(model_name="Qwen/Qwen2-VL-7B-Instruct")

# Create custom prompt
prompt = judge.create_evaluation_prompt(
    task_description="assess cultural appropriateness",
    evaluation_criteria=[
        "Respectful representation",
        "Cultural accuracy",
        "No stereotypes"
    ],
    scoring_scale="1-5"
)

# Evaluate single image
result = judge.evaluate_image(
    image_path="./image.jpg",
    prompt=prompt
)

print(f"Score: {result['score']}")
print(f"Justification: {result['justification']}")

# Batch evaluation
results = judge.batch_evaluate(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    prompts=prompt,
    save_path="results.json"
)
```

## Evaluating from Metadata CSV

Use the helper script to evaluate images from the I2I transcreation output:

```bash
python evaluate_from_metadata.py \
    --metadata ./outputs/part1/instructpix2pix/japan/metadata.csv \
    --config configs/cultural_eval.yaml \
    --output results/japan_cultural_scores.json
```

## Output Format

```json
[
  {
    "image_path": "./image1.jpg",
    "prompt": "Evaluation prompt...",
    "raw_response": "Score: 4\nJustification: The image...",
    "score": 4.0,
    "justification": "The image shows appropriate cultural elements..."
  },
  ...
]
```

## Models

Supported Qwen-VL models:
- `Qwen/Qwen2-VL-2B-Instruct` (smaller, faster)
- `Qwen/Qwen2-VL-7B-Instruct` (default, balanced)
- `Qwen/Qwen2-VL-72B-Instruct` (best quality, requires more GPU)

## Tips

1. **Temperature**: Use lower values (0.1-0.3) for more consistent scoring
2. **Max Tokens**: 512 is usually enough for score + justification
3. **Batch Size**: Process in batches if memory limited
4. **Custom Criteria**: Be specific and measurable in your evaluation criteria

## Example Workflows

### Evaluate Cultural Transcreation
```bash
# 1. Generate transcreated images
python I2I_trancreation.py --config configs/part1/japan.yaml --model instructpix2pix

# 2. Evaluate results
python qwen_vl_judge.py --config configs/transcreation_eval.yaml --output results/japan_scores.json

# 3. Analyze scores
python analyze_scores.py --input results/japan_scores.json
```

### Compare Multiple Models
```bash
# Evaluate InstructPix2Pix output
python evaluate_from_metadata.py \
    --metadata ./outputs/part1/instructpix2pix/japan/metadata.csv \
    --config configs/cultural_eval.yaml \
    --output results/instructpix2pix_scores.json

# Evaluate MagicBrush output
python evaluate_from_metadata.py \
    --metadata ./outputs/part1/magicbrush/japan/metadata.csv \
    --config configs/cultural_eval.yaml \
    --output results/magicbrush_scores.json
```

## License

Same as parent project.
