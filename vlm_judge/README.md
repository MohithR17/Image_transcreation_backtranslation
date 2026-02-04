# LLM Judge

VLM-based evaluation system for multimodal tasks.

## Structure

```
llm_judge/
├── vlms/                 # VLM model implementations
│   └── qwen_vl.py       # Qwen2-VL evaluator
├── shared/              # Shared utilities
│   ├── data_loader.py   # Load CSV/JSON metadata
│   └── response_parser.py  # Parse VLM responses
├── tasks/               # Task-specific evaluation logic
│   └── image_transcreation.py
├── configs/             # Task configurations
│   └── image_transcreation.yaml
├── scripts/             # Batch processing scripts
│   └── batch_image_transcreation.sh
├── run_evaluation.py    # Main entry point
└── requirements.txt
```

## Installation

```bash
cd llm_judge
pip install -r requirements.txt
```

## Usage

### Single Evaluation

```bash
python run_evaluation.py \
  --config configs/image_transcreation.yaml \
  --model flux2-klein \
  --country brazil \
  --output results/flux2-klein_brazil.json
```

### Override VLM Judge

```bash
python run_evaluation.py \
  --config configs/image_transcreation.yaml \
  --model flux2-klein \
  --country brazil \
  --vlm "Qwen/Qwen2-VL-2B-Instruct" \
  --output results/flux2-klein_brazil.json
```

### Batch Evaluation (All Countries)

```bash
chmod +x scripts/batch_image_transcreation.sh
./scripts/batch_image_transcreation.sh flux2-klein
```

## Output Format

Results directory structure:
```
results/
└── image_transcreation/
    └── {vlm_name}/          # e.g., "default", "Qwen2-VL-2B", "Qwen2-VL-72B"
        └── {i2i_model}/     # e.g., "flux2-klein", "instructpix2pix"
            ├── brazil.json
            ├── india.json
            ├── japan.json
            └── ...
```

Each JSON file contains:

```json
{
  "config": {
    "task": "image_transcreation",
    "i2i_model": "flux2-klein",
    "target_culture": "brazil",
    "vlm": "Qwen/Qwen2-VL-7B-Instruct",
    "vlm_short": "Qwen2-VL-7B"
  },
  "metrics": {
    "total": 150,
    "valid": 148,
    "valid_rate": 0.987,
    "A_source_cultural_appropriateness_mean": 2.45,
    "B_adapted_cultural_appropriateness_mean": 4.12,
    "overall_success_mean": 4.15,
    "overall_success_success_rate": 0.798
  },
  "results": [...]
}
```

## Adding New Tasks

1. Create `tasks/new_task.py` with `run_evaluation(config, model_name, output_path)` function
2. Create `configs/new_task.yaml`
3. Add to `TASK_MAP` in `run_evaluation.py`
4. Create `scripts/batch_new_task.sh` if needed

## Adding New VLMs

1. Create `vlms/new_vlm.py` with class implementing:
   - `load_model()`
   - `evaluate(system_prompt, user_prompt, image_paths)`
   - `cleanup()`
2. Update config to use new VLM model name
