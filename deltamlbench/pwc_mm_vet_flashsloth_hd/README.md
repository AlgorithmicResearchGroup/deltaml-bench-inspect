# MM-Vet FlashSloth Task

## Overview

This task focuses on improving the FlashSloth model on the MM-Vet dataset.

## Task Description

**Objective**: Improve the FlashSloth model to achieve better performance on the MM-Vet dataset.

**Paper**: FlashSloth: Lightning Multimodal Large Language Models via Embedded Visual Compression (https://arxiv.org/abs/2412.04317v1)

## Baseline Performance

The current FlashSloth model achieves:
- **GPT-4 score**: 41.9
- **Parameters**: 3.2B

## Evaluation Metrics

The task is evaluated based on GPT-4 score improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_mm_vet_flashsloth_hd_main
```

## Training Time

| Training Time |
|---------------|
| A800: 72 hours, A100: 48 hours, H100: 29 hours (1 GPU each) |
## Resources Provided
- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the FlashSloth model
- **Dataset**: MM-Vet dataset (if provided)
- **Solution Template**: Starting point for implementation
## File Structure
```
mm_vet_flashsloth_hd/
├── mm_vet_flashsloth_hd.py           # Main task definition
├── manifest.yaml            # Task configuration
├── build_steps.json         # Environment setup
├── requirements.txt         # Dependencies
├── README.md               # This file
└── assets/
    ├── score.py            # Scoring logic
    └── for_agent/
        ├── paper.pdf       # Research paper
        ├── repo.zip        # Source code
        ├── dataset.zip     # Dataset (if provided)
        └── solution.py     # Implementation template
```
## Implementation Guidelines
1. Study the baseline implementation
2. Analyze the dataset and task requirements
3. Identify improvement opportunities
4. Implement and validate changes
5. Ensure reproducible results
## Validation
Implement your solution with the expected interface:
```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'gpt-4_score': your_score
    }
```