# ImageNet PRO-DSC Task

## Overview

This task focuses on improving the PRO-DSC model on the ImageNet dataset.

## Task Description

**Objective**: Improve the PRO-DSC model to achieve better performance on the ImageNet dataset.

**Paper**: Exploring a Principled Framework for Deep Subspace Clustering

## Baseline Performance

The current PRO-DSC model achieves:
- **Accuracy**: 65.0

## Evaluation Metrics

The task is evaluated based on Accuracy improvement over baseline.

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the PRO-DSC model
- **Dataset**: ImageNet dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
tiny_imagenet_pro_dsc/
├── tiny_imagenet_pro_dsc.py           # Main task definition
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

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_tiny_imagenet_pro_dsc_main
```

## Validation

Implement your solution with the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'accuracy': your_score
    }
```