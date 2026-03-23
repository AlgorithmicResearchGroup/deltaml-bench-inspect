# Office-Home EUDA Task

## Overview

This task focuses on improving the EUDA model on the Office-Home dataset.

## Task Description

**Objective**: Improve the EUDA model to achieve better performance on the Office-Home dataset.

**Paper**: EUDA: An Efficient Unsupervised Domain Adaptation via Self-Supervised Vision Transformer

## Baseline Performance

The current EUDA model achieves:
- **Accuracy**: 92.0

## Evaluation Metrics

The task is evaluated based on Accuracy improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_office_31_euda_main
```

## Training Time

| Training Time |
|---------------|
| 1 hour 8 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the EUDA model
- **Dataset**: Office-Home dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
office_31_euda/
├── office_31_euda.py           # Main task definition
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
        'accuracy': your_score
    }
```