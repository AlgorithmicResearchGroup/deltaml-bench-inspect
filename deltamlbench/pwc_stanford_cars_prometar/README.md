# Stanford Cars ProMetaR Task

## Overview

This task focuses on improving the ProMetaR model on the Stanford Cars dataset.

## Task Description

**Objective**: Improve the ProMetaR model to achieve better performance on the Stanford Cars dataset.

**Paper**: Prompt Learning via Meta-Regularization

## Baseline Performance

The current ProMetaR model achieves:
- **Harmonic mean**: 76.72

## Evaluation Metrics

The task is evaluated based on Harmonic mean improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_stanford_cars_prometar_main
```

## Training Time

| Training Time |
|---------------|
| 2 hours 30 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the ProMetaR model
- **Dataset**: Stanford Cars dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
stanford_cars_prometar/
├── stanford_cars_prometar.py           # Main task definition
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
        'harmonic_mean': your_score
    }
```