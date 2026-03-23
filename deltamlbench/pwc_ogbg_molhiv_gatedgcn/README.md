# ogbg-molhiv GatedGCN+ Task

## Overview

This task focuses on improving the GatedGCN+ model on the ogbg-molhiv dataset.

## Task Description

**Objective**: Improve the GatedGCN+ model to achieve better performance on the ogbg-molhiv dataset.

**Paper**: Unlocking the Potential of Classic GNNs for Graph-level Tasks: Simple Architectures Meet Excellence

## Baseline Performance

The current GatedGCN+ model achieves:
- **ROC-AUC**: 0.7740 (77.40%)

## Evaluation Metrics

The task is evaluated based on ROC-AUC (Area Under ROC Curve) improvement over baseline.
This is a binary classification task for predicting molecular HIV activity.
Higher ROC-AUC is better.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_ogbg_molhiv_gatedgcn_main
```

## Training Time

| Training Time |
|---------------|
| Under 1 hour |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the GatedGCN+ model
- **Dataset**: ogbg-molhiv dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
ogbg_molhiv_gatedgcn/
├── ogbg_molhiv_gatedgcn.py           # Main task definition
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
        'roc_auc': your_score  # ROC-AUC (0-1 scale, higher is better)
    }
```