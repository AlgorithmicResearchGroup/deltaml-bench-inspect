# MalNet-Tiny GatedGCN+ Task

## Overview

This task focuses on improving the GatedGCN+ model on the MalNet-Tiny dataset.

## Task Description

**Objective**: Improve the GatedGCN+ model to achieve better performance on the MalNet-Tiny dataset.

**Paper**: Unlocking the Potential of Classic GNNs for Graph-level Tasks: Simple Architectures Meet Excellence

## Baseline Performance

The current GatedGCN+ model achieves:
- **Accuracy**: 94.600±0.570

## Evaluation Metrics

The task is evaluated based on Accuracy improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_malnet_tiny_gatedgcn_main
```

## Training Time

| Training Time |
|---------------|
| 1 hour |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the GatedGCN+ model
- **Dataset**: MalNet-Tiny dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
malnet_tiny_gatedgcn/
├── malnet_tiny_gatedgcn.py           # Main task definition
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