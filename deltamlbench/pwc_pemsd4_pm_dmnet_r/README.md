# PeMSD4 PM-DMNet(R) Task

## Overview

This task focuses on improving the PM-DMNet(R) model on the PeMSD4 dataset.

## Task Description

**Objective**: Improve the PM-DMNet(R) model to achieve better performance on the PeMSD4 dataset.

**Paper**: Pattern-Matching Dynamic Memory Network for Dual-Mode Traffic Prediction

## Baseline Performance

The current PM-DMNet(R) model achieves:
- **12 steps MAE**: 18.37

## Evaluation Metrics

The task is evaluated based on 12 steps MAE improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_pemsd4_pm_dmnet_r_main
```

## Training Time

| Training Time |
|---------------|
| 5 hours |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the PM-DMNet(R) model
- **Dataset**: PeMSD4 dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
pemsd4_pm_dmnet_r/
├── pemsd4_pm_dmnet_r.py           # Main task definition
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
        '12_steps_mae': your_score
    }
```