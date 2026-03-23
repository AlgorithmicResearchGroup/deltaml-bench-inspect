# Weather (192) xPatch Task

## Overview

This task focuses on improving the xPatch model on the Weather (192) dataset.

## Task Description

**Objective**: Improve the xPatch model to achieve better performance on the Weather (192) dataset.

**Paper**: xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition

## Baseline Performance

The current xPatch model achieves:
- **MAE**: 0.227

## Evaluation Metrics

The task is evaluated based on MAE improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_weather_192_xpatch_main
```

## Training Time

| Training Time |
|---------------|
| 30 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the xPatch model
- **Dataset**: Weather (192) dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
weather_192_xpatch/
├── weather_192_xpatch.py           # Main task definition
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
        'mae': your_score
    }
```