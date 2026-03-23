# York Urban Dataset DT-LSD Task

## Overview

This task focuses on improving the DT-LSD model on the York Urban Dataset dataset.

## Task Description

**Objective**: Improve the DT-LSD model to achieve better performance on the York Urban Dataset dataset.

**Paper**: DT-LSD: Deformable Transformer-based Line Segment Detection

## Baseline Performance

The current DT-LSD model achieves:
- **sAP10**: 33.2

## Evaluation Metrics

The task is evaluated based on sAP10 improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_york_urban_dataset_dt_lsd_main
```

## Training Time

| Training Time |
|---------------|
| 8 hours |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the DT-LSD model
- **Dataset**: York Urban Dataset dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
york_urban_dataset_dt_lsd/
├── york_urban_dataset_dt_lsd.py           # Main task definition
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
        'sap10': your_score
    }
```