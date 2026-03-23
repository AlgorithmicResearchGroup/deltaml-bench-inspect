# UCR Anomaly Archive KAN Task

## Overview

This task focuses on improving the KAN model on the UCR Anomaly Archive dataset.

## Task Description

**Objective**: Improve the KAN model to achieve better performance on the UCR Anomaly Archive dataset.

**Paper**: KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks

## Baseline Performance

The current KAN model achieves:
- **AUC ROC **: 0.7489

## Evaluation Metrics

The task is evaluated based on AUC ROC  improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_ucr_anomaly_archive_kan_main
```

## Training Time

| Training Time |
|---------------|
| 5 hours approximate |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the KAN model
- **Dataset**: UCR Anomaly Archive dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
ucr_anomaly_archive_kan/
├── ucr_anomaly_archive_kan.py           # Main task definition
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
        'auc_roc_': your_score
    }
```