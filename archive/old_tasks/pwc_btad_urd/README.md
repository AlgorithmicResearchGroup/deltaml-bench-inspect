# BTAD URD Task

## Overview

This task focuses on improving the URD model on the BTAD dataset.

## Task Description

**Objective**: Improve the URD model to achieve better performance on the BTAD dataset.

**Paper**: Unlocking the Potential of Reverse Distillation for Anomaly Detection

## Baseline Performance

The current URD model achieves:
- **Detection AUROC**: 93.9

## Evaluation Metrics

The task is evaluated based on Detection AUROC improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_btad_urd_main
```

## Training Time

| Training Time |
|---------------|
| 8-9 hours |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the URD model
- **Dataset**: BTAD dataset (5GB, downloaded from GCS during environment setup)
- **Solution Template**: Starting point for implementation

## File Structure

```
pwc_btad_urd/
├── pwc_btad_urd.py          # Main task definition
├── manifest.yaml            # Task configuration
├── build_steps.json         # Environment setup (includes GCS dataset download)
├── requirements.txt         # Dependencies
├── README.md               # This file
└── assets/
    ├── score.py            # Scoring logic
    └── for_agent/
        ├── paper.pdf       # Research paper
        ├── repo.zip        # Source code
        └── solution.py     # Implementation template
```

**Note**: The BTAD dataset (5GB XZ compressed) is downloaded from GCS (`gs://benchmarking_tasks_repos/`) during environment setup, not included in the task files.

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
        'detection_auroc': your_score
    }
```