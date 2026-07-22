# TXL-PBC: a freely accessible labeled peripheral blood cell dataset yolov5n Task

## Overview

This task focuses on improving the yolov5n model on the TXL-PBC: a freely accessible labeled peripheral blood cell dataset dataset.

## Task Description

**Objective**: Improve the yolov5n model to achieve better performance on the TXL-PBC: a freely accessible labeled peripheral blood cell dataset dataset.

**Paper**: TXL-PBC: a freely accessible labeled peripheral blood cell dataset

## Baseline Performance

The current yolov5n model achieves:
- **mAP50**: 0.958

## Evaluation Metrics

The task is evaluated based on mAP50 improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n_main
```

## Training Time

| Training Time |
|---------------|
| 29 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the yolov5n model
- **Dataset**: TXL-PBC: a freely accessible labeled peripheral blood cell dataset dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n/
├── txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n.py           # Main task definition
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
        'map50': your_score
    }
```