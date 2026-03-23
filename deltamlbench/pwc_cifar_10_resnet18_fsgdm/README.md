# CIFAR-10 ResNet18 (FSGDM) Task

## Overview

This task focuses on improving the ResNet18 (FSGDM) model on the CIFAR-10 dataset.

## Task Description

**Objective**: Improve the ResNet18 (FSGDM) model to achieve better performance on the CIFAR-10 dataset.

**Paper**: On the Performance Analysis of Momentum Method: A Frequency Domain Perspective

## Baseline Performance

The current ResNet18 (FSGDM) model achieves:
- **Percentage correct**: 95.66

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline accuracy (95.66%):

**Scoring**: 
```
Score = (new_accuracy - 95.66) / 95.66 * 100
```
Example: Accuracy = 96.0% → Score = (96.0 - 95.66) / 95.66 * 100 = 0.36%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_cifar_10_resnet18_fsgdm_main
```

## Training Time

| Training Time |
|---------------|
| 12 hours 30 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the ResNet18 (FSGDM) model
- **Dataset**: CIFAR-10 dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
pwc_cifar_10_resnet18_fsgdm/
├── pwc_cifar_10_resnet18_fsgdm.py  # Main task definition
├── manifest.yaml                    # Task configuration
├── build_steps.json                 # Environment setup
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── assets/
    ├── score.py                     # Scoring logic
    └── for_agent/
        ├── paper.pdf                # Research paper
        ├── repo.zip                 # Source code
        ├── dataset.zip              # CIFAR-10 dataset
        └── solution.py              # Implementation template
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
        'percentage_correct': your_score
    }
```