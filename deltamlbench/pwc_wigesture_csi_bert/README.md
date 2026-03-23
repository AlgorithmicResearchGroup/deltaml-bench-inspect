# WiGesture CSI-BERT Task

## Overview

This task focuses on improving the CSI-BERT model on the WiGesture dataset.

## Task Description

**Objective**: Improve the CSI-BERT model to achieve better performance on the WiGesture dataset.

**Paper**: Finding the Missing Data: A BERT-inspired Approach Against Package Loss in Wireless Sensing

## Baseline Performance

The current CSI-BERT model achieves:
- **Accuracy (% )**: 93.94

## Evaluation Metrics

The task is evaluated based on Accuracy (% ) improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_wigesture_csi_bert_main
```

## Training Time

| Training Time |
|---------------|
| 2 hours estimated |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the CSI-BERT model
- **Dataset**: WiGesture dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
wigesture_csi_bert/
├── wigesture_csi_bert.py           # Main task definition
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
        'accuracy_(%_)': your_score
    }
```