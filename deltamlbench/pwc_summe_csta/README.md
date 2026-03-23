# SumMe CSTA Task

## Overview

This task focuses on improving the CSTA model on the SumMe dataset.

## Task Description

**Objective**: Improve the CSTA model to achieve better performance on the SumMe dataset.

**Paper**: CSTA: CNN-based Spatiotemporal Attention for Video Summarization

## Baseline Performance

The current CSTA model achieves:
- **Kendall's Tau**: 0.246

## Evaluation Metrics

The task is evaluated based on Kendall's Tau improvement over baseline.

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the CSTA model
- **Dataset**: SumMe dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
summe_csta/
├── summe_csta.py           # Main task definition
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

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_summe_csta_main
```

## Validation

Implement your solution with the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'kendall_tau': your_score  # Kendall's Tau (correlation metric, higher is better)
    }
```