# PDBbind BAPULM Task

## Overview

This task focuses on improving the BAPULM model on the PDBbind dataset.

## Task Description

**Objective**: Improve the BAPULM model to achieve better performance on the PDBbind dataset.

**Paper**: BAPULM: Binding Affinity Prediction using Language Models

## Baseline Performance

The current BAPULM model achieves:
- **RMSE**: 0.898±0.0172

## Evaluation Metrics

The task is evaluated based on RMSE improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_pdbbind_bapulm_main
```

## Training Time

| Training Time |
|---------------|
| 16 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the BAPULM model
- **Dataset**: PDBbind dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
pdbbind_bapulm/
├── pdbbind_bapulm.py           # Main task definition
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
        'rmse': your_score
    }
```