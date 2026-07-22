# ogbl-ddi GCN (node embedding) Task

## Overview

This task focuses on improving the GCN (node embedding) model on the ogbl-ddi dataset.

## Task Description

**Objective**: Improve the GCN (node embedding) model to achieve better performance on the ogbl-ddi dataset.

**Paper**: Can GNNs Learn Link Heuristics? A Concise Review and Evaluation of Link Prediction Methods

## Baseline Performance

The current GCN (node embedding) model achieves:
- **Hit@20**: 0.9549 (95.49%)

## Evaluation Metrics

The task is evaluated based on Hit@20 (link prediction accuracy) improvement over baseline.
Higher Hit@20 is better.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_ogbl_ddi_gcn_node_embedding_main
```

## Training Time

| Training Time |
|---------------|
| 16 hours 40 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the GCN (node embedding) model
- **Dataset**: ogbl-ddi dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
ogbl_ddi_gcn_node_embedding/
├── ogbl_ddi_gcn_node_embedding.py           # Main task definition
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
        'hit20': your_score  # Hit@20 (0-1 scale, higher is better)
    }
```