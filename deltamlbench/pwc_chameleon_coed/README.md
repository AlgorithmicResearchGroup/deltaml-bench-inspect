# Chameleon CoED Graph Neural Network Task

## Overview

This task focuses on improving the CoED (Continuous Edge Directions) model for node classification on the Chameleon dataset. The goal is to enhance the model's performance beyond the established baseline.

## Task Description

**Objective**: Improve the CoED model to achieve better node classification accuracy on the Chameleon dataset.

**Dataset**: Chameleon - A heterophilic graph dataset where connected nodes tend to have different labels, making it challenging for traditional GNNs.

**Model**: CoED (Continuous Edge Directions) - A graph neural network that learns continuous edge directions to improve message passing.

## Baseline Performance

The current CoED model achieves the following performance on Chameleon:

- **Accuracy**: 79.69 ± 1.35%

## Evaluation Metrics

The task is evaluated based on classification accuracy:

**Scoring**: The final score is calculated as the percentage improvement over baseline accuracy:
```
Score = (new_accuracy - baseline_accuracy) / baseline_accuracy * 100
```

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_chameleon_coed_main
```

## Training Time

| Training Time |
|---------------|
| 2 minutes |

## Resources Provided

- **Research Paper**: Original CoED paper with detailed methodology
- **Source Code**: Complete implementation of the CoED model
- **Dataset**: Chameleon dataset (32MB)
- **Solution Template**: Starting point for implementation

## Implementation Guidelines

1. **Study the baseline**: Understand the CoED architecture and continuous edge direction learning
2. **Analyze the dataset**: Explore the Chameleon dataset structure and heterophilic properties
3. **Identify improvements**: Consider architectural modifications, training strategies, or edge learning enhancements
4. **Implement changes**: Modify the model while maintaining the evaluation interface
5. **Validate results**: Ensure improvements generalize to the test set

## Technical Requirements

- **GPU**: 1x A100 (recommended for training)
- **Memory**: 32GB RAM
- **Framework**: PyTorch with PyTorch Geometric

## File Structure

```
pwc_chameleon_coed/
├── pwc_chameleon_coed.py   # Main task definition
├── manifest.yaml            # Task configuration
├── build_steps.json         # Environment setup
├── requirements.txt         # Dependencies
├── README.md               # This file
└── assets/
    ├── score.py            # Scoring logic
    └── for_agent/
        ├── paper.pdf       # Research paper
        ├── repo.zip        # Source code
        ├── dataset.zip     # Chameleon dataset
        └── solution.py     # Implementation template
```

## Success Criteria

A successful solution should:
- Achieve improved classification accuracy on the Chameleon dataset
- Demonstrate understanding of heterophilic graph properties
- Show effective continuous edge direction learning
- Provide reproducible results

## Common Improvement Strategies

- **Enhanced edge direction learning**: Improve the continuous edge direction computation
- **Architecture modifications**: Modify the GNN layers or message passing mechanisms
- **Training improvements**: Better optimization, learning rate scheduling, or loss functions
- **Regularization**: Add dropout, batch normalization, or other regularization techniques
- **Ensemble methods**: Combine multiple models or use different aggregation strategies
- **Feature engineering**: Enhance node or edge features

## Graph Neural Network Considerations

The Chameleon dataset is heterophilic, meaning:
- Connected nodes tend to have different labels
- Traditional GNN assumptions may not hold
- Edge direction learning becomes crucial
- Local neighborhood aggregation needs careful design

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'accuracy': your_accuracy_score  # As percentage (e.g., 82.5 for 82.5%)
    }
```
