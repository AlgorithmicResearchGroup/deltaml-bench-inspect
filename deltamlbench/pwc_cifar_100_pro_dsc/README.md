# PRO-DSC: Deep Subspace Clustering on CIFAR-100

## Overview

This task focuses on improving the PRO-DSC (Principled fRamewOrk for Deep Subspace Clustering) model on the CIFAR-100 dataset.

## Task Description

**Objective**: Improve the PRO-DSC model to achieve better clustering performance on the CIFAR-100 dataset (100 classes, unsupervised clustering).

**Paper**: "Exploring a Principled Framework for Deep Subspace Clustering" (https://arxiv.org/abs/2503.17288v1)

**Repository**: https://github.com/mengxianghan123/PRO-DSC

**Model**: PRO-DSC - A principled framework for deep subspace clustering that learns structured representations and self-expressive coefficients in a unified manner. The model uses CLIP features for clustering.

## Baseline Performance

The current PRO-DSC model achieves the following baseline performance on CIFAR-100 clustering:

- **Accuracy**: 0.773 (77.3%)
- **NMI (Normalized Mutual Information)**: 0.824 (82.4%)

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline accuracy:

**Scoring**: 
```
Score = (new_accuracy - 0.773) / 0.773 * 100
```
Example: Accuracy = 0.787 (78.7%) → Score = (0.787 - 0.773) / 0.773 * 100 = 1.81%

**IMPORTANT**: All accuracy and NMI values must be returned in decimal format (0-1 range), not as percentages!

**Note**: NMI is reported for analysis but does not affect the score.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_cifar_100_pro_dsc_main
```

## Training Time

| Training Time |
|---------------|
| 7 hours 35 minutes |

## Resources Provided

- **Research Paper**: Original PRO-DSC paper with detailed methodology
- **Source Code**: Complete implementation of the PRO-DSC model with configs for CIFAR-100
- **Dataset**: CIFAR-100 CLIP features (may need to be downloaded using PRO-DSC's data preparation scripts)
- **Solution Template**: Starting point for implementation

## Technical Requirements

- **GPU**: 1x (recommended for faster training)
- **Memory**: 16GB RAM (adjust as needed)
- **Framework**: PyTorch
- **Key Dependencies**: torch, torchvision, numpy, scipy

## File Structure

```
pwc_cifar_100_pro_dsc/
├── pwc_cifar_100_pro_dsc.py  # Main task definition
├── manifest.yaml              # Task configuration
├── build_steps.json           # Environment setup
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── assets/
    ├── score.py               # Scoring logic
    └── for_agent/
        ├── paper.pdf          # Research paper
        ├── repo.zip           # PRO-DSC source code
        ├── dataset.zip        # CIFAR-100 CLIP features (may be empty)
        └── solution.py        # Implementation template
```

**Note**: The `dataset.zip` may be empty. If so, follow the PRO-DSC repository instructions to download the CIFAR-100 CLIP features.

## Implementation Guidelines

1. **Study the baseline**: Understand the PRO-DSC architecture, especially:
   - Self-expressive model with regularization
   - Representation learning
   - Subspace clustering method
   - Key hyperparameters: gamma, beta, pieta, hidden_dim, z_dim

2. **Analyze the task**: Review `configs/cifar100.yaml` for CIFAR-100 specific configuration

3. **Identify improvements**: Consider:
   - Hyperparameter tuning (gamma, beta, pieta)
   - Architecture modifications (hidden_dim, z_dim)
   - Training strategies (learning rate, batch size, warmup)
   - Regularization techniques
   - Ensemble methods

4. **Implement changes**: Modify the model while maintaining the evaluation interface

5. **Validate results**: Ensure improvements generalize to the test set

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'accuracy': your_accuracy_score,  # e.g., 0.787 for 78.7% (decimal format!)
        'nmi': your_nmi_score              # e.g., 0.830 for 83.0% (decimal format!)
    }
```

## Key Concepts

- **Subspace Clustering**: Assumes high-dimensional data can be approximated by a union of subspaces
- **Self-Expressive Model**: Learns coefficients that express each data point as a linear combination of others
- **CLIP Features**: Uses pre-trained CLIP embeddings as input features
- **Unsupervised Learning**: No labels are used during training, only for evaluation

## References

- Paper: [Exploring a Principled Framework for Deep Subspace Clustering](https://arxiv.org/abs/2503.17288v1)
- Repository: [PRO-DSC on GitHub](https://github.com/mengxianghan123/PRO-DSC)
- ICLR 2025 accepted paper
