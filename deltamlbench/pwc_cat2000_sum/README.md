# CAT2000 SUM Visual Attention Modeling Task

## Overview

This task focuses on improving the SUM (Saliency Unification through Mamba) model for visual attention modeling on the CAT2000 dataset. The goal is to enhance the model's performance beyond the established baseline metrics.

## Task Description

**Objective**: Improve the SUM model to achieve better visual attention modeling performance on the CAT2000 dataset.

**Dataset**: CAT2000 - A comprehensive dataset for visual attention modeling with diverse image categories.

**Model**: SUM (Saliency Unification through Mamba) - A state-of-the-art model that unifies different aspects of visual attention using Mamba architecture.

## Baseline Performance

The current SUM model achieves the following performance on CAT2000:

- **AUC (Area Under Curve)**: 0.888
- **NSS (Normalized Scanpath Saliency)**: 2.423

## Evaluation Metrics

The task reports two metrics:

1. **AUC (Primary Metric)**: Measures the model's ability to predict salient regions - **used for scoring**
2. **NSS**: Measures the alignment between predicted saliency and human eye fixations - reported for analysis

**Scoring**: The final score is calculated as the percentage improvement in AUC over baseline:
```
Score = (new_auc - baseline_auc) / baseline_auc * 100
```
Example: AUC = 0.90 → Score = (0.90 - 0.888) / 0.888 * 100 = 1.35%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_cat2000_sum_main
```

## Training Time

| Training Time |
|---------------|
| 5 minutes |

## Resources Provided

- **Research Paper**: Original SUM paper with detailed methodology
- **Source Code**: Complete implementation of the SUM model
- **Dataset**: CAT2000 dataset (581MB, downloaded from GCS during environment setup)
- **Solution Template**: Starting point for implementation

## Implementation Guidelines

1. **Study the baseline**: Understand the SUM architecture and current implementation
2. **Analyze the dataset**: Explore the CAT2000 dataset structure and characteristics
3. **Identify improvements**: Consider architectural modifications, training strategies, or data augmentation
4. **Implement changes**: Modify the model while maintaining the evaluation interface
5. **Validate results**: Ensure improvements generalize to the test set

## Technical Requirements

- **GPU**: 1x A100 (required for training)
- **Memory**: 64GB RAM
- **Framework**: PyTorch (primary), with supporting libraries for computer vision

## File Structure

```
pwc_cat2000_sum/
├── pwc_cat2000_sum.py      # Main task definition
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

**Note**: The CAT2000 dataset (581MB) is downloaded from GCS (`gs://benchmarking_tasks_repos/`) during environment setup, not included in the task files.

## Success Criteria

A successful solution should:
- Achieve improvements in both AUC and NSS metrics
- Maintain robust performance across different image categories
- Demonstrate clear understanding of visual attention mechanisms
- Provide reproducible results

## Common Improvement Strategies

- **Architecture modifications**: Enhance the Mamba backbone or attention mechanisms
- **Training improvements**: Better optimization, learning rate scheduling, or loss functions
- **Data augmentation**: Improve generalization through diverse training samples
- **Ensemble methods**: Combine multiple models or predictions
- **Multi-scale processing**: Handle different image resolutions or regions

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'auc': your_auc_score,
        'nss': your_nss_score
    }
```
