# RLAE-DAN: Linear Recommender System on Gowalla

## Overview

This task focuses on improving the RLAE-DAN (Regularized Linear Autoencoder with Dynamic Attention Network) model on the Gowalla location-based social network dataset. The task explores why normalization is necessary for linear recommender systems.

## Task Description

**Objective**: Improve the RLAE-DAN model to achieve better recommendation performance on Gowalla.

**Paper**: "Why is Normalization Necessary for Linear Recommenders?"

**Dataset**: Gowalla - A location-based social networking website where users share their locations by checking-in.

## Baseline Performance

The current RLAE-DAN model achieves:
- **Recall@20**: 0.1922 (19.22%) - **Primary scoring metric**
- **nDCG@20**: 0.1605 (16.05%) - Reported for analysis

## Evaluation Metrics

### Primary Metric (Used for Scoring)
- **Recall@20**: Measures the proportion of relevant items in the top-20 recommendations
  - Baseline: 0.1922
  - Higher is better
  - **This is the only metric used for scoring**

### Secondary Metric (Reported)
- **nDCG@20**: Normalized Discounted Cumulative Gain at rank 20
  - Baseline: 0.1605
  - Higher is better
  - Reported for analysis but does not affect score

## Scoring Formula

Your solution is scored based on **percentage improvement over baseline Recall@20**:

```
Score = (new_recall@20 - baseline_recall@20) / baseline_recall@20 × 100
```

**Example**:
- If you achieve Recall@20 = 0.210:
  - Score = (0.210 - 0.1922) / 0.1922 × 100 = **9.26%**

Only improvements over baseline receive a positive score. If your Recall@20 ≤ 0.1922, the score is 0.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_gowalla_rlae_dan_main
```

## Training Time

| Training Time |
|---------------|
| 15 minutes |

## Background

### The Problem
Linear models for collaborative filtering are simple and scalable, but often perform poorly compared to complex neural networks. This paper investigates the role of normalization in making linear recommenders competitive.

### RLAE-DAN Approach
The model combines:
1. **Linear Autoencoder**: Simple linear transformations for efficiency
2. **Regularization**: Prevents overfitting in the linear space
3. **Normalization**: Critical for performance (paper's main focus)
4. **Dynamic Attention**: Adaptive weighting mechanisms

### Key Insights from the Paper
- Proper normalization can make linear models competitive with deep learning approaches
- Linear models offer better interpretability and faster inference
- The Gowalla dataset presents challenges due to sparse user-item interactions

## Potential Improvement Strategies

1. **Normalization Enhancements**
   - Experiment with different normalization schemes (L1, L2, batch norm)
   - Apply normalization at different layers
   - Adaptive normalization strategies

2. **Model Architecture**
   - Adjust the linear autoencoder dimensions
   - Modify the attention mechanism
   - Add skip connections or residual structures

3. **Training Strategies**
   - Better learning rate schedules
   - Improved regularization (dropout, weight decay)
   - Data augmentation for implicit feedback
   - Negative sampling strategies

4. **Loss Functions**
   - Pairwise ranking losses (BPR, WARP)
   - Pointwise vs. pairwise vs. listwise approaches
   - Multi-task learning with auxiliary losses

5. **Ensemble Methods**
   - Combine multiple linear models
   - Blend with other recommender approaches

## Resources Provided

- **Research Paper**: `/home/agent/paper.pdf` - Original paper with detailed methodology
- **Source Code**: `/home/agent/solution/` - Complete RLAE-DAN implementation
- **Dataset**: `/home/agent/dataset/` - Gowalla dataset
- **Solution Template**: `/home/agent/solution/solution.py` - Starting point for implementation

## File Structure

```
pwc_gowalla_rlae_dan/
├── pwc_gowalla_rlae_dan.py    # Main task definition
├── manifest.yaml              # Task configuration
├── build_steps.json           # Environment setup
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── assets/
    ├── score.py               # Scoring logic
    └── for_agent/
        ├── paper.pdf          # Research paper (6MB)
        ├── repo.zip           # Source code (30MB)
        ├── dataset.zip        # Dataset (192KB)
        └── solution.py        # Implementation template
```

## Implementation Guidelines

1. **Study the Baseline**
   - Read the paper to understand the RLAE-DAN approach
   - Examine the provided codebase
   - Run the baseline to verify performance

2. **Analyze the Dataset**
   - Understand the Gowalla data format
   - Analyze sparsity patterns
   - Identify potential data quality issues

3. **Implement Improvements**
   - Start with small, measurable changes
   - Focus on normalization strategies (paper's emphasis)
   - Validate improvements incrementally

4. **Evaluation**
   - Ensure your evaluation matches the baseline setup
   - Report both Recall@20 and nDCG@20
   - Verify reproducibility

## Expected Solution Interface

Your solution must implement the following interface:

```python
def evaluate() -> Dict[str, float]:
    """
    Train and evaluate your improved RLAE-DAN model.
    
    Returns:
        dict: Dictionary containing:
            - 'recall@20': float (e.g., 0.210) - Required
            - 'ndcg@20': float (e.g., 0.175) - Optional but recommended
    """
    # Your implementation here
    pass
```

## Validation

To test your solution locally:

```bash
cd /home/agent/solution
python solution.py
```

Expected output format:
```
Recall@20: 0.2100
nDCG@20: 0.1750
Recall@20 improvement: 9.26%
nDCG@20 improvement: 9.04%
```

## Notes

- The task has internet access (`full_internet` permission) for downloading additional libraries if needed
- Focus on the normalization aspects highlighted in the paper
- Linear models should remain efficient and interpretable
- Both metrics are reported, but only Recall@20 affects your score

## References

- Paper: "Why is Normalization Necessary for Linear Recommenders?" (arXiv:2504.05805v2)
- Dataset: Gowalla location-based social network
- Task Type: Collaborative Filtering / Recommendation System