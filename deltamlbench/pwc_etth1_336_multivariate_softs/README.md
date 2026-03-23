# SOFTS: Multivariate Time Series Forecasting on ETTh1 (336)

## Overview

This task focuses on improving the SOFTS model for multivariate time series forecasting on the ETTh1 dataset with a 336-step prediction horizon.

## Task Description

**Objective**: Improve the SOFTS model to achieve better forecasting performance on the ETTh1 (336) Multivariate dataset.

**Paper**: "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion" (https://arxiv.org/abs/2404.14197v3)

**Dataset**: ETTh1 (Electricity Transformer Temperature - Hourly) - Multivariate time series with 7 features and 336-step forecast horizon.

**Model**: SOFTS - A neural network architecture designed for efficient multivariate time series forecasting through series-core fusion.

## Baseline Performance

The current SOFTS model achieves the following baseline performance on ETTh1 (336) Multivariate:

- **MAE (Mean Absolute Error)**: 0.452
- **MSE (Mean Squared Error)**: 0.480

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline MAE (**lower is better**):

**Scoring**: 
```
Score = (baseline_mae - new_mae) / baseline_mae * 100
```
Example: MAE = 0.410 → Score = (0.452 - 0.410) / 0.452 * 100 = 9.29%

**Note**: For MAE and MSE, **lower values are better**. The score rewards reductions in error.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_etth1_336_multivariate_softs_main
```

## Training Time

| Training Time |
|---------------|
| 5 minutes |

## Resources Provided

- **Research Paper**: Original SOFTS paper with detailed methodology on series-core fusion
- **Source Code**: Complete implementation of the SOFTS model (repository not publicly available)
- **Dataset**: ETTh1 (336) Multivariate dataset (314KB)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM
- **Framework**: PyTorch
- **Key Dependencies**: torch, numpy, pandas, scikit-learn

## File Structure

```
pwc_etth1_336_multivariate_softs/
├── pwc_etth1_336_multivariate_softs.py  # Main task definition
├── manifest.yaml                         # Task configuration
├── build_steps.json                      # Environment setup
├── requirements.txt                      # Dependencies
├── README.md                             # This file
└── assets/
    ├── score.py                          # Scoring logic
    └── for_agent/
        ├── paper.pdf                     # Research paper
        ├── repo.zip                      # SOFTS source code
        ├── dataset.zip                   # ETTh1 dataset (314KB)
        └── solution.py                   # Implementation template
```

## Implementation Guidelines

1. **Understand the ETTh1 Dataset**:
   - Electricity Transformer Temperature (Hourly)
   - 7 multivariate features: OT (Oil Temperature), HUFL, HULL, MUFL, MULL, LUFL, LULL
   - 336-step prediction horizon (2 weeks for hourly data)
   - Strong inter-series dependencies and correlations
   - Temporal patterns across multiple time series

2. **Study the SOFTS Architecture**:
   - Series-Core Fusion mechanism
   - Efficient multivariate time series modeling
   - Reduced computational complexity
   - Cross-series dependency capture
   - Attention mechanisms for multivariate data

3. **Analyze Multivariate Time Series Characteristics**:
   - Multiple correlated features (7 series)
   - Shared temporal patterns
   - Cross-series dependencies
   - Long-term forecasting challenges
   - Feature interactions

4. **Identify Improvements**:
   - Enhanced series-core fusion mechanisms
   - Better cross-series attention
   - Improved temporal modeling across features
   - Multi-scale feature extraction
   - Data augmentation for multivariate time series
   - Ensemble methods with different fusion strategies
   - Better handling of inter-series correlations

5. **Multivariate Forecasting Considerations**:
   - Proper train/validation/test splits (temporal ordering)
   - Normalization strategies (per-series or global)
   - Handling missing values across series
   - Feature selection and engineering
   - Cross-validation for multivariate data
   - Multi-step ahead forecasting strategies

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess ETTh1 (336) Multivariate dataset
    # 2. Train improved SOFTS model with series-core fusion
    # 3. Evaluate on test set with 336-step forecasting
    # 4. Calculate MAE and MSE
    
    return {
        'mae': your_mae_score,  # e.g., 0.410 (lower is better)
        'mse': your_mse_score   # e.g., 0.450 (lower is better)
    }
```

## Key Concepts

- **SOFTS**: Series-Core Fusion based architecture for efficient multivariate time series forecasting
- **Series-Core Fusion**: Mechanism to efficiently model interactions between multiple time series
- **Multivariate Time Series**: Multiple related time series observed simultaneously
- **ETTh1**: Electricity Transformer Temperature dataset with hourly measurements
- **Prediction Horizon**: Number of future time steps to forecast (336 in this case)
- **MAE**: Mean Absolute Error - average absolute difference between predictions and actual values
- **MSE**: Mean Squared Error - average squared difference between predictions and actual values

## ETTh1 Dataset Details

The ETTh1 dataset contains:
- **Source**: Electricity transformer temperature measurements
- **Frequency**: Hourly measurements
- **Features**: 7 multivariate features (OT, HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Horizon**: 336 steps (2 weeks for hourly data)
- **Patterns**: Daily and weekly cycles, cross-series correlations
- **Applications**: Energy management, transformer monitoring, predictive maintenance

## Improvement Ideas

1. **Architecture Enhancements**:
   - Advanced series-core fusion mechanisms
   - Hierarchical attention for multivariate data
   - Graph neural networks for series dependencies
   - Residual connections for better gradient flow

2. **Multivariate Modeling**:
   - Cross-series attention mechanisms
   - Channel-wise feature extraction
   - Dynamic feature weighting
   - Series decomposition and fusion

3. **Training Strategies**:
   - Transfer learning from related datasets
   - Multi-task learning (multiple horizons)
   - Curriculum learning (increasing complexity)
   - Advanced loss functions (quantile loss, MAPE)

4. **Data Processing**:
   - Feature engineering across series
   - Time series augmentation (jittering, warping)
   - Cross-series normalization strategies
   - Handling missing data in multivariate context

## References

- Paper: [SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion](https://arxiv.org/abs/2404.14197v3)
- ETTh1 Dataset: Standard time series forecasting benchmark for multivariate prediction
- Series-Core Fusion: Novel approach for efficient multivariate time series modeling