# SparseTSF: Long-term Time Series Forecasting on ETTh1 (720)

## Overview

This task focuses on improving the SparseTSF model for long-term multivariate time series forecasting on the ETTh1 dataset with a 720-step prediction horizon.

## Task Description

**Objective**: Improve the SparseTSF model to achieve better performance on the ETTh1 (720) Multivariate dataset with extreme long-term forecasting.

**Paper**: "SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters" (https://arxiv.org/abs/2405.00946v2)

**Dataset**: ETTh1 (Electricity Transformer Temperature - Hourly) - Multivariate time series with 7 features and 720-step forecast horizon.

**Model**: SparseTSF - A parameter-efficient neural network architecture (~1k parameters) designed for long-term time series forecasting through sparse temporal modeling.

## Baseline Performance

The current SparseTSF model achieves the following baseline performance on ETTh1 (720) Multivariate:

- **MSE (Mean Squared Error)**: 0.426

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline MSE (**lower is better**):

**Scoring**: 
```
Score = (baseline_mse - new_mse) / baseline_mse * 100
```
Example: MSE = 0.380 → Score = (0.426 - 0.380) / 0.426 * 100 = 10.80%

**Note**: For MSE, **lower values are better**. The score rewards reductions in error.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_etth1_720_multivariate_sparsetsf_main
```

## Training Time

| Training Time |
|---------------|
| 2 minutes |

## Resources Provided

- **Research Paper**: Original SparseTSF paper with detailed methodology on parameter-efficient long-term forecasting
- **Source Code**: Complete implementation of the SparseTSF model (repository not publicly available)
- **Dataset**: ETTh1 (720) Multivariate dataset (314KB)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM
- **Framework**: PyTorch
- **Key Dependencies**: torch, numpy, pandas, scikit-learn

## File Structure

```
pwc_etth1_720_multivariate_sparsetsf/
├── pwc_etth1_720_multivariate_sparsetsf.py  # Main task definition
├── manifest.yaml                             # Task configuration
├── build_steps.json                          # Environment setup
├── requirements.txt                          # Dependencies
├── README.md                                 # This file
└── assets/
    ├── score.py                              # Scoring logic
    └── for_agent/
        ├── paper.pdf                         # Research paper
        ├── repo.zip                          # SparseTSF source code
        ├── dataset.zip                       # ETTh1 dataset (314KB)
        └── solution.py                       # Implementation template
```

## Implementation Guidelines

1. **Understand the ETTh1 Dataset**:
   - Electricity Transformer Temperature (Hourly)
   - 7 multivariate features: OT (Oil Temperature), HUFL, HULL, MUFL, MULL, LUFL, LULL
   - 720-step prediction horizon (30 days for hourly data - EXTREME long-term)
   - Strong inter-series dependencies and correlations
   - Temporal patterns across multiple time series

2. **Study the SparseTSF Architecture**:
   - Parameter-efficient design (~1k parameters)
   - Sparse temporal feature extraction
   - Long-term dependency modeling
   - Efficient architecture for extended horizons
   - Reduced computational complexity

3. **Analyze Long-term Forecasting Challenges**:
   - 720 steps ahead is extremely challenging
   - Information decay over long horizons
   - Need to balance capacity and efficiency
   - Gradient flow through very long sequences
   - Multi-scale temporal patterns

4. **Identify Improvements**:
   - Enhanced sparse temporal modeling
   - Better parameter utilization without increasing count
   - Improved long-term dependency capture
   - Multi-scale feature extraction
   - Data augmentation for long-term forecasting
   - Ensemble methods with different sparsity patterns
   - Better handling of information decay

5. **Long-term Multivariate Forecasting Considerations**:
   - Proper train/validation/test splits (temporal ordering)
   - Normalization strategies (critical for long horizons)
   - Handling vanishing/exploding gradients
   - Multi-step vs direct forecasting strategies
   - Computational efficiency for 720-step predictions
   - Cross-validation for multivariate data

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess ETTh1 (720) Multivariate dataset
    # 2. Train improved SparseTSF model (keep ~1k parameters)
    # 3. Evaluate on test set with 720-step forecasting
    # 4. Calculate MSE
    
    return {
        'mse': your_mse_score  # e.g., 0.380 (lower is better)
    }
```

## Key Concepts

- **SparseTSF**: Parameter-efficient architecture for long-term time series forecasting with ~1k parameters
- **Sparse Temporal Modeling**: Selective feature extraction focusing on most informative temporal patterns
- **Long-term Forecasting**: Predicting far into the future (720 steps = 30 days)
- **Parameter Efficiency**: Achieving good performance with minimal parameters
- **ETTh1**: Electricity Transformer Temperature dataset with hourly measurements
- **MSE**: Mean Squared Error - average squared difference between predictions and actual values

## ETTh1 Dataset Details

The ETTh1 dataset contains:
- **Source**: Electricity transformer temperature measurements
- **Frequency**: Hourly measurements
- **Features**: 7 multivariate features (OT, HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Horizon**: 720 steps (30 days for hourly data - EXTREME long-term)
- **Patterns**: Daily and weekly cycles, cross-series correlations
- **Challenge**: Predicting 30 days ahead is extremely difficult
- **Applications**: Energy management, long-term planning, capacity forecasting

## Improvement Ideas

1. **Architecture Enhancements**:
   - Advanced sparse temporal attention
   - Hierarchical feature extraction
   - Multi-resolution temporal modeling
   - Efficient residual connections

2. **Parameter Efficiency**:
   - Weight sharing strategies
   - Low-rank decompositions
   - Sparse connection patterns
   - Knowledge distillation from larger models

3. **Long-term Modeling**:
   - Progressive forecasting (short to long)
   - Multi-scale temporal features
   - Explicit trend and seasonality modeling
   - Memory mechanisms for long dependencies

4. **Training Strategies**:
   - Curriculum learning (increasing horizon)
   - Multi-task learning (multiple horizons)
   - Advanced loss functions (weighted MSE, quantile loss)
   - Gradient clipping and normalization

5. **Data Processing**:
   - Feature engineering for long-term patterns
   - Time series augmentation
   - Cross-series normalization
   - Handling missing data

## References

- Paper: [SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters](https://arxiv.org/abs/2405.00946v2)
- ETTh1 Dataset: Standard time series forecasting benchmark for long-term multivariate prediction
- Parameter-efficient Deep Learning: Achieving performance with minimal parameters