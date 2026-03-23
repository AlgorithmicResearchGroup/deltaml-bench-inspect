# CycleNet: Time Series Forecasting on Electricity (192)

## Overview

This task focuses on improving the CycleNet model for time series forecasting on the Electricity dataset with a 192-step prediction horizon.

## Task Description

**Objective**: Improve the CycleNet model to achieve better forecasting performance on the Electricity dataset with 192-step ahead predictions.

**Paper**: "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns" (https://arxiv.org/abs/2409.18479v2)

**Repository**: https://github.com/ACAT-SCUT/CycleNet

**Dataset**: Electricity - Time series of electricity consumption from multiple clients with 192-step forecast horizon.

**Model**: CycleNet - A neural network architecture designed to enhance time series forecasting by explicitly modeling periodic patterns and cycles.

## Baseline Performance

The current CycleNet model achieves the following baseline performance on Electricity (192):

- **MSE (Mean Squared Error)**: 0.144
- **MAE (Mean Absolute Error)**: 0.237

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline MSE (**lower is better**):

**Scoring**: 
```
Score = (baseline_mse - new_mse) / baseline_mse * 100
```
Example: MSE = 0.130 → Score = (0.144 - 0.130) / 0.144 * 100 = 9.72%

**Note**: For MSE and MAE, **lower values are better**. The score rewards reductions in error.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_electricity_192_cyclenet_main
```

## Training Time

| Training Time |
|---------------|
| 13 hours |

## Resources Provided

- **Research Paper**: Original CycleNet paper with detailed methodology on cycle-aware forecasting
- **Source Code**: Complete implementation of the CycleNet model with periodic pattern modeling
- **Dataset**: Electricity (192) time series dataset (2.5KB - may need additional data download)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM
- **Framework**: PyTorch
- **Key Dependencies**: torch, numpy, pandas, scikit-learn

## File Structure

```
pwc_electricity_192_cyclenet/
├── pwc_electricity_192_cyclenet.py  # Main task definition
├── manifest.yaml                     # Task configuration
├── build_steps.json                  # Environment setup
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── assets/
    ├── score.py                      # Scoring logic
    └── for_agent/
        ├── paper.pdf                 # Research paper
        ├── repo.zip                  # CycleNet source code
        ├── dataset.zip               # Electricity dataset
        └── solution.py               # Implementation template
```

## Implementation Guidelines

1. **Understand the Electricity Dataset**:
   - Time series of electricity consumption from multiple clients
   - 192-step prediction horizon (long-term forecasting)
   - Strong daily and weekly periodic patterns
   - Seasonality and trend components
   - Multiple time series (multi-variate)

2. **Study the CycleNet Architecture**:
   - Cycle-aware neural networks
   - Periodic pattern decomposition
   - Temporal convolution and attention mechanisms
   - Multi-scale cycle modeling
   - Enhanced representation learning for periodic data

3. **Analyze Time Series Characteristics**:
   - Electricity consumption exhibits strong cycles (daily, weekly)
   - Peak usage during business hours
   - Seasonal variations
   - Holiday effects
   - Weather correlations

4. **Identify Improvements**:
   - Enhanced cycle detection algorithms
   - Better periodic pattern extraction
   - Improved temporal attention mechanisms
   - Multi-scale cycle modeling (hourly, daily, weekly)
   - Data augmentation for time series
   - Ensemble methods with different cycle decompositions
   - Better handling of trend and seasonality

5. **Time Series Forecasting Considerations**:
   - Proper train/validation/test splits (temporal ordering)
   - Normalization strategies (per-client or global)
   - Handling missing values
   - Outlier detection and treatment
   - Rolling window validation
   - Multi-step ahead forecasting strategies

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess Electricity (192) dataset
    # 2. Train improved CycleNet model
    # 3. Evaluate on test set with 192-step forecasting
    # 4. Calculate MSE and MAE
    
    return {
        'mse': your_mse_score,  # e.g., 0.130 (lower is better)
        'mae': your_mae_score   # e.g., 0.220 (lower is better)
    }
```

## Key Concepts

- **CycleNet**: Neural architecture that explicitly models periodic patterns in time series
- **Periodic Patterns**: Recurring patterns at regular intervals (daily, weekly, seasonal)
- **Time Series Forecasting**: Predicting future values based on historical observations
- **Prediction Horizon**: Number of future time steps to forecast (192 in this case)
- **MSE**: Mean Squared Error - average squared difference between predictions and actual values
- **MAE**: Mean Absolute Error - average absolute difference between predictions and actual values

## Electricity Dataset Details

The Electricity dataset contains:
- **Source**: Electricity consumption data from multiple clients
- **Frequency**: Hourly measurements
- **Horizon**: 192 steps (8 days for hourly data)
- **Patterns**: Strong daily cycles (24 hours) and weekly cycles (7 days)
- **Applications**: Load forecasting, grid management, energy planning

## Improvement Ideas

1. **Architecture Enhancements**:
   - Multi-scale cycle decomposition
   - Hierarchical attention mechanisms
   - Residual connections for better gradient flow
   - Graph neural networks for spatial correlations

2. **Cycle Modeling**:
   - Fourier analysis for period detection
   - Wavelet transforms for multi-resolution analysis
   - Adaptive cycle length learning
   - Multiple periodicity handling

3. **Training Strategies**:
   - Transfer learning from similar time series
   - Multi-task learning (short and long-term forecasting)
   - Curriculum learning (increasing horizon gradually)
   - Advanced loss functions (quantile loss, MAPE)

4. **Data Processing**:
   - Detrending and deseasonalization
   - Time series augmentation (jittering, scaling)
   - Feature engineering (lag features, rolling statistics)
   - External variables (weather, calendar features)

## References

- Paper: [CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns](https://arxiv.org/abs/2409.18479v2)
- Repository: [CycleNet on GitHub](https://github.com/ACAT-SCUT/CycleNet)
- Electricity Dataset: Standard time series forecasting benchmark
