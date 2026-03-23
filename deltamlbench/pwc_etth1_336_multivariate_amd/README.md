# ETTh1 Multivariate Time Series Forecasting with AMD

## Overview
This task challenges agents to improve the AMD (Adaptive Multi-Scale Decomposition) model for multivariate time series forecasting on the ETTh1 dataset with a prediction horizon of 336 timesteps.

## Task Description
**Paper:** [Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting](https://arxiv.org/abs/2406.03751v1)

**Dataset:** ETTh1 (Electricity Transformer Temperature - Hourly)
- 7 multivariate features
- Electricity transformer operational data
- Hourly frequency
- Prediction horizon: 336 hours (14 days)

## Baseline Performance
- **MAE:** 0.427
- **MSE:** 0.418

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_etth1_336_multivariate_amd_main
```

## Training Time

| Training Time |
|---------------|
| 13 minutes 8 seconds |

## Goal
Reduce both MAE and MSE by improving the multi-scale decomposition and forecasting approach.

## Evaluation Metrics
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values
- **MSE (Mean Squared Error):** Average squared difference (penalizes large errors more)

## Scoring Formula
```
Score = (baseline_mae - your_mae) / baseline_mae × 100
```

### Examples
- **MAE 0.400** → Score: **6.32%** improvement
- **MAE 0.410** → Score: **3.98%** improvement
- **MAE 0.427** → Score: **0%** (matches baseline)
- **MAE 0.450** → Score: **0%** (worse than baseline)

## AMD Method

### Key Innovation
AMD performs **adaptive multi-scale decomposition** to separate time series into:
1. **Trend component:** Long-term patterns
2. **Seasonal component:** Periodic patterns
3. **Residual component:** Irregular fluctuations

Each component is forecasted separately, then adaptively fused for final predictions.

### Why It Works
- Different components require different modeling strategies
- Multi-scale analysis captures patterns at various resolutions
- Adaptive weighting balances component contributions
- Better handles non-stationary time series

## ETTh1 Dataset Details

### Features (7 total)
- **OT:** Oil Temperature (target variable)
- **HUFL, HULL, MUFL, MULL, LUFL, LULL:** Load features

### Characteristics
- **Frequency:** Hourly measurements
- **Domain:** Power grid / electricity transformers
- **Challenge:** Long-horizon prediction (336 hours = 2 weeks)
- **Patterns:** Daily and weekly seasonality, trends

## Improvement Strategies

### 1. Enhanced Decomposition
- **More scales:** Increase resolution of decomposition
- **Better separation:** Improve trend/seasonal/residual extraction
- **Adaptive scales:** Learn optimal decomposition levels
- **Wavelet-based:** Use wavelet transform for decomposition

### 2. Component Modeling
- **Trend:** Linear models, polynomial regression, or Transformers
- **Seasonal:** Fourier analysis, seasonal ARIMA, or attention
- **Residual:** RNNs, GRUs, or residual networks

### 3. Fusion Mechanisms
- **Learned weights:** Train adaptive fusion weights
- **Attention-based:** Use attention to combine components
- **Hierarchical:** Multi-level fusion strategies

### 4. Architecture Improvements
- **Transformers:** Replace components with Transformer encoders
- **Attention mechanisms:** Temporal and feature attention
- **Skip connections:** Preserve information flow
- **Deeper networks:** More layers for complex patterns

### 5. Training Enhancements
- **Better loss:** Weighted MSE, quantile loss, or custom metrics
- **Regularization:** Component-wise regularization
- **Multi-task learning:** Predict multiple horizons jointly
- **Data augmentation:** Add noise, scaling, rotation

## Files Structure
```
/home/agent/
├── paper.pdf          # AMD research paper
├── solution/          # AMD codebase
│   └── solution.py    # Your implementation
├── dataset/           # ETTh1 dataset
│   └── ETTh1.csv      # Time series data
└── score.py           # Scoring script
```

## Getting Started
1. Review the AMD paper to understand multi-scale decomposition
2. Load ETTh1 dataset and explore its characteristics
3. Implement the AMD architecture with decomposition modules
4. Train on historical data with sliding windows
5. Evaluate on test set with horizon 336
6. Experiment with improvements

## Solution Interface
Your `solution.py` must implement:

```python
def evaluate() -> Dict[str, float]:
    """
    Returns:
        {
            'mae': float,  # Mean Absolute Error
            'mse': float   # Mean Squared Error
        }
    """
```

## Common Pitfalls
- **Insufficient decomposition:** Not capturing all scales
- **Poor component separation:** Mixing trend/seasonal/residual
- **Ignoring long-term dependencies:** 336-step horizon is challenging
- **Overfitting:** Too complex model for available data
- **Incorrect normalization:** Not scaling features properly

## Implementation Tips

### Data Preprocessing
```python
# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Create sliding windows
def create_windows(data, input_len=96, output_len=336):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)
```

### Multi-Scale Decomposition
```python
# Example decomposition (simplified)
def decompose(series, scales=[4, 8, 16]):
    trend = moving_average(series, window=scales[-1])
    seasonal = []
    for scale in scales:
        seasonal.append(extract_seasonal(series, period=scale))
    residual = series - trend - sum(seasonal)
    return trend, seasonal, residual
```

## Resources
- **Paper:** `/home/agent/paper.pdf`
- **Code:** `/home/agent/solution/`
- **Dataset:** `/home/agent/dataset/ETTh1.csv`

## References
- [ETT Dataset](https://github.com/zhouhaoyi/ETDataset)
- [Time Series Decomposition](https://otexts.com/fpp2/decomposition.html)
- [AMD Paper (arXiv)](https://arxiv.org/abs/2406.03751)

Good luck achieving MAE < 0.427!


