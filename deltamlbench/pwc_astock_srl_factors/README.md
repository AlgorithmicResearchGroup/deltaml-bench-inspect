# Astock SRL&Factors Financial Forecasting Task

## Overview

This task focuses on improving the SRL&Factors model for stock earnings forecasting on the Astock dataset. The task involves analyzing news articles using Semantic Role Labeling (SRL) to extract factors that predict stock earnings performance.

## Task Description

**Objective**: Improve the SRL&Factors model to achieve better performance on the Astock stock earnings forecasting dataset.

**Paper**: FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model  
**Paper URL**: https://arxiv.org/abs/2403.02647v1

## Baseline Performance

The current SRL&Factors model achieves:
- **Accuracy**: 69.48% (PRIMARY METRIC for scoring)
- **F1-score**: 69.28%
- **Precision**: 69.54%
- **Recall**: 69.41%

## Evaluation Metrics

1. **Primary Metric (Score)**: Percentage improvement in Accuracy over baseline (69.48%)
2. **Reported Metrics** (not scored, but important for analysis):
   - F1-score, Precision, Recall

### Scoring Formula

```
score = (new_accuracy - baseline_accuracy) / baseline_accuracy * 100
```

Returns 0 if new_accuracy ≤ baseline_accuracy.

**Example**: Accuracy of 75.0% → Score = (75.0 - 69.48) / 69.48 * 100 = 7.95%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_astock_srl_factors_main
```

## Training Time

| Training Time |
|---------------|
| Under 5 minutes |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the SRL&Factors model
- **Dataset**: Astock dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
astock_srl_factors/
├── astock_srl_factors.py           # Main task definition
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
        'accuracy': 69.48,      # Your improved accuracy (percentage)
        'f1_score': 69.28,      # Your F1-score (percentage)
        'precision': 69.54,     # Your precision (percentage)
        'recall': 69.41         # Your recall (percentage)
    }
```

## Key Concepts

### Financial AI Task
This is a specialized financial AI task that requires:
- **News Sentiment Analysis**: Understanding how news affects stock performance
- **Factor Extraction**: Using SRL to identify key factors from news articles
- **Earnings Prediction**: Binary classification - will earnings increase or decrease?

### SRL&Factors Model
The model combines:
1. **Semantic Role Labeling (SRL)**: Extracts structured information from news text
2. **Factor Analysis**: Identifies which news factors predict earnings changes
3. **Classification**: Predicts earnings direction based on extracted factors