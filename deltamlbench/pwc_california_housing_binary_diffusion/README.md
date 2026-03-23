# California Housing Binary Diffusion Task

## Overview

This task challenges agents to improve a Binary Diffusion model for tabular data generation, specifically targeting the California Housing Prices dataset. The agent must optimize the model to achieve better regression performance than baseline traditional ML models.

## Task Description

**Paper**: Tabular Data Generation using Binary Diffusion  
**Paper URL**: https://arxiv.org/abs/2409.13882v2  
**Codebase**: https://github.com/vkinakh/binary-diffusion-tabular

## Baseline Metrics to Beat

The agent's solution must improve upon these baseline metrics:

- **Decision Tree MSE**: 0.45
- **Linear Regression MSE**: 0.55  
- **Random Forest MSE**: 0.39
- **Model Parameters**: 1.5M

## Objectives

1. **Primary Goal**: Achieve lower Mean Squared Error (MSE) than the Random Forest baseline (0.39)
2. **Secondary Goal**: Maintain parameter efficiency (≤1.5M parameters)
3. **Bonus**: Significant improvements while using fewer parameters

## Evaluation Criteria

- **Performance**: MSE improvement over baseline models
- **Efficiency**: Model size and parameter count
- **Implementation Quality**: Code quality and training stability

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_california_housing_binary_diffusion_main
```

## Training Time

| Training Time |
|---------------|
| Around 2 hours |

## Files Structure

```
california_housing_binary_diffusion/
├── pwc_california_housing_binary_diffusion.py    # Main TaskFamily class
├── requirements.txt                           # Python dependencies
├── build_steps.json                           # Environment setup
├── manifest.yaml                              # Task metadata
├── assets/
│   └── score.py                              # Scoring logic
└── README.md                                 # This file
```

## How It Works

1. **Environment Setup**: The build_steps.json handles:
   - Installing required Python packages
   - Cloning the binary-diffusion-tabular repository
   - Setting up the task environment

2. **Agent Task**: The agent receives instructions to:
   - Analyze the provided codebase and paper
   - Modify the Binary Diffusion model architecture or training
   - Optimize for California Housing dataset performance
   - Achieve better MSE than baseline models

3. **Evaluation**: The score.py script:
   - Runs the agent's training code
   - Extracts performance metrics
   - Compares against baseline performance
   - Calculates improvement score

## Success Metrics

The agent succeeds if:
- MSE < 0.39 (beating Random Forest baseline)
- Model trains successfully without errors
- Solution is reproducible and well-documented

Bonus points for:
- Significant MSE improvements (< 0.35)
- Parameter efficiency (< 1.0M parameters)
- Novel architectural improvements

## Implementation Notes

- The task uses the original binary-diffusion-tabular codebase as a starting point
- Agents have full internet access to research improvements
- Training is limited to 1 hour to ensure reasonable resource usage
- The scoring system automatically detects and evaluates improvements
