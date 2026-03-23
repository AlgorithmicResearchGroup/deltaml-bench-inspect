# CODE-CL: 5-Datasets Continual Learning Task

## Overview

This task focuses on improving the CODE-CL (Conceptor-Based Gradient Projection) model's performance on the 5-Datasets continual learning benchmark, which consists of CIFAR-10, MNIST, SVHN, Fashion-MNIST, and notMNIST.

## Task Description

**Objective**: Improve the CODE-CL model to achieve better performance on the 5-Datasets continual learning benchmark, focusing on maximizing average accuracy while minimizing catastrophic forgetting.

**Paper**: CODE-CL: Conceptor-Based Gradient Projection for Deep Continual Learning  
**Paper URL**: https://arxiv.org/abs/2411.15235  
**Repository**: https://github.com/mapolinario94/CODE-CL

## Baseline Performance

The current CODE-CL model achieves the following performance on the 5-Datasets benchmark:

- **Average Accuracy**: 93.32%
- **BWT (Backward Transfer)**: -0.25

### Individual Dataset Accuracies
- CIFAR-10: 89.1%
- MNIST: 98.5%
- SVHN: 87.3%
- Fashion-MNIST: 91.2%
- notMNIST: 75.9%

## Evaluation Metrics

The task is evaluated based on:

1. **Primary Metric (Score)**: Percentage improvement in Average Accuracy over baseline (93.32%)
2. **Reported Metrics** (not scored, but important for analysis):
   - BWT (Backward Transfer) (lower is better, baseline -0.25)

### Scoring Formula

```
score = (new_avg_acc - baseline_avg_acc) / baseline_avg_acc * 100
```

Returns 0 if new_avg_acc ≤ baseline_avg_acc.

**Example**: Average Accuracy of 95.0% → Score = (95.0 - 93.32) / 93.32 * 100 = 1.80%

## Resources Provided

- **Research Paper* *: Original paper with detailed methodology on conceptor-based gradient projection
- **Source Code**: Complete implementation of the CODE-CL model (cloned from GitHub)
- **notMNIST Dataset**: Pre-downloaded notMNIST dataset files (other datasets downloaded automatically)
- **Solution Template**: Starting point for implementation

## File Structure

```
5_datasets_code_cl/
├── 5_datasets_code_cl.py           # Main task definition
├── manifest.yaml                   # Task configuration
├── build_steps.json                # Environment setup
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── assets/
    ├── score.py                    # Scoring logic
    └── for_agent/
        ├── paper.pdf               # Research paper
        ├── repo.zip                # Source code backup
        ├── dataset.zip             # notMNIST dataset
        └── solution.py             # Implementation template
```

## Implementation Guidelines

1. **Study the baseline implementation**
   - Review the CODE-CL paper to understand conceptor-based gradient projection
   - Examine the provided codebase structure and training pipeline
   - Understand the continual learning evaluation protocol

2. **Analyze the task requirements**
   - The model must learn 5 different datasets sequentially
   - Catastrophic forgetting is the main challenge to overcome
   - Trade-offs exist between learning new tasks and retaining old knowledge

3. **Identify improvement opportunities**
   - Adjust conceptor aperture parameters for better gradient projection
   - Modify threshold values for convolutional and linear layers
   - Tune learning rates and optimization strategies
   - Experiment with different architectures (ResNet18, AlexNet, MLP)
   - Improve data augmentation techniques
   - Adjust basis batch sizes and free dimensions

4. **Implement and validate changes**
   - Test changes incrementally
   - Monitor both accuracy and forgetting metrics
   - Ensure reproducibility

5. **Ensure reproducible results**
   - Use fixed random seeds when appropriate
   - Document all hyperparameter changes

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_5_datasets_code_cl_main
```

## Training Time

| Training Time |
|---------------|
| 1-2 hours |

## Environment Setup

The environment includes:
- Python 3.8+ with PyTorch 2.2.1
- Avalanche continual learning library 0.5.0
- CODE-CL repository cloned to `/home/agent/CODE-CL/`
- notMNIST dataset at `/home/agent/Datasets/binary_mixture_5_Data/`
- Other datasets (CIFAR-10, MNIST, SVHN, Fashion-MNIST) downloaded automatically

## Example Training Command

Based on the original implementation:

```bash
python /home/agent/CODE-CL/main_5datasets.py \
    --lr 0.01 \
    --aperture 3 3 3 3 3 3 \
    --data-aug \
    --dropout \
    --basis-batch-size 100 \
    --avg-pool \
    --threshold-conv 0.9 \
    --threshold-linear 0.95 \
    --epochs 100 \
    --model ResNet18 \
    --n-experiences 5 \
    --batch-size 32 \
    --aperture-gain 0.5 \
    --patience 5 \
    --lr-threshold 1e-6 \
    --lr-decay 1.5 \
    --num-free-dim 10
```

## Validation

Implement your solution with the expected interface:

```python
def evaluate() -> Dict[str, float]:
    """
    Train and evaluate CODE-CL model on 5-Datasets benchmark
    
    Returns:
        dict: {
            'average_accuracy': float,  # e.g., 93.32
            'bwt': float               # e.g., -0.25
        }
    """
    # Your implementation here
    pass
```

## Key Concepts

### Continual Learning
The model must learn tasks sequentially without forgetting previously learned tasks. This is evaluated through:
- **Average Accuracy**: Performance across all tasks after sequential training
- **BWT (Backward Transfer)**: How much performance degrades on old tasks when learning new ones (lower is better, negative means forgetting)

### Conceptor-Based Gradient Projection
CODE-CL uses conceptors to:
1. Capture the subspace of important features for each task
2. Project gradients to avoid interfering with previous tasks
3. Balance plasticity (learning new tasks) vs stability (retaining old knowledge)

## Tips for Improvement

- **Aperture tuning**: Higher aperture allows more forgetting but better new task learning
- **Threshold optimization**: Affects which layers use conceptor projection
- **Architecture selection**: Different models have different capacity/forgetting trade-offs
- **Learning rate scheduling**: Important for balancing old and new task performance
- **Data augmentation**: Can improve generalization and reduce overfitting
- **BWT optimization**: Focus on minimizing backward transfer to reduce catastrophic forgetting