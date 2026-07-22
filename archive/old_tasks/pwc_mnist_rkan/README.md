# MNIST Classification with Rational Kolmogorov-Arnold Networks

## Overview
This task challenges agents to improve the rKAN (Rational Kolmogorov-Arnold Network) model for MNIST digit classification. rKAN uses trainable rational functions as activation functions, offering advantages over traditional polynomial-based activations.

## Task Description
**Paper:** [rKAN: Rational Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.14495v1)

**Repository:** https://github.com/alirezaafzalaghaei/rkan

**Innovation:** rKAN incorporates Kolmogorov-Arnold Networks with trainable adaptive rational-orthogonal Jacobi functions, providing:
- Non-polynomial behavior
- Activity for both positive and negative inputs
- Faster execution than standard KAN
- Better accuracy

## Baseline Performance
- **Accuracy:** 99.293%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_mnist_rkan_main
```

## Training Time

| Training Time |
|---------------|
| 35 minutes |

## Goal
Achieve higher classification accuracy on MNIST. Note: The baseline is already very high, so even 0.1-0.2% improvement is significant!

## Evaluation Metric
**Classification Accuracy:** Percentage of correctly classified test images.

## Scoring Formula
```
Score = (your_accuracy - baseline_accuracy) / baseline_accuracy × 100
```

### Examples
- **Accuracy 99.50%** → Score: **0.21%** improvement
- **Accuracy 99.40%** → Score: **0.11%** improvement  
- **Accuracy 99.293%** → Score: **0%** (matches baseline)
- **Accuracy 99.00%** → Score: **0%** (worse than baseline)

## rKAN Architecture Components

### 1. Jacobi rKAN Layers
```python
JacobiRKAN(degree)  # degree: typically 2-5
```
- Uses Jacobi polynomials as basis functions
- Trainable parameters for adaptivity
- Good for smooth, continuous activations
- Lower computational cost

### 2. Pade rKAN Layers
```python
PadeRKAN(m, n)  # Padé approximant [m/n]
```
- Uses rational functions (ratio of polynomials)
- m = numerator degree, n = denominator degree
- Better for modeling asymptotic behavior
- More expressive but slightly more computational cost

### Comparison
| Feature | Jacobi rKAN | Pade rKAN |
|---------|-------------|-----------|
| Basis | Jacobi polynomials | Rational functions |
| Parameters | Degree (2-5) | [m/n] (e.g., [2/6]) |
| Best for | Smooth activations | Complex behaviors |
| Speed | Faster | Slightly slower |
| Expressiveness | Good | Better |

## Approach

### Network Architecture Design
```python
# Example rKAN network for MNIST
Input (784) → Linear(128) → JacobiRKAN(3) → Linear(64) → PadeRKAN(2,6) → Output(10)
```

**Key decisions:**
1. **Layer types:** Mix Jacobi and Pade layers strategically
2. **Degrees:** Higher degrees = more expressive but risk overfitting
3. **Depth:** More layers can capture complex patterns
4. **Width:** Layer sizes affect capacity

## Improvement Strategies

### 1. Architecture Optimization
- **Mixed rKAN layers:** Combine Jacobi (early layers) + Pade (later layers)
- **Optimal degrees:** Try Jacobi(3-5) and Pade([2/6] to [4/8])
- **Network depth:** Experiment with 2-4 rKAN layers
- **Skip connections:** ResNet-style connections with rKAN

### 2. Training Enhancements
- **Better optimization:**
  - Adam optimizer with weight decay
  - Learning rate scheduling (cosine annealing)
  - Gradient clipping
  - Warmup phase
- **Regularization:**
  - Dropout between rKAN layers
  - Batch normalization
  - Label smoothing
  - Weight decay

### 3. Data Augmentation
- **Random rotations:** ±10-15 degrees
- **Random translations:** ±2-3 pixels
- **Elastic transformations:** Simulate handwriting variations
- **Cutout/RandomErasing:** Improve robustness
- **MixUp/CutMix:** Advanced augmentation

### 4. Advanced Techniques
- **Ensemble methods:** Average predictions from multiple rKAN models
- **Knowledge distillation:** Train smaller rKAN from larger teacher
- **Self-training:** Use pseudo-labels on augmented data
- **Test-time augmentation:** Average predictions on augmented test images

### 5. Hyperparameter Tuning
- Learning rate: 0.0001 - 0.01
- Batch size: 32 - 256
- rKAN degrees: Jacobi(2-5), Pade([1/5] - [5/9])
- Dropout rate: 0.1 - 0.5
- Training epochs: 20-100

## Files Structure
```
/home/agent/
├── paper.pdf              # rKAN research paper
├── solution/              # rKAN codebase
│   ├── rkan/              # rKAN package
│   │   ├── torch/         # PyTorch implementations
│   │   └── tensorflow/    # TensorFlow implementations
│   ├── examples/          # Example usage
│   ├── setup.py           # Package installer
│   └── solution.py        # Your implementation
└── dataset/               # MNIST dataset
```

## Getting Started
1. Review the rKAN paper to understand rational activations
2. Examine examples in `solution/examples/`
3. Try the basic Jacobi and Pade layers
4. Design your network architecture
5. Implement training and evaluation in `solution/solution.py`
6. Experiment with different configurations

## Solution Interface
Your `solution.py` must implement:

```python
def evaluate() -> Dict[str, float]:
    """
    Returns:
        {
            'accuracy': float  # Classification accuracy (0-100)
        }
    """
```

## Common Pitfalls
- **Overfitting:** High rKAN degrees without regularization
- **Underfitting:** Too simple architecture for MNIST complexity
- **Poor initialization:** Rational functions need careful init
- **Insufficient training:** rKAN may need more epochs to converge
- **No augmentation:** MNIST benefits significantly from augmentation
- **Wrong normalization:** MNIST mean=0.1307, std=0.3081

## Implementation Tips

### PyTorch Example
```python
import torch.nn as nn
from rkan.torch import JacobiRKAN, PadeRKAN

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.BatchNorm1d(128),
    JacobiRKAN(3),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    PadeRKAN(2, 6),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)
```

### TensorFlow Example
```python
from tensorflow.keras import layers
from rkan.tensorflow import JacobiRKAN, PadeRKAN

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128),
    layers.BatchNormalization(),
    JacobiRKAN(3),
    layers.Dropout(0.2),
    layers.Dense(64),
    layers.BatchNormalization(),
    PadeRKAN(2, 6),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

## Dataset Information
**MNIST:**
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images
- 10 classes (digits 0-9)
- Well-balanced classes

## Resources
- **Paper:** `/home/agent/paper.pdf`
- **Code:** `/home/agent/solution/`
- **Dataset:** `/home/agent/dataset/`
- **Examples:** `/home/agent/solution/examples/`

## References
- [rKAN Paper (arXiv)](https://arxiv.org/abs/2406.14495)
- [rKAN GitHub Repository](https://github.com/alirezaafzalaghaei/rkan)
- [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## Challenge
Since the baseline is already 99.293%, achieving even 99.4% (0.1% improvement) requires sophisticated techniques. Focus on:
- Ensemble of multiple rKAN variants
- Extensive data augmentation
- Careful hyperparameter tuning
- Advanced training strategies

Good luck achieving that extra 0.1-0.5% improvement!

