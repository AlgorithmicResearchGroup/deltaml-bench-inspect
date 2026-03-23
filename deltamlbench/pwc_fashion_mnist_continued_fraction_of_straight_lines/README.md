# Continued Fraction of Straight Lines: Fashion-MNIST Classification

## Overview

This task focuses on improving the Continued fraction of straight lines model for image classification on the Fashion-MNIST dataset.

## Task Description

**Objective**: Improve the Continued fraction of straight lines model to achieve better classification performance on Fashion-MNIST.

**Paper**: "Real-valued continued fraction of straight lines" (https://arxiv.org/abs/2412.16191v1)

**Dataset**: Fashion-MNIST - 28x28 grayscale images of 10 clothing categories.

**Model**: Continued fraction of straight lines - A novel mathematical approach using real-valued continued fractions and straight line segments for neural network classification.

## Baseline Performance

The current Continued fraction of straight lines model achieves the following baseline performance on Fashion-MNIST:

- **Accuracy**: 84.12%
- **NMI (Normalized Mutual Information)**: 74.4
- **Trainable Parameters**: 7870

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline Accuracy (**higher is better**):

**Scoring**: 
```
Score = (new_accuracy - baseline_accuracy) / baseline_accuracy * 100
```
Example: Accuracy = 87.50% → Score = (87.50 - 84.12) / 84.12 * 100 = 4.02%

**Note**: For Accuracy, **higher values are better**. NMI and Trainable Parameters are reported for analysis but do not affect the score.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_fashion_mnist_continued_fraction_of_straight_lines_main
```

## Training Time

| Training Time |
|---------------|
| 40 seconds to 1 minute |

## Resources Provided

- **Research Paper**: Original paper on real-valued continued fractions for neural networks
- **Source Code**: Complete implementation of the continued fraction model (repository not publicly available)
- **Dataset**: Fashion-MNIST dataset (421KB)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 4-8GB RAM
- **Framework**: PyTorch or TensorFlow
- **Key Dependencies**: torch, numpy, PIL, scikit-learn

## File Structure

```
pwc_fashion_mnist_continued_fraction_of_straight_lines/
├── pwc_fashion_mnist_continued_fraction_of_straight_lines.py  # Main task definition
├── manifest.yaml                                               # Task configuration
├── build_steps.json                                            # Environment setup
├── requirements.txt                                            # Dependencies
├── README.md                                                   # This file
└── assets/
    ├── score.py                                                # Scoring logic
    └── for_agent/
        ├── paper.pdf                                           # Research paper
        ├── repo.zip                                            # Model source code
        ├── dataset.zip                                         # Fashion-MNIST (421KB)
        └── solution.py                                         # Implementation template
```

## Implementation Guidelines

1. **Understand the Fashion-MNIST Dataset**:
   - 60,000 training images, 10,000 test images
   - 28x28 grayscale images
   - 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
   - More challenging than MNIST due to higher visual complexity
   - Similar structure to MNIST but requires texture and shape understanding

2. **Study the Continued Fraction Model**:
   - Real-valued continued fraction representation
   - Straight line segments for classification boundaries
   - Parameter-efficient architecture (~7870 parameters)
   - Novel mathematical formulation for neural networks
   - Interpretable structure based on continued fractions

3. **Analyze Model Characteristics**:
   - Compact representation with few parameters
   - Mathematical elegance through continued fractions
   - Straight line approximations for decision boundaries
   - Trade-off between model capacity and efficiency
   - Potential for interpretability

4. **Identify Improvements**:
   - Enhanced continued fraction representations
   - Better straight line approximations
   - Improved initialization strategies
   - Data augmentation (rotation, translation, scaling)
   - Regularization techniques
   - Ensemble methods with different fraction depths
   - Balance accuracy improvements with parameter efficiency

5. **Image Classification Considerations**:
   - Proper data normalization (pixel values 0-1)
   - Data augmentation for robustness
   - Cross-validation strategies
   - Learning rate scheduling
   - Early stopping to prevent overfitting
   - Batch normalization or layer normalization

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, Union[float, int]]:
    # Your implementation here
    # 1. Load and preprocess Fashion-MNIST dataset
    # 2. Train improved Continued fraction model
    # 3. Evaluate on test set
    # 4. Calculate metrics
    
    return {
        'accuracy': your_accuracy,             # e.g., 87.50 (higher is better)
        'nmi': your_nmi,                       # e.g., 76.5 (higher is better)
        'trainable_parameters': your_params    # e.g., 8000
    }
```

## Key Concepts

- **Continued Fractions**: Mathematical representation as a sequence of fractions: a₀ + 1/(a₁ + 1/(a₂ + ...))
- **Straight Line Segments**: Using linear functions to approximate complex decision boundaries
- **Fashion-MNIST**: Dataset of clothing items, more visually complex than MNIST digits
- **NMI**: Normalized Mutual Information - measures clustering quality and class separation
- **Parameter Efficiency**: Achieving good performance with minimal model parameters
- **Accuracy**: Percentage of correctly classified test samples

## Fashion-MNIST Dataset Details

The Fashion-MNIST dataset contains:
- **Source**: Zalando's article images
- **Size**: 60,000 training + 10,000 test images
- **Format**: 28x28 grayscale images
- **Classes**: 10 clothing categories
  - 0: T-shirt/top
  - 1: Trouser
  - 2: Pullover
  - 3: Dress
  - 4: Coat
  - 5: Sandal
  - 6: Shirt
  - 7: Sneaker
  - 8: Bag
  - 9: Ankle boot
- **Challenge**: More complex than MNIST due to intra-class variation and inter-class similarity
- **Applications**: Fashion recognition, retail automation, image understanding

## Improvement Ideas

1. **Architecture Enhancements**:
   - Deeper continued fraction representations
   - Hierarchical straight line approximations
   - Multi-resolution feature extraction
   - Attention mechanisms for important regions

2. **Continued Fraction Optimization**:
   - Better convergence properties
   - Adaptive fraction depth
   - Regularization on fraction coefficients
   - Learned initialization strategies

3. **Training Strategies**:
   - Data augmentation (rotation, shift, zoom)
   - Learning rate scheduling (cosine annealing)
   - Mixup or cutmix augmentation
   - Label smoothing
   - Progressive training strategies

4. **Parameter Efficiency**:
   - Weight sharing across fraction levels
   - Low-rank approximations
   - Pruning techniques
   - Knowledge distillation

5. **Data Processing**:
   - Advanced normalization techniques
   - Contrast enhancement
   - Feature engineering from images
   - Multi-scale representations

## References

- Paper: [Real-valued continued fraction of straight lines](https://arxiv.org/abs/2412.16191v1)
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Continued Fractions: Mathematical representation for efficient neural network approximation