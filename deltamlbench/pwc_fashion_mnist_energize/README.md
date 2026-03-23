# ENERGIZE: Physically Plausible Neuroevolution on Fashion-MNIST

## Overview

This task focuses on improving the ENERGIZE model for image classification on the Fashion-MNIST dataset with physical plausibility and energy efficiency constraints.

## Task Description

**Objective**: Improve the ENERGIZE model to achieve better classification performance on Fashion-MNIST while maintaining physical plausibility.

**Paper**: "Towards Physical Plausibility in Neuroevolution Systems" (https://arxiv.org/abs/2401.17733v1)

**Dataset**: Fashion-MNIST - 28x28 grayscale images of 10 clothing categories.

**Model**: ENERGIZE - A neuroevolution-based system that emphasizes physical plausibility and energy efficiency in neural network design.

## Baseline Performance

The current ENERGIZE model achieves the following baseline performance on Fashion-MNIST:

- **Accuracy**: 0.902 (90.2%)
- **Percentage Error**: 9.8%
- **Power Consumption**: 71.92

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline Accuracy (**higher is better**):

**Scoring**: 
```
Score = (new_accuracy - baseline_accuracy) / baseline_accuracy * 100
```
Example: Accuracy = 0.930 (93.0%) → Score = (0.930 - 0.902) / 0.902 * 100 = 3.10%

**Note**: For Accuracy, **higher values are better**. Percentage Error and Power Consumption are reported for analysis but do not affect the score.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_fashion_mnist_energize_main
```

## Training Time

| Training Time |
|---------------|
| 2 hours 30 minutes |

## Resources Provided

- **Research Paper**: Original ENERGIZE paper on physical plausibility in neuroevolution
- **Source Code**: Complete implementation of the ENERGIZE model (repository not publicly available)
- **Dataset**: Fashion-MNIST dataset (82MB)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM (neuroevolution can be memory-intensive)
- **Framework**: PyTorch or TensorFlow
- **Key Dependencies**: torch, numpy, PIL, scikit-learn, DEAP (evolutionary algorithms)

## File Structure

```
pwc_fashion_mnist_energize/
├── pwc_fashion_mnist_energize.py  # Main task definition
├── manifest.yaml                  # Task configuration
├── build_steps.json               # Environment setup
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── assets/
    ├── score.py                   # Scoring logic
    └── for_agent/
        ├── paper.pdf              # Research paper
        ├── repo.zip               # ENERGIZE source code
        ├── dataset.zip            # Fashion-MNIST (82MB)
        └── solution.py            # Implementation template
```

## Implementation Guidelines

1. **Understand the Fashion-MNIST Dataset**:
   - 60,000 training images, 10,000 test images
   - 28x28 grayscale images
   - 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
   - More challenging than MNIST due to higher visual complexity

2. **Study the ENERGIZE Model**:
   - Neuroevolution-based approach
   - Physical plausibility constraints
   - Energy-efficient neural network design
   - Evolution-based optimization (genetic algorithms, evolution strategies)
   - Multi-objective optimization (accuracy vs power consumption)
   - Biologically-inspired architectures

3. **Analyze Physical Plausibility**:
   - Realistic energy consumption models
   - Biological constraints on network structure
   - Efficient synaptic connections
   - Sparse connectivity patterns
   - Power-aware training and inference

4. **Identify Improvements**:
   - Enhanced neuroevolution operators (mutation, crossover)
   - Better fitness function design (balance accuracy and power)
   - Improved population management and diversity
   - Novel architectural search strategies
   - Data augmentation for robustness
   - Regularization techniques
   - Ensemble methods with evolved networks

5. **Neuroevolution Specific Considerations**:
   - Population size and generations
   - Selection pressure and elitism
   - Mutation and crossover rates
   - Fitness evaluation strategies
   - Parallel fitness evaluation
   - Speciation and diversity preservation
   - Multi-objective Pareto optimization

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess Fashion-MNIST dataset
    # 2. Train/evolve improved ENERGIZE model
    # 3. Evaluate on test set
    # 4. Calculate metrics
    
    return {
        'accuracy': your_accuracy,                       # e.g., 0.930 (higher is better)
        'percentage_error': your_error,                  # e.g., 7.5 (lower is better)
        'power_consumption': your_power_consumption      # e.g., 65.0 (lower is better)
    }
```

## Key Concepts

- **Neuroevolution**: Using evolutionary algorithms to optimize neural network architectures and weights
- **Physical Plausibility**: Neural networks that respect biological and physical constraints
- **Energy Efficiency**: Minimizing power consumption during training and inference
- **Multi-objective Optimization**: Balancing multiple objectives (accuracy, power, model size)
- **Genetic Algorithms**: Evolution-inspired optimization with selection, crossover, and mutation
- **Fitness Function**: Objective function guiding the evolution process
- **Fashion-MNIST**: Dataset of clothing items, more visually complex than MNIST digits

## Fashion-MNIST Dataset Details

The Fashion-MNIST dataset contains:
- **Source**: Zalando's article images
- **Size**: 60,000 training + 10,000 test images
- **Format**: 28x28 grayscale images
- **Classes**: 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Challenge**: More complex than MNIST due to intra-class variation and inter-class similarity
- **Applications**: Fashion recognition, retail automation, image understanding

## Improvement Ideas

1. **Neuroevolution Enhancements**:
   - Advanced evolutionary operators
   - NEAT (NeuroEvolution of Augmenting Topologies)
   - CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
   - Novelty search
   - Multi-objective evolutionary algorithms (NSGA-II, SPEA2)

2. **Physical Plausibility Improvements**:
   - Biologically-inspired activation functions
   - Sparse connectivity constraints
   - Energy-aware layer design
   - Synaptic pruning strategies
   - Efficient weight representations

3. **Training Strategies**:
   - Hybrid gradient descent + evolution
   - Data augmentation (rotation, shift, zoom)
   - Transfer learning from evolved architectures
   - Ensemble of evolved models
   - Progressive evolution (simple to complex)

4. **Energy Efficiency**:
   - Quantization techniques
   - Sparse activations
   - Efficient operations (depth-wise convolutions)
   - Power-aware fitness functions
   - Model compression techniques

5. **Architecture Search**:
   - Automated architecture discovery
   - Cell-based search spaces
   - Hierarchical evolution
   - Weight sharing across population
   - Progressive complexity increase

## References

- Paper: [Towards Physical Plausibility in Neuroevolution Systems](https://arxiv.org/abs/2401.17733v1)
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Neuroevolution: Evolution-based optimization of neural networks with physical constraints
- Energy-Efficient Neural Networks: Designing models that minimize power consumption