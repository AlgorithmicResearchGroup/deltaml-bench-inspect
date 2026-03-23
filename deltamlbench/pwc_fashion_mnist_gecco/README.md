# GECCO: Efficient Grayscale Image Classification on Fashion-MNIST

## Overview

This task focuses on improving the GECCO (Graph Convolution) model for image classification on the Fashion-MNIST dataset using efficient graph-based approaches.

## Task Description

**Objective**: Improve the GECCO model to achieve better image classification performance on Fashion-MNIST using single graph convolution.

**Paper**: "A Single Graph Convolution Is All You Need: Efficient Grayscale Image Classification" (https://arxiv.org/abs/2402.00564v6)

**Dataset**: Fashion-MNIST - 28x28 grayscale images of fashion products (10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

**Model**: GECCO - A graph convolution-based model that treats images as graphs and uses a single efficient graph convolution layer for classification.

## Baseline Performance

The current GECCO model achieves the following baseline performance on Fashion-MNIST:

- **Accuracy**: 88.09%
- **Percentage Error**: 11.91%

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline Accuracy (**higher is better**):

**Scoring**: 
```
Score = (new_accuracy - baseline_accuracy) / baseline_accuracy * 100
```
Example: Accuracy = 89.00% → Score = (89.00 - 88.09) / 88.09 * 100 = 1.03%

**Note**: For Accuracy, **higher values are better**. Percentage Error is reported for analysis but does not affect the score.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_fashion_mnist_gecco_main
```

## Training Time

| Training Time |
|---------------|
| 5 hours 30 minutes |

## Resources Provided

- **Research Paper**: Original GECCO paper on efficient graph convolution for image classification
- **Source Code**: Complete implementation of the GECCO model (repository not publicly available)
- **Dataset**: Fashion-MNIST dataset (82MB)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 4-8GB RAM
- **Framework**: PyTorch or TensorFlow with graph neural network support
- **Key Dependencies**: torch, torch-geometric, numpy, PIL, scikit-learn

## File Structure

```
pwc_fashion_mnist_gecco/
├── pwc_fashion_mnist_gecco.py  # Main task definition
├── manifest.yaml               # Task configuration
├── build_steps.json            # Environment setup
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── assets/
    ├── score.py                # Scoring logic
    └── for_agent/
        ├── paper.pdf           # Research paper
        ├── repo.zip            # GECCO source code
        ├── dataset.zip         # MNIST (82MB)
        └── solution.py         # Implementation template
```

## Implementation Guidelines

1. **Understand the MNIST Dataset**:
   - 60,000 training images, 10,000 test images
   - 28x28 grayscale images
   - 10 digit classes (0-9)
   - Standard benchmark for image classification
   - High baseline performance (>97% typical)

2. **Study the GECCO Model**:
   - Single graph convolution layer for efficiency
   - Converts images to graph representation
   - Each pixel becomes a node in the graph
   - Edges connect spatially adjacent pixels
   - Efficient learnable graph convolutions
   - Minimal parameters compared to CNNs

3. **Analyze Graph-based Image Representation**:
   - Pixel-to-node mapping strategies
   - Edge connectivity patterns (4-neighbors, 8-neighbors, etc.)
   - Node feature initialization (pixel intensity)
   - Spatial structure preservation in graphs
   - Graph pooling strategies

4. **Identify Improvements**:
   - Enhanced graph convolution operators
   - Better pixel-to-graph conversion methods
   - Adaptive edge connections based on pixel similarity
   - Multi-scale graph representations
   - Improved graph pooling
   - Data augmentation (rotation, translation)
   - Regularization techniques
   - Ensemble methods

5. **Graph Convolution Specific Considerations**:
   - Efficient message passing on graphs
   - Handling variable graph sizes
   - Spectral vs spatial graph convolutions
   - Over-smoothing in deep graph networks
   - Graph normalization strategies
   - Computational efficiency

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess MNIST dataset
    # 2. Convert images to graph representation
    # 3. Train improved GECCO model
    # 4. Evaluate on test set
    # 5. Calculate metrics
    
    return {
        'accuracy': your_accuracy,          # e.g., 98.50 (higher is better)
        'percentage_error': your_error      # e.g., 1.50 (lower is better)
    }
```

## Key Concepts

- **Graph Convolution**: Convolutional operation on graph-structured data
- **Graph Neural Networks (GNNs)**: Neural networks that operate on graph structures
- **Image-to-Graph**: Converting grid-structured images to graph representation
- **Message Passing**: Core operation in graph neural networks where nodes exchange information
- **Spatial Graph Convolution**: Convolution based on local neighborhoods in graphs
- **Node Features**: Attributes associated with each node (e.g., pixel intensity)
- **Edge Connectivity**: Pattern of connections between nodes in the graph
- **MNIST**: Standard benchmark dataset of handwritten digits

## MNIST Dataset Details

The MNIST dataset contains:
- **Source**: Handwritten digit images from NIST database
- **Size**: 60,000 training + 10,000 test images
- **Format**: 28x28 grayscale images
- **Classes**: 10 digits (0-9)
- **Baseline**: Modern models achieve >99% accuracy
- **Challenge**: Despite high baseline, improving further is non-trivial
- **Applications**: Digit recognition, OCR, benchmarking ML algorithms

## Improvement Ideas

1. **Graph Construction Enhancements**:
   - Adaptive edge connections based on pixel similarity
   - Multi-scale graph pyramids
   - Learned graph structure
   - Dynamic edge weighting
   - Long-range connections for global context

2. **Graph Convolution Improvements**:
   - Attention-based graph convolution
   - Graph transformers
   - Residual connections in graph layers
   - Spectral graph convolutions
   - Edge-conditioned convolutions

3. **Training Strategies**:
   - Data augmentation (rotation, shift, elastic deformation)
   - Mixup on graph representations
   - Knowledge distillation from CNNs
   - Self-supervised pre-training
   - Curriculum learning

4. **Architecture Modifications**:
   - Multiple graph convolution layers (while maintaining efficiency)
   - Hierarchical graph pooling
   - Global graph pooling strategies
   - Skip connections between graph layers
   - Batch normalization for graph networks

5. **Efficiency Optimizations**:
   - Sparse graph operations
   - Graph coarsening
   - Approximate nearest neighbors for edges
   - Quantization of graph features
   - Pruning redundant edges

## Troubleshooting

### Known Issue: CUDA Device-Side Assert

**Problem**: The provided repository code may trigger CUDA errors like:
```
AcceleratorError: CUDA error: device-side assert triggered
```

**Root Causes**:
- Graph construction may create invalid tensor indices
- Out-of-bounds array access in CUDA kernels
- NaN/Inf propagation during training
- Bugs in the original repository code

**Solutions**:

1. **Enable Synchronous CUDA Execution** (for debugging):
   ```python
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   ```

2. **Add CPU Fallback**:
   ```python
   import torch
   try:
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)
   except RuntimeError as e:
       print(f"CUDA error, falling back to CPU: {e}")
       device = torch.device('cpu')
       model = model.to(device)
   ```

3. **Validate Data Before GPU Operations**:
   ```python
   # Check for NaN/Inf
   if torch.isnan(tensor).any() or torch.isinf(tensor).any():
       tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
   
   # Validate indices
   assert labels.max() < num_classes, f"Label {labels.max()} exceeds num_classes {num_classes}"
   ```

4. **Fix Graph Construction**:
   - Ensure edge indices are valid: `0 <= edge_index < num_nodes`
   - Clamp or validate all indices before creating graphs
   - Check for empty graphs or disconnected components

5. **If All Else Fails, Use CPU**:
   ```python
   device = torch.device('cpu')
   ```
   Training will be slower but should complete successfully.

## References

- Paper: [A Single Graph Convolution Is All You Need: Efficient Grayscale Image Classification](https://arxiv.org/abs/2402.00564v6)
- MNIST: http://yann.lecun.com/exdb/mnist/
- Graph Neural Networks: Deep learning on graph-structured data
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

## Note on Directory Name

The directory is named `pwc_fashion_mnist_gecco` but this task actually uses **MNIST** (digits), not Fashion-MNIST (clothing). This is confirmed by the metadata and the 98.04% accuracy baseline, which is typical for MNIST but would be unusually high for Fashion-MNIST.