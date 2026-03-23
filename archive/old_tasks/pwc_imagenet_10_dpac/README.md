# ZLaP: Zero-shot Label Propagation for ImageNet Classification

## Overview

This task focuses on improving the ZLaP (Zero-shot Label Propagation) model for image classification on ImageNet using vision-language models and graph-based label propagation.

**Note**: Despite the directory name (`pwc_imagenet_10_dpac`), this task is about the **ZLaP** model for zero-shot classification on ImageNet.

## Task Description

**Objective**: Improve the ZLaP model to achieve better zero-shot classification performance on ImageNet.

**Paper**: "Label Propagation for Zero-shot Classification with Vision-Language Models" (https://arxiv.org/abs/2404.04072v1)

**Dataset**: ImageNet - Large-scale image classification benchmark with 1,000 classes.

**Model**: ZLaP - Combines pre-trained vision-language models (like CLIP) with graph-based label propagation for improved zero-shot classification.

## Baseline Performance

The current ZLaP model achieves the following baseline performance on ImageNet:

- **Percentage Error**: 11.91% (lower is better)
- **Accuracy**: 88.09% (higher is better - primary scoring metric)

## Evaluation Metrics

### Primary Metric (Used for Scoring)
- **Accuracy**: Percentage of images correctly classified
  - Baseline: 88.09%
  - Higher is better
  - **This is the primary metric used for scoring**

### Secondary Metric (Reported but not scored)
- **Percentage Error**: Percentage of incorrect predictions
  - Baseline: 11.91%
  - Lower is better
  - **This metric is reported for analysis but does not affect the score**

## Scoring Formula

Your solution is scored based on **percentage improvement over baseline Accuracy**:

```
Score = (new_accuracy - baseline_accuracy) / baseline_accuracy × 100
```

**Example**:
- If you achieve Accuracy = 90.0%:
  - Score = (90.0 - 88.09) / 88.09 × 100 = **2.17%**

Only improvements over baseline receive a positive score. If your Accuracy ≤ 88.09%, the score is 0.

## Training Time

**Estimated Training Time**: 36 hours

## Performance Metrics

| Metric | Training Time |
|--------|---------------|
| Estimated Duration | 36 hours |

## Background

### The Problem
Zero-shot image classification aims to classify images into categories not seen during training. Traditional approaches rely solely on pre-trained vision-language models (VLMs) like CLIP, but they can be improved.

**Challenges**:
- **Limited Training Data**: No access to ImageNet training labels (zero-shot)
- **Ambiguous Classes**: Many ImageNet classes are visually similar
- **Domain Gap**: Pre-trained VLMs may not align perfectly with ImageNet
- **Single-Shot Predictions**: Traditional VLMs make independent predictions per image

### ZLaP Approach
The model combines:
1. **Vision-Language Models**: Pre-trained models (CLIP, ALIGN, etc.) for initial predictions
2. **Label Propagation**: Graph-based semi-supervised learning to refine predictions
3. **Similarity Graph**: Connects similar images to propagate information
4. **Zero-shot Setting**: No training on ImageNet labels, only evaluation

### Key Insights from the Paper
- Label propagation can significantly improve zero-shot classification
- Building similarity graphs over test images helps correct individual errors
- Combining VLM predictions with graph structure is powerful
- The approach is training-free and works with any pre-trained VLM

## Potential Improvement Strategies

1. **Label Propagation Enhancements**
   - Better graph construction (k-NN, ε-radius, learned edges)
   - Improved propagation algorithms (Laplacian, PageRank, diffusion)
   - Multi-hop vs. single-hop propagation
   - Weighted edge construction based on feature similarity
   - Iterative refinement strategies

2. **Vision-Language Model Improvements**
   - Better pre-trained models (larger CLIP, OpenCLIP, EVA-CLIP)
   - Ensemble multiple VLMs
   - Model-specific prompt engineering
   - Temperature scaling for better calibration
   - Multi-view predictions (different crops/augmentations)

3. **Feature Extraction**
   - Multi-scale features (different layers, resolutions)
   - Better image preprocessing
   - Feature normalization strategies
   - Dimensionality reduction (PCA, t-SNE)
   - Learned feature projections

4. **Graph Construction**
   - Adaptive k for k-NN graphs
   - Symmetrization strategies
   - Edge weighting schemes
   - Hierarchical graph structures
   - Dynamic graph updates during propagation

5. **Prompt Engineering**
   - Better class name templates
   - Multiple prompts per class
   - Context-aware prompts
   - Ensemble over prompt variations
   - Task-specific prompt tuning

6. **Calibration and Post-processing**
   - Temperature scaling
   - Platt scaling
   - Confidence thresholding
   - Prediction smoothing
   - Ensemble with baseline VLM

7. **Hybrid Approaches**
   - Combine multiple propagation methods
   - Iterative label propagation
   - Self-training with high-confidence predictions
   - Transductive learning techniques

## Resources Provided

- **Research Paper**: `/home/agent/paper.pdf` - Original ZLaP paper with detailed methodology (1MB)
- **Source Code**: `/home/agent/solution/` - Complete ZLaP implementation (128KB)
- **Dataset**: ImageNet (typically auto-downloaded via torchvision/HuggingFace)
- **Solution Template**: `/home/agent/solution/solution.py` - Starting point for implementation

## File Structure

```
pwc_imagenet_10_dpac/
├── pwc_imagenet_10_dpac.py   # Main task definition
├── manifest.yaml              # Task configuration
├── build_steps.json           # Environment setup
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── assets/
    ├── score.py               # Scoring logic
    └── for_agent/
        ├── paper.pdf          # Research paper (1MB)
        ├── repo.zip           # Source code (128KB)
        └── solution.py        # Implementation template
```

## Implementation Guidelines

1. **Study the Baseline**
   - Read the ZLaP paper to understand label propagation
   - Examine the provided codebase
   - Run the baseline to verify performance

2. **Analyze the Approach**
   - Understand how label propagation works
   - Study the graph construction
   - Identify bottlenecks in the pipeline

3. **Implement Improvements**
   - Start with small, measurable changes
   - Focus on graph-based propagation (paper's emphasis)
   - Validate improvements on a subset first

4. **Evaluation**
   - Ensure zero-shot setting (no ImageNet training)
   - Evaluate on ImageNet validation set (50K images)
   - Report Top-1 Accuracy accurately
   - Verify reproducibility

## Expected Solution Interface

Your solution must implement the following interface:

```python
def evaluate() -> Dict[str, float]:
    """
    Evaluate your improved ZLaP model on ImageNet.
    
    Returns:
        dict: Dictionary containing:
            - 'percentage_error': float (e.g., 10.5) - Required
            - 'accuracy': float (e.g., 89.5) - Required
    """
    # Your implementation here
    pass
```

## Validation

To test your solution locally:

```bash
cd /home/agent/solution
python solution.py
```

Expected output format:
```
Percentage Error: 10.50%
Accuracy: 89.50%

Percentage Error improvement: 11.84%
Accuracy improvement: 1.60%
```

## Dataset Details

### ImageNet
- **Size**: 1,000 classes, ~1.28M training images, 50K validation images
- **Format**: Images of various sizes (typically resized to 224×224 or 384×384)
- **Evaluation**: Zero-shot on validation set (no training on ImageNet labels)
- **Classes**: Wide variety (animals, objects, scenes, etc.)
- **Download**: Typically via `torchvision.datasets.ImageNet` or HuggingFace

### Zero-shot Setting
- **No Training**: Cannot train on ImageNet training set
- **Pre-trained Models Only**: Use models pre-trained on other datasets
- **Evaluation Only**: Run inference on validation set
- **Label Propagation**: Can use test-time graph structures

## Common Challenges

1. **Computational Cost**: Label propagation over 50K images can be expensive
2. **Graph Sparsity**: Building effective similarity graphs at scale
3. **Class Imbalance**: ImageNet has varying numbers of images per class
4. **Similar Classes**: Many fine-grained distinctions (dog breeds, bird species)
5. **VLM Limitations**: Pre-trained models may have biases or gaps

## Notes

- The task has internet access (`full_internet` permission) for downloading models/datasets
- Focus on the label propagation approach highlighted in the paper
- Zero-shot constraint: no training on ImageNet labels
- Pre-trained vision-language models (CLIP, etc.) are allowed
- Graph-based methods can significantly improve performance

## References

- Paper: [Label Propagation for Zero-shot Classification with Vision-Language Models](https://arxiv.org/abs/2404.04072v1)
- ImageNet: https://www.image-net.org/
- CLIP: https://github.com/openai/CLIP
- Label Propagation: Semi-supervised learning technique
- Task Type: Zero-shot Image Classification