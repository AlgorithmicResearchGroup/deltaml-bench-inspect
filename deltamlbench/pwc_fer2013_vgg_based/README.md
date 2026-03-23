# VGG-Based Facial Expression Recognition on FER2013

## Overview

This task focuses on improving a VGG-based model for facial expression recognition on the FER2013 dataset.

## Task Description

**Objective**: Improve the VGG-based model to achieve better facial expression recognition performance on FER2013 using 5-class classification.

**Paper**: "IdentiFace : A VGG Based Multimodal Facial Biometric System" (https://arxiv.org/abs/2401.01227v2)

**Dataset**: FER2013 - A facial expression recognition dataset with 48x48 grayscale face images.

**Model**: VGG-based (Visual Geometry Group) - Deep convolutional neural network architecture with small filters and multiple stacked layers.

## Baseline Performance

The current VGG-based model achieves the following baseline performance on FER2013:

- **5-class Test Accuracy**: 66.13%

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline 5-class Test Accuracy (**higher is better**):

**Scoring**: 
```
Score = (new_accuracy - baseline_accuracy) / baseline_accuracy * 100
```
Example: Accuracy = 70.00% → Score = (70.00 - 66.13) / 66.13 * 100 = 5.85%

**Note**: For Accuracy, **higher values are better**.

## Critical Bug Fixed

⚠️ **The original scoring script had a critical bug**: The baseline was set to **0.0** instead of **66.13%**, which would cause division by zero errors or incorrect scoring. This has been **fixed** in the current version.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_fer2013_vgg_based_main
```

## Training Time

| Training Time |
|---------------|
| 20 minutes |

## Resources Provided

- **Research Paper**: IdentiFace paper with VGG-based multimodal facial biometric system
- **Source Code**: Complete implementation of the VGG-based model (repository not publicly available)
- **Dataset**: FER2013 dataset (5.1KB - likely metadata, actual images may be downloaded)
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM
- **Framework**: PyTorch or TensorFlow
- **Key Dependencies**: torch, torchvision, numpy, PIL, opencv, scikit-learn

## File Structure

```
pwc_fer2013_vgg_based/
├── pwc_fer2013_vgg_based.py  # Main task definition
├── manifest.yaml             # Task configuration
├── build_steps.json          # Environment setup
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── assets/
    ├── score.py              # Scoring logic (FIXED baseline from 0.0 to 66.13%)
    └── for_agent/
        ├── paper.pdf         # Research paper
        ├── repo.zip          # VGG-based model source code
        ├── dataset.zip       # FER2013 dataset (5.1KB)
        └── solution.py       # Implementation template (FIXED baseline from 0.0 to 66.13%)
```

## Implementation Guidelines

1. **Understand the FER2013 Dataset**:
   - Facial Expression Recognition 2013
   - 48x48 pixel grayscale face images
   - Originally 7 emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
   - This task uses 5 classes (likely merged some classes)
   - ~35,000 images total (train + validation + test)
   - Challenging due to low resolution and varied lighting/pose

2. **Study the VGG Architecture**:
   - Visual Geometry Group architecture from Oxford
   - Key principles:
     - Small 3x3 convolutional filters
     - Multiple stacked conv layers
     - Max pooling for spatial downsampling
     - Deep networks (11-19 layers in original VGG)
     - Fully connected layers at the end
   - Variants: VGG11, VGG13, VGG16, VGG19

3. **Analyze Facial Expression Recognition**:
   - Subtle facial features distinguish emotions
   - Low-resolution 48x48 images are challenging
   - Face alignment and preprocessing crucial
   - Data augmentation important for robustness
   - Class imbalance may be present

4. **Identify Improvements**:
   - Deeper VGG variants (VGG16, VGG19)
   - Transfer learning from ImageNet pre-trained VGG
   - Batch normalization between layers
   - Dropout for regularization
   - Data augmentation (horizontal flip, rotation, brightness, contrast)
   - Advanced optimizers (Adam, SGD with momentum)
   - Learning rate scheduling
   - Ensemble of multiple VGG models
   - Attention mechanisms
   - Residual connections (VGG + ResNet hybrid)

5. **FER-Specific Considerations**:
   - Face preprocessing (alignment, cropping, normalization)
   - Handle class imbalance with weighted loss
   - Data augmentation preserving facial features
   - Proper train/validation/test splits
   - Cross-validation strategies
   - Fine-tuning strategies for FER

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess FER2013 dataset
    # 2. Train improved VGG-based model
    # 3. Evaluate on test set with 5-class classification
    # 4. Calculate accuracy
    
    return {
        'accuracy': your_accuracy  # e.g., 70.00 (higher is better)
    }
```

## Key Concepts

- **VGG**: Deep CNN architecture using small 3x3 filters and many layers
- **Facial Expression Recognition (FER)**: Classifying emotions from facial images
- **FER2013**: Standard benchmark dataset for facial expression recognition
- **5-class Classification**: Predicting one of 5 emotion categories
- **Transfer Learning**: Using pre-trained models (e.g., VGG on ImageNet) as starting point
- **Data Augmentation**: Artificially expanding training data with transformations

## FER2013 Dataset Details

The FER2013 dataset contains:
- **Source**: Faces collected from the web
- **Format**: 48x48 grayscale images
- **Original Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **This Task**: 5 classes (specific classes depend on merging strategy)
- **Size**: ~35,000 images total
- **Challenge**: Low resolution, varied conditions, subtle expressions
- **Applications**: Emotion recognition, human-computer interaction, mental health monitoring

## Improvement Ideas

1. **Architecture Enhancements**:
   - Use deeper VGG variants (VGG16, VGG19)
   - Add batch normalization after conv layers
   - Add dropout for regularization
   - Residual connections for better gradient flow
   - Attention mechanisms (channel attention, spatial attention)
   - Multi-scale feature extraction

2. **Transfer Learning**:
   - Start with VGG pre-trained on ImageNet
   - Fine-tune on FER2013
   - Progressive unfreezing of layers
   - Layer-wise learning rate adaptation

3. **Training Strategies**:
   - Data augmentation (flip, rotation, brightness, contrast)
   - Learning rate scheduling (step decay, cosine annealing)
   - Advanced optimizers (Adam, AdamW, SGD with momentum)
   - Early stopping with patience
   - Gradient clipping
   - Mixup or cutmix augmentation

4. **Data Processing**:
   - Face alignment and normalization
   - Histogram equalization for lighting
   - Adaptive preprocessing per image
   - Handle class imbalance (weighted loss, oversampling)
   - Cross-validation for robust evaluation

5. **Ensemble Methods**:
   - Multiple VGG models with different depths
   - Different initialization seeds
   - Voting or averaging predictions
   - Stacking with meta-learner

## References

- Paper: [IdentiFace : A VGG Based Multimodal Facial Biometric System](https://arxiv.org/abs/2401.01227v2)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- FER2013: Challenges in Representation Learning competition (ICML 2013)
- Facial Expression Recognition: Survey at https://arxiv.org/abs/1804.08348