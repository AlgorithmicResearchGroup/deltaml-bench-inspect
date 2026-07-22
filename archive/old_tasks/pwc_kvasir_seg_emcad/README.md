# Kvasir-SEG EMCAD: Medical Polyp Segmentation

## Overview

This task focuses on improving the EMCAD (Efficient Multi-scale Convolutional Attention Decoding) model for medical polyp segmentation on the Kvasir-SEG dataset, a benchmark dataset for gastrointestinal polyp detection.

## Task Description

**Objective**: Improve the EMCAD model to achieve better segmentation performance on the Kvasir-SEG dataset.

**Paper**: EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation

**Dataset**: Kvasir-SEG - A dataset of 1000 polyp images and corresponding segmentation masks from gastrointestinal endoscopy

## Baseline Performance

The current EMCAD model achieves:
- **mean Dice**: 0.928 (92.8%) - **Primary scoring metric**

## Evaluation Metrics

### Primary Metric (Used for Scoring)
- **mean Dice coefficient**: Measures overlap between predicted and ground truth segmentation
  - Baseline: 0.928
  - Higher is better (range: 0 to 1)
  - **This is the only metric used for scoring**

## Scoring Formula

Your solution is scored based on **percentage improvement over baseline mean Dice**:

```
Score = (new_dice - baseline_dice) / baseline_dice × 100
```

**Example**:
- If you achieve mean Dice = 0.950:
  - Score = (0.950 - 0.928) / 0.928 × 100 = **2.37%**

Only improvements over baseline receive a positive score. If your mean Dice ≤ 0.928, the score is 0.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_kvasir_seg_emcad_main
```

## Training Time

| Training Time |
|---------------|
| 21 hours |

## Background

### The Problem
Polyp detection and segmentation in gastrointestinal endoscopy is crucial for:
- **Early Cancer Detection**: Polyps can develop into colorectal cancer
- **Treatment Planning**: Accurate segmentation guides intervention
- **Clinical Decision Support**: Automated detection assists physicians
- **Quality Assurance**: Reduces missed polyps during procedures

### Challenges
- **Variable Appearance**: Polyps vary in size, shape, color, and texture
- **Difficult Boundaries**: Unclear edges between polyp and surrounding tissue
- **Lighting Conditions**: Inconsistent illumination in endoscopic images
- **Artifacts**: Specular highlights, motion blur, and image noise
- **Class Imbalance**: Polyp regions are small compared to background

### EMCAD Approach
The model combines:
1. **Multi-scale Features**: Captures polyps at different scales
2. **Convolutional Attention**: Focuses on relevant regions
3. **Efficient Decoding**: Fast inference for clinical use
4. **Skip Connections**: Preserves spatial information
5. **Deep Supervision**: Improves gradient flow during training

## Potential Improvement Strategies

1. **Architecture Enhancements**
   - Improve multi-scale feature extraction
   - Enhanced attention mechanisms (channel, spatial, self-attention)
   - Better encoder-decoder connections
   - Residual or dense connections
   - Feature pyramid networks

2. **Data Augmentation**
   - Geometric: rotation, flipping, elastic deformation
   - Color: brightness, contrast, saturation adjustments
   - Medical-specific: simulated lighting variations
   - MixUp or CutMix for polyp images
   - Test-time augmentation

3. **Loss Functions**
   - Dice loss for better overlap optimization
   - Focal loss for hard example mining
   - Boundary loss for edge refinement
   - Combined losses (e.g., Dice + BCE)
   - Multi-scale supervision

4. **Training Strategies**
   - Better learning rate scheduling
   - Warm-up strategies
   - Data resampling for class balance
   - Progressive training (coarse to fine)
   - Knowledge distillation

5. **Post-processing**
   - Conditional Random Fields (CRF)
   - Morphological operations
   - Connected component analysis
   - Ensemble predictions

6. **Ensemble Methods**
   - Multiple models with different architectures
   - Different training seeds
   - Test-time augmentation ensemble

## Resources Provided

- **Research Paper**: `/home/agent/paper.pdf` - Original EMCAD paper with detailed methodology
- **Source Code**: `/home/agent/solution_repo/` - Complete EMCAD implementation
- **Dataset**: `/home/agent/dataset/` - Kvasir-SEG dataset with images and masks
- **Solution Template**: `/home/agent/solution/solution.py` - Starting point with helper functions

## File Structure

```
pwc_kvasir_seg_emcad/
├── kvasir_seg_emcad.py     # Main task definition
├── manifest.yaml            # Task configuration
├── build_steps.json         # Environment setup
├── requirements.txt         # Dependencies (PyTorch, OpenCV, etc.)
├── README.md               # This file
└── assets/
    ├── score.py            # Scoring logic
    └── for_agent/
        ├── paper.pdf       # Research paper
        ├── repo.zip        # EMCAD source code
        ├── dataset.zip     # Kvasir-SEG dataset
        └── solution.py     # Implementation template with helpers
```

## Implementation Guidelines

1. **Study the Baseline**
   - Read the EMCAD paper to understand the architecture
   - Examine the provided codebase structure
   - Understand the multi-scale attention mechanism

2. **Analyze the Dataset**
   - Explore Kvasir-SEG image characteristics
   - Analyze polyp sizes, shapes, and appearances
   - Check data distribution and split
   - Identify challenging cases

3. **Implement Improvements**
   - Start with the provided EMCAD baseline
   - Make incremental, measurable changes
   - Focus on multi-scale attention and feature extraction
   - Validate improvements on validation set

4. **Evaluation**
   - Use proper train/validation/test splits
   - Calculate Dice coefficient correctly
   - Ensure reproducible results (set random seeds)
   - Verify your implementation matches expected format

## Expected Solution Interface

Your solution must implement the following interface:

```python
def evaluate() -> Dict[str, float]:
    """
    Train and evaluate your improved EMCAD model.
    
    Returns:
        dict: Dictionary containing:
            - 'mean_dice': float (e.g., 0.950) - Required
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
mean Dice: 0.9500

mean Dice improvement: 2.37%
```

## Dataset Details

### Kvasir-SEG
- **Images**: 1000 polyp images from gastrointestinal endoscopy
- **Masks**: Corresponding binary segmentation masks
- **Format**: RGB images with variable sizes
- **Quality**: High-resolution endoscopic images
- **Annotations**: Expert-validated segmentation masks

### Image Characteristics
- Variable polyp sizes (small to large)
- Different polyp types and locations
- Varying lighting and image quality
- Specular highlights and reflections
- Complex backgrounds

## Notes

- The task has GPU access (1x A100) for training
- Task has internet access (`full_internet` permission) if needed
- Focus on efficient multi-scale attention mechanisms
- mean Dice is a strict metric - both precision and recall matter
- Consider clinical applicability and inference speed

## Common Pitfalls

1. **Overfitting**: Limited dataset size, use augmentation and regularization
2. **Class Imbalance**: Polyp regions are small, adjust loss weights
3. **Boundary Errors**: Focus on edge refinement techniques
4. **Scale Variance**: Ensure multi-scale processing works well
5. **Evaluation**: Make sure Dice calculation is correct

## References

- Paper: EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation
- Dataset: Kvasir-SEG - https://datasets.simula.no/kvasir-seg/
- Task Type: Medical Image Segmentation / Polyp Detection
- Clinical Application: Colorectal Cancer Screening and Prevention