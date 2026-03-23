# RASM: Regional Attention for Shadow Removal on ISTD+

## Overview

This task focuses on improving the RASM (Regional Attention for Shadow Removal) model for removing shadows from images on the ISTD+ dataset.

## Task Description

**Objective**: Improve the RASM model to achieve better shadow removal performance on ISTD+.

**Paper**: "Regional Attention for Shadow Removal" (https://arxiv.org/abs/2411.14201v1)

**Dataset**: ISTD+ - A shadow removal dataset with shadow images, shadow masks, and shadow-free ground truth.

**Model**: RASM - An encoder-decoder model with regional attention mechanisms specifically designed for shadow detection and removal.

## Baseline Performance

The current RASM model achieves the following baseline performance on ISTD+:

- **RMSE**: 2.53 - **Primary scoring metric**

## Evaluation Metrics

### Primary Metric (Used for Scoring)
- **RMSE (Root Mean Square Error)**: Pixel-level error between shadow-removed image and ground truth
  - Baseline: 2.53
  - **Lower is better** (common error metric)
  - **This is the only metric used for scoring**

## Scoring Formula

Your solution is scored based on **percentage improvement over baseline RMSE**:

```
Score = (baseline_rmse - new_rmse) / baseline_rmse × 100
```

**Example**:
- If you achieve RMSE = 2.3:
  - Score = (2.53 - 2.3) / 2.53 × 100 = **9.09%**

**IMPORTANT**: RMSE is "lower is better", so **reducing RMSE gives positive improvement**.

Only improvements over baseline receive a positive score. If your RMSE ≥ 2.53, the score is 0.

## Training Time

**Estimated Training Time**: 4 days 10 hours

## Performance Metrics

| Metric | Training Time |
|--------|---------------|
| Estimated Duration | 4 days 10 hours |

## Background

### The Problem
Shadow removal aims to detect and remove shadows from images while preserving image quality and matching the appearance of non-shadow regions.

**Challenges**:
- **Shadow Detection**: Accurately identifying shadow regions
- **Color Correction**: Matching shadow region colors with non-shadow regions
- **Boundary Handling**: Managing hard and soft shadow boundaries
- **Detail Preservation**: Maintaining texture and structure in shadow regions
- **Illumination Consistency**: Ensuring consistent lighting across the image

### RASM Approach
The model combines:
1. **Encoder-Decoder Architecture**: Captures multi-scale features for shadow understanding
2. **Regional Attention**: Focuses on different shadow characteristics in various regions
3. **Shadow Mask Guidance**: Uses shadow masks to guide the removal process
4. **Spatial Attention**: Weights different spatial locations based on shadow properties
5. **Feature Fusion**: Combines low-level and high-level features for better removal

### Key Insights from the Paper
- Regional attention helps handle varying shadow properties (intensity, color, sharpness)
- Different regions require different removal strategies
- Attention mechanisms improve detail preservation
- Multi-scale features are crucial for accurate shadow removal

## Potential Improvement Strategies

1. **Regional Attention Enhancements**
   - Better region segmentation strategies
   - Adaptive attention based on shadow properties
   - Multi-head attention for different shadow characteristics
   - Hierarchical attention (global + local)
   - Cross-attention between shadow and non-shadow regions

2. **Shadow Detection Improvements**
   - Better shadow mask prediction
   - Hard vs. soft shadow classification
   - Edge-aware shadow boundary detection
   - Multi-scale shadow detection
   - Uncertainty-aware detection

3. **Color Correction**
   - Better illumination transfer from non-shadow to shadow regions
   - Physical illumination models
   - Color histogram matching
   - Perceptual color loss
   - Adaptive gamma correction

4. **Architecture Modifications**
   - Deeper encoder-decoder networks
   - Residual connections
   - Feature pyramid networks
   - Dilated convolutions for larger receptive fields
   - Transformer-based architectures

5. **Training Strategies**
   - Data augmentation (different shadow types, intensities)
   - Synthetic shadow generation
   - Multi-task learning (shadow detection + removal)
   - Adversarial training for realistic results
   - Progressive training (coarse to fine)

6. **Loss Functions**
   - Perceptual loss (VGG features)
   - Edge-aware loss
   - Color consistency loss
   - SSIM loss for structure preservation
   - Adversarial loss for realism
   - Multi-scale losses

7. **Post-processing**
   - Edge refinement
   - Color smoothing in shadow boundaries
   - Noise reduction
   - Sharpening in previously shadowed regions

## Resources Provided

- **Research Paper**: `/home/agent/paper.pdf` - Original RASM paper with detailed methodology (24MB)
- **Source Code**: `/home/agent/solution/` - Complete RASM implementation (22MB)
- **Dataset**: `/home/agent/dataset/` - ISTD+ dataset (8KB - may need separate download)
- **Solution Template**: `/home/agent/solution/solution.py` - Starting point for implementation

## File Structure

```
pwc_istd_rasm/
├── pwc_istd_rasm.py          # Main task definition
├── manifest.yaml              # Task configuration
├── build_steps.json           # Environment setup
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── assets/
    ├── score.py               # Scoring logic (CRITICAL BUG FIXED: RMSE is lower-is-better)
    └── for_agent/
        ├── paper.pdf          # Research paper (24MB)
        ├── repo.zip           # Source code (22MB)
        ├── dataset.zip        # Dataset (8KB)
        └── solution.py        # Implementation template
```

## Implementation Guidelines

1. **Study the Baseline**
   - Read the RASM paper to understand regional attention
   - Examine the provided codebase
   - Run the baseline to verify performance

2. **Analyze the Dataset**
   - Understand ISTD+ data format (shadow images, masks, ground truth)
   - Analyze shadow characteristics (hard/soft, intensity, color)
   - Identify common failure cases

3. **Implement Improvements**
   - Start with small, measurable changes
   - Focus on regional attention (paper's emphasis)
   - Validate improvements on validation set

4. **Evaluation**
   - Ensure your evaluation matches the baseline setup
   - Calculate RMSE accurately (pixel-level error)
   - Verify reproducibility

## Expected Solution Interface

Your solution must implement the following interface:

```python
def evaluate() -> Dict[str, float]:
    """
    Train and evaluate your improved RASM model.
    
    Returns:
        dict: Dictionary containing:
            - 'rmse': float (e.g., 2.3) - Required (lower is better!)
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
RMSE: 2.30

RMSE improvement: 9.09%
✓ Your RMSE is LOWER than baseline (better!)
```

## Dataset Details

### ISTD+
- **Size**: Training and test images with shadows
- **Format**: 
  - Shadow images (images with shadows)
  - Shadow masks (binary masks indicating shadow regions)
  - Shadow-free images (ground truth)
- **Image Size**: Typically 640×480 or similar
- **Shadow Types**: Various (hard shadows, soft shadows, different intensities)

### Challenges
- Varied shadow intensities and colors
- Complex shadow boundaries
- Multiple overlapping shadows
- Different surface materials under shadows
- Illumination variations

## Common Pitfalls

1. **Over-smoothing**: Removing too much detail in shadow regions
2. **Color Mismatch**: Shadow regions don't match non-shadow appearance
3. **Boundary Artifacts**: Visible seams at shadow boundaries
4. **Global vs. Local**: Applying uniform correction instead of regional adaptation
5. **Training Instability**: Shadow removal can be sensitive to training dynamics

## Critical Bug Fix

**⚠️ IMPORTANT**: The original scoring logic incorrectly treated RMSE as "higher is better". This has been fixed:

**Old (WRONG)**:
```python
if value <= baseline_value:
    return 0.0
improvement = (value - baseline_value) / baseline_value * 100
```

**New (CORRECT)**:
```python
if value >= baseline_value:  # RMSE lower is better
    return 0.0
improvement = (baseline_value - value) / baseline_value * 100
```

## Notes

- The task has internet access (`full_internet` permission) for downloading additional libraries if needed
- Focus on the regional attention approach highlighted in the paper
- RMSE is a strict metric - small improvements can be significant
- Shadow removal is challenging - baseline is already quite strong

## References

- Paper: [Regional Attention for Shadow Removal](https://arxiv.org/abs/2411.14201v1)
- Dataset: ISTD+ - https://github.com/cvlab-stonybrook/shadow-removal
- Related Work: DeshadowNet, ST-CGAN, DHAN, Auto
- Task Type: Image-to-Image Translation / Shadow Removal