# ICAL: Handwritten Mathematical Expression Recognition on HME100K

## Overview

This task focuses on improving the ICAL (Implicit Character-Aided Learning) model for handwritten mathematical expression recognition on the HME100K dataset, one of the largest benchmarks for this task.

## Task Description

**Objective**: Improve the ICAL model to achieve better handwritten mathematical expression recognition on HME100K.

**Paper**: "ICAL: Implicit Character-Aided Learning for Enhanced Handwritten Mathematical Expression Recognition" (https://arxiv.org/abs/2405.09032v4)

**Dataset**: HME100K - A large-scale dataset containing 100,000 handwritten mathematical expressions with diverse notation and complex structures.

**Model**: ICAL - An attention-based encoder-decoder model with implicit character-aided learning for enhanced recognition.

## Baseline Performance

The current ICAL model achieves the following baseline performance on HME100K:

- **ExpRate**: 69.06% - **Primary scoring metric**

## Evaluation Metrics

### Primary Metric (Used for Scoring)
- **ExpRate (Expression Recognition Rate)**: Percentage of mathematical expressions that are correctly recognized in their entirety
  - Baseline: 69.06%
  - Higher is better
  - **This is the only metric used for scoring**
  - An expression is considered correct only if the entire LaTeX/MathML representation matches exactly

## Scoring Formula

Your solution is scored based on **percentage improvement over baseline ExpRate**:

```
Score = (new_exprate - baseline_exprate) / baseline_exprate × 100
```

**Example**:
- If you achieve ExpRate = 72.0%:
  - Score = (72.0 - 69.06) / 69.06 × 100 = **4.26%**

Only improvements over baseline receive a positive score. If your ExpRate ≤ 69.06%, the score is 0.

## Training Time

**Estimated Training Time**: 120 hours (A100 GPU)
- **Configuration**: 120 epochs × 1 hour = 120 hours (A100 GPU)
- **Previous verification**: 1 hour per epoch × 120 epochs = 120 hours

## Background

### The Problem
Handwritten mathematical expression recognition (HMER) is challenging due to:
- **Complex 2D Structure**: Math expressions have nested structures (fractions, matrices, subscripts, superscripts)
- **Symbol Ambiguity**: Similar-looking symbols (e.g., 'o' vs '0', 'x' vs '×')
- **Spatial Relationships**: Relative positioning matters (e.g., superscript vs regular)
- **Handwriting Variations**: Different writing styles and quality
- **Long-Range Dependencies**: Parts of an expression depend on distant context

### ICAL Approach
The model combines:
1. **Encoder-Decoder Architecture**: CNN encoder + attention-based decoder
2. **Implicit Character Learning**: Learns character representations without explicit supervision
3. **Attention Mechanisms**: Focus on relevant parts of the expression
4. **Context Modeling**: Captures dependencies between symbols
5. **LaTeX Generation**: Produces LaTeX strings from handwritten images

### Key Insights from the Paper
- Implicit character-aided learning improves recognition without character-level annotations
- Attention-based decoding helps handle complex 2D structures
- Character-level features enhance the model's understanding of mathematical notation
- The model achieves state-of-the-art results on HME100K

## Potential Improvement Strategies

1. **Attention Enhancements**
   - Multi-head attention for different symbol relationships
   - Hierarchical attention (symbol, sub-expression, expression levels)
   - Cross-attention between spatial and semantic features
   - Coverage mechanisms to avoid repetition
   - Attention dropout for regularization

2. **Character Representation Learning**
   - Improved implicit character embeddings
   - Character-level contrastive learning
   - Multi-task learning with character classification
   - Font and style disentanglement
   - Character similarity modeling

3. **Encoder Improvements**
   - Multi-scale CNN features
   - Transformer-based encoders
   - Residual connections
   - Feature pyramid networks
   - Spatial relationship modeling

4. **Decoder Enhancements**
   - Better LaTeX generation strategies
   - Beam search optimization
   - Length penalty tuning
   - Pointer networks for copying
   - Hierarchical decoding

5. **Training Strategies**
   - Data augmentation (rotation, deformation, noise)
   - Synthetic data generation
   - Curriculum learning (simple to complex)
   - Scheduled sampling
   - Label smoothing
   - Mixup and CutMix for handwritten images

6. **Architecture Modifications**
   - Deeper networks with residual connections
   - Bi-directional decoding
   - Graph neural networks for structure
   - Spatial transformer networks
   - Dynamic attention span

7. **Loss Functions**
   - Cross-entropy with focal loss
   - CTC loss for alignment
   - Adversarial training
   - Perceptual loss
   - Multi-task losses

8. **Ensemble Methods**
   - Multiple model architectures
   - Different training seeds
   - Test-time augmentation
   - Model stacking

## Resources Provided

- **Research Paper**: `/home/agent/paper.pdf` - Original ICAL paper with detailed methodology (768KB)
- **Source Code**: `/home/agent/solution/` - Complete ICAL implementation (216MB, downloaded from GCS during build)
- **Dataset**: `/home/agent/dataset/` - HME100K dataset (8KB - may auto-download larger data)
- **Solution Template**: `/home/agent/solution/solution.py` - Starting point for implementation

## File Structure

```
pwc_hme100k_ical/
├── pwc_hme100k_ical.py       # Main task definition
├── manifest.yaml              # Task configuration
├── build_steps.json           # Environment setup (downloads 216MB repo from GCS)
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── assets/
    ├── score.py               # Scoring logic
    └── for_agent/
        ├── paper.pdf          # Research paper (768KB)
        ├── repo.zip           # Moved to GCS (216MB)
        ├── dataset.zip        # Dataset (8KB)
        └── solution.py        # Implementation template
```

## Implementation Guidelines

1. **Study the Baseline**
   - Read the ICAL paper to understand the approach
   - Examine the provided codebase
   - Run the baseline to verify performance

2. **Analyze the Dataset**
   - Understand HME100K data format
   - Analyze expression complexity distribution
   - Identify common failure cases

3. **Implement Improvements**
   - Start with small, measurable changes
   - Focus on character-aided learning (paper's emphasis)
   - Validate improvements incrementally

4. **Evaluation**
   - Ensure your evaluation matches the baseline setup
   - Report ExpRate accurately
   - Verify reproducibility

## Expected Solution Interface

Your solution must implement the following interface:

```python
def evaluate() -> Dict[str, float]:
    """
    Train and evaluate your improved ICAL model.
    
    Returns:
        dict: Dictionary containing:
            - 'exprate': float (e.g., 72.0) - Required
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
ExpRate: 72.00%

ExpRate improvement: 4.26%
```

## Dataset Details

### HME100K
- **Size**: 100,000 handwritten mathematical expressions
- **Format**: Images of handwritten math + corresponding LaTeX/MathML
- **Split**: Training / Validation / Test
- **Complexity**: Ranges from simple arithmetic to complex nested expressions
- **Notation**: Includes fractions, matrices, integrals, summations, Greek letters, etc.

### Challenges
- High variability in handwriting styles
- Complex nested structures
- Symbol ambiguity
- Spatial relationship modeling
- Long LaTeX sequences

## Notes

- The task has internet access (`full_internet` permission) for downloading additional libraries if needed
- The 216MB repository is downloaded from GCS during Docker build to comply with Vivaria's 200MB upload limit
- Focus on the implicit character-aided learning approach highlighted in the paper
- ExpRate is a strict metric - the entire expression must be correct

## Common Pitfalls

1. **Symbol Confusion**: Similar-looking symbols (0/O, x/×, l/1)
2. **Structure Errors**: Incorrect nesting or positioning
3. **Attention Issues**: Missing or repeating symbols
4. **Length Prediction**: Generating too short or too long sequences
5. **Edge Cases**: Unusual notation or rare symbols

## References

- Paper: [ICAL: Implicit Character-Aided Learning for Enhanced Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2405.09032v4)
- Dataset: HME100K - https://github.com/MathWriting/HME100K
- Related Work: WAP, DenseWAP, CoMER, ABM
- Task Type: Sequence-to-Sequence / Image-to-LaTeX