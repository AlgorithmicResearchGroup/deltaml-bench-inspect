# Food-101 Image Classification with MANO

## Overview
This task challenges agents to improve the Swin-MANO model for Food-101 image classification. MANO (Multipole Attention Neural Operator) is a novel linear attention mechanism that incorporates global context.

## Task Description
**Paper:** [Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics](https://arxiv.org/abs/2507.02748)

**Venue:** ECLR, ICCV 2025

**Model:** Swin Transformer with MANO attention mechanism

**Dataset:** Food-101 (4.7GB, downloaded from GCS at runtime)
- 101 food categories
- 101,000 images total
- 75,750 training images
- 25,250 test images

## Baseline Performance
- **Top-1 Accuracy:** 82.48% (estimated baseline for Swin-MANO on Food-101)

## Training Time

**Estimated Training Time**: 15 hours 56 minutes
- **Configuration**: 16 minutes × 100 epochs = 27 hours
- **Previous verification**: 15 hours 56 minutes total

## Goal
Achieve higher Top-1 classification accuracy by improving the Swin-MANO model.

## Evaluation Metric
**Top-1 Accuracy:** Percentage of test images correctly classified.

## Scoring Formula
```
Score = (your_accuracy - baseline_accuracy) / baseline_accuracy × 100
```

### Examples
- **Accuracy 88.0%** → Score: **6.69%** improvement
- **Accuracy 85.0%** → Score: **3.05%** improvement  
- **Accuracy 82.48%** → Score: **0%** (matches baseline)
- **Accuracy 80.0%** → Score: **0%** (worse than baseline)

## Approach
The Swin-MANO model combines:
1. **Swin Transformer:** Hierarchical vision transformer with shifted windows
2. **MANO Attention:** Linear attention with global context via multipole expansion
3. **Benefits:** O(n) complexity vs O(n²) in standard attention

## Improvement Strategies
1. **Enhanced MANO Attention**
   - Optimize multipole order and expansion parameters
   - Experiment with different attention configurations
   - Improve global context aggregation

2. **Architecture Improvements**
   - Adjust model depth and width
   - Optimize patch size and window size
   - Better downsampling strategies

3. **Training Enhancements**
   - Advanced data augmentation (RandAugment, MixUp, CutMix)
   - Longer training with better schedules
   - Knowledge distillation from larger models
   - Self-supervised pretraining

4. **Optimization**
   - Better hyperparameters (lr, weight decay)
   - Advanced optimizers (AdamW, LAMB)
   - Gradient clipping and mixed precision

## Files Structure
```
/home/agent/
├── paper.pdf              # Research paper
├── solution/              # MANO codebase
│   ├── models/            # Model implementations
│   ├── config/            # Configuration files
│   │   └── cfg_IC.yaml    # Image classification config
│   ├── trainer/           # Training utilities
│   ├── train_IC.py        # Training script
│   └── solution.py        # Your implementation
└── dataset/               # Food-101 dataset (downloaded from GCS)
```

## Getting Started
1. Review the MANO paper to understand the attention mechanism
2. Examine the Swin-MANO architecture in `solution/models/`
3. Check the configuration in `config/cfg_IC.yaml`
4. Implement your training and evaluation in `solution/solution.py`
5. Test with the provided `evaluate()` function

## Solution Interface
Your `solution.py` must implement:

```python
def evaluate() -> Dict[str, float]:
    """
    Returns:
        {
            'accuracy': float  # Top-1 accuracy (0-100)
        }
    """
```

## Common Pitfalls
- Insufficient data augmentation for Food-101
- Not training long enough (need 100+ epochs)
- Poor learning rate schedule
- Forgetting to normalize Food-101 images properly
- Not leveraging mixed precision training
- Suboptimal MANO attention configuration

## Dataset Note
**Important:** The Food-101 dataset is 4.7GB and will be downloaded from GCS at runtime to comply with Vivaria's 200MB upload limit.

## Resources
- **Paper:** `/home/agent/paper.pdf`
- **Code:** `/home/agent/solution/`
- **Dataset:** `/home/agent/dataset/` (downloaded from GCS)
- **Config:** `/home/agent/solution/config/cfg_IC.yaml`

## References
- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [MANO Paper](https://arxiv.org/abs/2507.02748)

Good luck!

