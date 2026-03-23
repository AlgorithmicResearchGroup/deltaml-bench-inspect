# Tiny ImageNet Classification with MANO-tiny

## Overview
Improve MANO-tiny model for Tiny ImageNet (200 classes, 64×64 images, 500 training images per class).

## Baseline
- **Validation Accuracy:** 87.52%

## Task
Train on 100,000 images (500 per class), evaluate on 10,000 validation images.

## Scoring
```
Score = (your_accuracy - 87.52) / 87.52 × 100
```

**Examples:**
- 89.0% → 1.69% improvement
- 88.5% → 1.12% improvement

## Dataset: Tiny ImageNet
- **Source:** http://cs231n.stanford.edu/tiny-imagenet-200.zip
- **200 classes** (subset of ImageNet)
- **100,000 train images** (500 per class)
- **10,000 val images** (50 per class)
- **64×64 RGB images** (vs 224×224 in full ImageNet)

## MANO-tiny Architecture
- Lightweight Multipole Attention Neural Operator
- Linear O(n) complexity attention
- Optimized for 64×64 images

## Improvement Strategies
1. **Architecture:** Adjust for low resolution
2. **Augmentation:** RandAugment, MixUp, CutMix
3. **Training:** Longer epochs, better LR schedule
4. **Regularization:** Dropout, label smoothing
5. **Ensemble:** Multiple MANO variants

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_tiny_imagenet_classification_mano_tiny_main
```

## Training Time

| Training Time |
|---------------|
| 15 hours 56 minutes |

## Solution Interface
```python
def evaluate() -> Dict[str, float]:
    return {'accuracy': float}  # 0-100
```

Good luck!

