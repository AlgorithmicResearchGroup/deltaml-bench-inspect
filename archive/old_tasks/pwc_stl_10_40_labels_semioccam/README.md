# STL-10 Semi-Supervised Learning with SemiOccam

## Overview
This task challenges agents to improve the SemiOccam model for semi-supervised image classification on STL-10 **with only 40 labeled examples**. This is an extreme few-shot semi-supervised learning scenario where leveraging unlabeled data is crucial.

## Task Description
**Paper:** [SemiOccam: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels](https://arxiv.org/abs/2506.03582v1)

**Repository:** https://github.com/Shu1L0n9/SemiOccam

**Challenge:** Train a model that achieves high accuracy on STL-10 using:
- **Only 40 labeled examples** (4 per class)
- **~5,000 unlabeled examples**

## Baseline Performance
- **Accuracy:** 95.43% (with only 40 labels!)

## Goal
Achieve higher classification accuracy by better leveraging the unlabeled data through semi-supervised learning techniques.

## Evaluation Metric
**Classification Accuracy:** Percentage of correctly classified test images.

## Scoring Formula
```
Score = (your_accuracy - baseline_accuracy) / baseline_accuracy × 100
```

### Examples
- **Accuracy 96.0%** → Score: **0.60%** improvement
- **Accuracy 95.8%** → Score: **0.39%** improvement  
- **Accuracy 95.43%** → Score: **0%** (matches baseline)
- **Accuracy 95.0%** → Score: **0%** (worse than baseline)

## Dataset: CleanSTL-10

### Why CleanSTL-10?
The original STL-10 dataset contains duplicates and label noise. SemiOccam uses a cleaned version available on HuggingFace.

### Dataset Splits
- **train_labeled:** 40 examples (4 per class)
- **train_unlabeled:** ~5,000 examples (no labels)
- **test:** 8,000 examples

### Classes (10 total)
airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck

### Image Properties
- Resolution: 96×96 pixels
- Format: RGB
- Normalized with ImageNet statistics

### Accessing the Dataset
```python
from datasets import load_dataset

# Requires HuggingFace authentication
# Run: huggingface-cli login
ds = load_dataset("Shu1L0n9/CleanSTL-10")

train_labeled = ds['train_labeled']      # 40 examples
train_unlabeled = ds['train_unlabeled']  # ~5,000 examples
test = ds['test']                         # 8,000 examples
```

## SemiOccam Method

### Architecture Components

1. **Vision Transformer (ViT)**
   - Pre-trained backbone for feature extraction
   - Fine-tuned on STL-10
   - Provides powerful representations

2. **Semi-supervised GMM (SGMM)**
   - Models class distributions as Gaussian mixtures
   - Works with both labeled and unlabeled data
   - Assigns soft pseudo-labels

3. **Pseudo-labeling**
   - Generates confident predictions for unlabeled data
   - Uses confidence thresholding
   - Progressively improves labels

4. **Consistency Regularization**
   - Ensures consistent predictions under augmentation
   - Uses strong/weak augmentation pairs
   - Reduces prediction variance

### Why It Works with 40 Labels

**Problem:** Traditional supervised learning fails catastrophically with only 40 examples.

**Solution:** SemiOccam effectively leverages the 5,000 unlabeled examples:
- **SGMM** learns class boundaries from unlabeled data distribution
- **Pseudo-labeling** expands the labeled set iteratively
- **Consistency** prevents overfitting to the tiny labeled set
- **Strong augmentation** provides diversity

## Improvement Strategies

### 1. Better Semi-Supervised Learning
- **FixMatch:** Strong/weak augmentation with confidence thresholding
- **MixMatch:** Mixup + pseudo-labeling + consistency
- **UDA:** Unsupervised Data Augmentation
- **ReMixMatch:** Advanced MixMatch variant
- **SimCLR pretraining:** Self-supervised initialization

### 2. Enhanced SGMM
- **More GMM components:** Better model capacity
- **Adaptive components:** Learn optimal number
- **Covariance structures:** Full vs diagonal
- **Regularization:** Prevent component collapse

### 3. Advanced Augmentation
- **RandAugment:** Automated augmentation policies
- **CutMix/Cutout:** Occlusion robustness
- **MixUp:** Linear interpolation between examples
- **AutoAugment:** Learned augmentation strategies
- **AugMax:** Adversarial augmentation

### 4. Training Strategies
- **EMA teacher:** Exponential moving average model for pseudo-labels
- **Confidence scheduling:** Gradually increase pseudo-label threshold
- **Curriculum learning:** Start with easy unlabeled examples
- **Multi-stage training:** Pretrain → semi-supervised → fine-tune
- **Longer training:** Semi-supervised methods need more epochs

### 5. Architecture Improvements
- **Larger ViT:** ViT-Base vs ViT-Small
- **Better initialization:** Pre-trained on ImageNet-21k
- **Ensemble:** Multiple models with different augmentations
- **Knowledge distillation:** Teacher-student framework

## Files Structure
```
/home/agent/
├── paper.pdf          # SemiOccam research paper
├── solution/          # SemiOccam codebase
│   ├── src/           # Source code
│   ├── scripts/       # Training scripts
│   └── solution.py    # Your implementation
└── dataset/           # Will download from HuggingFace
```

## Getting Started

### 1. HuggingFace Setup
```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (you'll need a HF account and token)
huggingface-cli login

# Test dataset access
python -c "from datasets import load_dataset; ds = load_dataset('Shu1L0n9/CleanSTL-10'); print(ds)"
```

### 2. Explore the Code
- Review SemiOccam architecture in `solution/src/`
- Understand SGMM implementation
- Examine training scripts in `solution/scripts/`

### 3. Implement Your Solution
- Modify `solution/solution.py`
- Implement improved semi-supervised strategy
- Train and evaluate

## Solution Interface
Your `solution.py` must implement:

```python
def evaluate() -> Dict[str, float]:
    """
    Returns:
        {
            'accuracy': float  # Classification accuracy (0-100)
        }
    """
```

## Common Pitfalls

### Data-Related
- **Not using unlabeled data:** The 40 labeled examples alone won't suffice
- **Weak augmentation:** Strong augmentation is critical for semi-supervised learning
- **Ignoring class imbalance:** Ensure balanced sampling from labeled set
- **Data leakage:** Don't use test set for pseudo-labeling!

### Training-Related
- **Overfitting to labeled data:** Must regularize heavily
- **Poor pseudo-label quality:** Use high confidence thresholds (0.95+)
- **Insufficient training:** Semi-supervised needs 500-1000 epochs
- **Unstable training:** Use EMA teacher model for stability

### Model-Related
- **Too simple model:** ViT-Base works better than ViT-Tiny
- **Poor initialization:** Use ImageNet pre-trained weights
- **Ignoring SGMM:** The GMM component is crucial for SemiOccam

## Implementation Tips

### Pseudo-labeling Example
```python
# Generate pseudo-labels for unlabeled data
@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_data, threshold=0.95):
    model.eval()
    pseudo_labels = []
    confidences = []
    
    for images in unlabeled_data:
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        confidence, predictions = probs.max(dim=1)
        
        # Only use high-confidence predictions
        mask = confidence >= threshold
        pseudo_labels.append(predictions[mask])
        confidences.append(confidence[mask])
    
    return pseudo_labels, confidences
```

### Consistency Loss Example
```python
def consistency_loss(model, images, alpha=1.0):
    # Weak augmentation
    weak_aug = weak_augment(images)
    
    # Strong augmentation
    strong_aug = strong_augment(images)
    
    # Predictions should be consistent
    pred_weak = model(weak_aug)
    pred_strong = model(strong_aug)
    
    # MSE or KL divergence between predictions
    loss = F.mse_loss(pred_weak, pred_strong)
    return alpha * loss
```

## Benchmarks

| Method | Accuracy (40 labels) |
|--------|---------------------|
| Supervised Only | ~40-50% |
| Pi-Model | ~70% |
| FixMatch | ~85% |
| **SemiOccam (Baseline)** | **95.43%** |
| **Your Goal** | **>95.43%** |

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_stl_10_40_labels_semioccam_main
```

## Training Time

| Training Time |
|---------------|
| 3 hours |

## Resources
- **Paper:** `/home/agent/paper.pdf`
- **Code:** `/home/agent/solution/`
- **Dataset:** HuggingFace `Shu1L0n9/CleanSTL-10`
- **SGMM Implementation:** https://github.com/Shu1L0n9/Semi-Supervised-GMM

## References
- [SemiOccam Paper (arXiv)](https://arxiv.org/abs/2506.03582)
- [CleanSTL-10 Dataset](https://huggingface.co/datasets/Shu1L0n9/CleanSTL-10)
- [FixMatch Paper](https://arxiv.org/abs/2001.07685)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)

## Challenge
Achieving 95%+ accuracy with only 40 labeled examples is extremely challenging. Focus on:
- **Maximizing unlabeled data utilization**
- **Strong augmentation strategies**
- **High-quality pseudo-labels**
- **Stable training with EMA**

Good luck pushing beyond 95.43% with minimal supervision!

