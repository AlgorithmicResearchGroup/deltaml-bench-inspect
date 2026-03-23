# Digital Twin-Supported Fault Diagnosis with DANN

## Overview
This task challenges agents to improve a Domain Adaptation Neural Network (DANN) for fault diagnosis in manufacturing systems supported by digital twins.

## Task Description
**Paper:** [A domain adaptation neural network for digital twin-supported fault diagnosis](https://arxiv.org/abs/2505.21046v1)

**Challenge:** Bridge the domain gap between digital twin simulations and real-world manufacturing data for accurate fault diagnosis.

## Baseline Performance
- **Accuracy:** 80.22%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann_main
```

## Training Time

| Training Time |
|---------------|
| 3 hours |

## Goal
Achieve higher classification accuracy by improving the DANN model's ability to learn domain-invariant features.

## Evaluation Metric
**Classification Accuracy:** Percentage of correctly classified faults on the target (real-world) domain.

## Scoring Formula
```
Score = (your_accuracy - baseline_accuracy) / baseline_accuracy × 100
```

### Examples
- **Accuracy 85.0%** → Score: **5.96%** improvement
- **Accuracy 82.5%** → Score: **2.84%** improvement
- **Accuracy 80.22%** → Score: **0%** (matches baseline)
- **Accuracy 75.0%** → Score: **0%** (worse than baseline)

## Approach
The DANN model addresses domain shift through:
1. **Feature Extractor:** Learns representations from both source (digital twin) and target (real) domains
2. **Fault Classifier:** Predicts fault types using extracted features
3. **Domain Discriminator:** Adversarially trained to make features domain-invariant
4. **Gradient Reversal:** Ensures features are discriminative for faults but not for domains

## Improvement Strategies
1. **Enhanced Feature Extraction**
   - Better network architectures (ResNet, EfficientNet, etc.)
   - Attention mechanisms for salient fault patterns
   - Multi-scale feature fusion

2. **Improved Domain Adaptation**
   - Advanced adversarial training strategies
   - Multiple domain discriminators
   - Contrastive learning for domain alignment
   - Self-supervised pretraining

3. **Better Training Procedures**
   - Optimized hyperparameters
   - Curriculum learning strategies
   - Data augmentation techniques
   - Regularization methods

4. **Architecture Enhancements**
   - Deeper or wider networks
   - Skip connections and residual blocks
   - Ensemble methods
   - Meta-learning approaches

## Files Structure
```
/home/agent/
├── paper.pdf          # Research paper
├── solution/          # DANN codebase (downloaded from GCS)
│   └── solution.py    # Your implementation
├── dataset/           # Fault diagnosis dataset
└── score.py           # Scoring script
```

## Getting Started
1. Review the domain adaptation paper
2. Understand the DANN architecture and gradient reversal
3. Analyze the digital twin vs. real data characteristics
4. Implement improvements to reduce domain gap
5. Test with the provided `evaluate()` function

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
- Overfitting to source (digital twin) domain
- Inadequate domain adaptation leading to poor target performance
- Mode collapse in adversarial training
- Imbalanced fault class distribution
- Insufficient regularization

## Resources
- **Paper:** `/home/agent/paper.pdf`
- **Code:** `/home/agent/solution/`
- **Dataset:** `/home/agent/dataset/`

## References
- [Domain Adaptation Neural Networks](https://arxiv.org/abs/1505.07818)
- [Gradient Reversal Layer](https://jmlr.org/papers/volume17/15-239/15-239.pdf)

Good luck!

