# ANDHRA Bandersnatch: Training Neural Networks to Predict Parallel Realities on CIFAR-10

## Overview

This task focuses on improving the ANDHRA Bandersnatch (ABNet) model with 2 Groups (AB-2GR0) on the CIFAR-10 image classification task.

**Note**: The directory name `pwc_cifar_10_abnet_2g_r0` refers to **ANDHRA Bandersnatch with 2 Groups, Run 0** on CIFAR-10. The original metadata incorrectly listed this as a "CMA-R" task, but the actual content is the ANDHRA Bandersnatch model.

## Task Description

**Objective**: Improve the ANDHRA Bandersnatch (AB-2GR0) model to achieve better test accuracy on the CIFAR-10 dataset.

**Repository**: https://github.com/dvssajay/New_World.git

**Model**: ANDHRA Bandersnatch (AB-2GR0) - A novel neural network architecture that uses ANDHRA activation (dual ReLU heads) to train networks that can "predict parallel realities".

## Baseline Performance

The current ANDHRA Bandersnatch (AB-2GR0) model achieves the following baseline performance on CIFAR-10:

- **Test Accuracy**: 94.118%

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline test accuracy:

**Scoring**: 
```
Score = (new_accuracy - 94.118) / 94.118 * 100
```
Example: Test Accuracy = 95.5% → Score = (95.5 - 94.118) / 94.118 * 100 = 1.47%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_cifar_10_abnet_2g_r0_main
```

## Training Time

| Training Time |
|---------------|
| 4 hours 23 minutes |

## Resources Provided

- **Research Paper**: ANDHRA Bandersnatch paper with detailed methodology
- **Source Code**: Complete implementation of the ANDHRA Bandersnatch model
- **Dataset**: CIFAR-10 (downloaded automatically, 50k train + 10k test images)
- **Solution Template**: Starting point for implementation
- **Dataset Note**: The provided `dataset.zip` is empty (22 bytes). The training scripts download CIFAR-10 automatically.

## Technical Requirements

- **GPU**: 1x (recommended for faster training)
- **Memory**: 16GB RAM
- **Framework**: PyTorch
- **Key Dependencies**: torch, torchvision, numpy, wandb (optional for logging)

## File Structure

```
pwc_cifar_10_abnet_2g_r0/
├── pwc_cifar_10_abnet_2g_r0.py  # Main task definition
├── manifest.yaml                 # Task configuration
├── build_steps.json              # Environment setup
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── assets/
    ├── score.py                  # Scoring logic
    └── for_agent/
        ├── paper.pdf             # Research paper
        ├── repo.zip              # ANDHRA Bandersnatch source code
        └── solution.py           # Implementation template
```

## Implementation Guidelines

1. **Study the baseline**: Understand the ANDHRA Bandersnatch architecture:
   - **ANDHRA Activation**: Uses dual ReLU heads to create parallel prediction paths
   - **AB-2GR0**: ANDHRA Bandersnatch with 2 Groups, Run 0
   - **ResBlocks**: Uses custom ResBlock architecture with parallel processing
   - **Training**: 200 epochs, SGD optimizer, cosine annealing scheduler

2. **Analyze the task**: 
   - CIFAR-10: 10 classes, 32x32 RGB images
   - Review `mainAB2GR0_10_1.py` for training pipeline
   - Review `models/ab_2GR0_10.py` for model architecture
   - Review `test_10_BS.py` for evaluation approach

3. **Identify improvements**: Consider:
   - Modifying the ANDHRA activation function
   - Adjusting the number of groups/heads
   - Hyperparameter tuning (learning rate, batch size, weight decay)
   - Data augmentation strategies
   - Training schedule modifications
   - Architecture changes (channels, layers, etc.)
   - Ensemble methods

4. **Implement changes**: Modify the model while maintaining the evaluation interface

5. **Validate results**: Ensure improvements generalize to the test set

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'test_accuracy': your_test_accuracy  # e.g., 95.5 for 95.5%
    }
```

## Key Concepts

- **ANDHRA Activation**: A novel activation approach using dual ReLU heads that allows the network to predict "parallel realities" during training
- **Parallel Realities**: The concept that the network learns multiple possible representations simultaneously
- **Multi-Head Prediction**: The model uses multiple prediction heads that can be ensembled for better performance
- **AB-2GR0 Notation**: ANDHRA Bandersnatch with 2 Groups, Run 0

## Papers With Code

This model holds state-of-the-art positions on:
- CIFAR-10 Image Classification
- CIFAR-100 Image Classification

## Training Details

**Baseline Training Configuration (from `mainAB2GR0_10_1.py`)**:
- Epochs: 200
- Batch Size: 128
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Scheduler: CosineAnnealingLR
- Initial Learning Rate: 0.1
- Augmentation: Random crop, random horizontal flip
- Normalization: CIFAR-10 mean/std

## References

- Repository: [New_World on GitHub](https://github.com/dvssajay/New_World.git)
- Papers With Code: [ANDHRA Bandersnatch on CIFAR-10](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=andhra-bandersnatch-training-neural-networks)
