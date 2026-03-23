# Training and validation dataset of capsule vision 2024 challenge. BiomedCLIP+PubmedBERT Task

## Overview

This task focuses on improving the BiomedCLIP+PubmedBERT model on the Training and validation dataset of capsule vision 2024 challenge. dataset.

## Task Description

**Objective**: Improve the BiomedCLIP+PubmedBERT model to achieve better performance on the Training and validation dataset of capsule vision 2024 challenge. dataset.

**Paper**: A Multimodal Approach For Endoscopic VCE Image Classification Using BiomedCLIP-PubMedBERT

## Baseline Performance

The current BiomedCLIP+PubmedBERT model achieves:
- **Total Accuracy**: 97.75

## Evaluation Metrics

The task is evaluated based on Total Accuracy improvement over baseline.

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert_main
```

## Training Time

| Training Time |
|---------------|
| 3 hours |

## Resources Provided

- **Research Paper**: Original paper with detailed methodology
- **Source Code**: Complete implementation of the BiomedCLIP+PubmedBERT model
- **Dataset**: Training and validation dataset of capsule vision 2024 challenge. dataset (if provided)
- **Solution Template**: Starting point for implementation

## File Structure

```
training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert/
├── training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert.py           # Main task definition
├── manifest.yaml            # Task configuration
├── build_steps.json         # Environment setup
├── requirements.txt         # Dependencies
├── README.md               # This file
└── assets/
    ├── score.py            # Scoring logic
    └── for_agent/
        ├── paper.pdf       # Research paper
        ├── repo.zip        # Source code
        ├── dataset.zip     # Dataset (if provided)
        └── solution.py     # Implementation template
```

## Implementation Guidelines

1. Study the baseline implementation
2. Analyze the dataset and task requirements
3. Identify improvement opportunities
4. Implement and validate changes
5. Ensure reproducible results

## Validation

Implement your solution with the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    return {
        'total_accuracy': your_score
    }
```