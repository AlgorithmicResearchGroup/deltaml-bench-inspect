# YOLO-SAM2: Medical Polyp Segmentation on Kvasir-SEG

## Overview

This task focuses on improving the YOLO-SAM2 model for medical polyp detection and segmentation on the Kvasir-SEG dataset.

## Task Description

**Objective**: Improve the YOLO-SAM2 model to achieve better performance on the Kvasir-SEG dataset for medical polyp detection and segmentation.

**Codebase**: https://github.com/sajjad-sh33/YOLO_SAM2.git

## Baseline Performance

The current YOLO-SAM2 model achieves:
- **F1 Score**: 90.7
- **Mean Dice**: 0.866
- **mIoU**: 0.764

## Evaluation Metrics

The task is evaluated based on average percentage improvement across all 3 metrics:
- **F1 Score**: Harmonic mean of precision and recall (baseline: 90.7)
- **Mean Dice**: Overlap measure for medical segmentation (baseline: 0.866)
- **mIoU**: Mean Intersection over Union for segmentation quality (baseline: 0.764)

### Scoring Formula

Your solution is scored based on the **average percentage improvement across all 3 metrics**:

```
Score = Average of [(new_metric - baseline_metric) / baseline_metric × 100] for each metric
```

**Example Calculation**:
If you achieve: F1=92.0, Dice=0.88, mIoU=0.78
- F1 improvement: (92.0 - 90.7) / 90.7 × 100 = 1.43%
- Dice improvement: (0.88 - 0.866) / 0.866 × 100 = 1.62%
- mIoU improvement: (0.78 - 0.764) / 0.764 × 100 = 2.09%
- **Average Score**: (1.43 + 1.62 + 2.09) / 3 = **1.71%**

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_kvasir_seg_yolo_sam_2_main
```

## Training Time

| Training Time |
|---------------|
| 33 minutes |

## Resources Provided

- **Source Code**: Complete YOLO-SAM2 implementation from GitHub repository
- **Dataset**: Kvasir-SEG dataset with polyp images for medical segmentation
- **Solution Template**: Starting point for implementation

## File Structure

```
kvasir_seg_yolo_sam_2/
├── kvasir_seg_yolo_sam_2.py           # Main task definition
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
    """
    Train and evaluate your improved YOLO-SAM2 model.
    
    Returns:
        dict: Dictionary containing:
            - 'f1_score': float (e.g., 92.0) - Required
            - 'mean_dice': float (e.g., 0.88) - Required
            - 'miou': float (e.g., 0.78) - Required
    """
    # Your implementation here
    pass
```