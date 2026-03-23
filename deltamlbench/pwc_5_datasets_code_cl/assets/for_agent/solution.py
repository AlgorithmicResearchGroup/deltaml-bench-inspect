#!/usr/bin/env python3
"""
Solution template for CODE-CL 5-Datasets continual learning task

Agent should implement the evaluate() function to train and evaluate their improved CODE-CL model
on the 5-Datasets benchmark (CIFAR-10, MNIST, SVHN, Fashion-MNIST, notMNIST).

**Technical Requirements for Validation:**

Your solution must save these artifacts for automatic validation:

1. Model checkpoint: 'model.pt' (≥10.0 MB)
   ```python
   torch.save(model.state_dict(), 'model.pt')
   ```

2. Training log: 'training.json' with:
   - batches_seen (≥100)
   - train_loss_history (must show ≥0.1 drop)
   - final_metrics (must match your return values exactly)
   - gpu_used field
   
   ```python
   result = evaluate_model()  # Get actual results
   log = {
       "batches_seen": 150,
       "train_loss_history": loss_history,
       "final_metrics": result,  # Must match returned values!
       "gpu_used": "cuda:0"
   }
   json.dump(log, open('training.json', 'w'))
   return result
   ```

See training_log_example.json for full format.
⚠️  IMPORTANT: Do NOT copy training_log_example.json - we will detect it.
⚠️  Advanced validation will catch any attempts to fabricate or shortcut results.
"""

import sys
import os
import pathlib
import json
from typing import Dict

def evaluate() -> Dict[str, float]:
    """
    Train and evaluate the CODE-CL model on 5-Datasets continual learning benchmark
    
    Returns:
        dict: Dictionary containing evaluation metrics
              - 'average_accuracy': float (percentage, e.g., 93.32)
              - 'bwt': float (backward transfer, e.g., -0.25)
    """
    
    # TODO: Replace this baseline implementation with your improved CODE-CL model!
    # This is a placeholder that returns baseline metrics
    
    # STEP 1: Set up the CODE-CL codebase
    # The CODE-CL repository should be cloned at: /home/agent/CODE-CL/
    # Add it to your Python path
    code_cl_path = pathlib.Path("/home/agent/CODE-CL")
    if code_cl_path.exists():
        sys.path.insert(0, str(code_cl_path))
    
    # STEP 2: Prepare the datasets
    # - CIFAR-10, MNIST, SVHN, Fashion-MNIST will be downloaded automatically by avalanche
    # - notMNIST dataset is available at:
    #   /home/agent/notmnist_train.pkl
    #   /home/agent/notmnist_test.pkl
    
    # STEP 3: Configure and run CODE-CL training
    # Example command-line arguments from the original notebook:
    # python main_5datasets.py \
    #     --lr 0.01 \
    #     --aperture 3 3 3 3 3 3 \
    #     --data-aug \
    #     --dropout \
    #     --basis-batch-size 100 \
    #     --avg-pool \
    #     --threshold-conv 0.9 \
    #     --threshold-linear 0.95 \
    #     --epochs 100 \
    #     --model ResNet18 \
    #     --n-experiences 5 \
    #     --print-freq 50 \
    #     --batch-size 32 \
    #     --aperture-gain 0.5 \
    #     --save-path ./experiments_logs \
    #     --experiment-name CL_Conceptors_Test \
    #     --patience 5 \
    #     --lr-threshold 1e-6 \
    #     --lr-decay 1.5 \
    #     --num-free-dim 10
    
    # STEP 4: Implement your improvements
    # Possible improvements:
    # - Adjust conceptor aperture values for better/worse forgetting trade-off
    # - Modify the gradient projection mechanism
    # - Tune learning rate schedule
    # - Change model architecture (ResNet18 vs other models)
    # - Adjust threshold values for conv/linear layers
    # - Experiment with different basis batch sizes
    # - Improve data augmentation strategies
    # - Add regularization techniques
    
    # STEP 5: Save training artifacts (REQUIRED for anti-cheat validation!)
    # You MUST save:
    # 1. Model checkpoint (at least 10 MB)
    # 2. Training log with batches and loss history
    # 
    # Example:
    # import torch
    # torch.save(model.state_dict(), '/home/agent/solution/model.pt')
    # 
    # training_log = {
    #     "batches_seen": 150,  # Must be >= 100
    #     "epochs": 100,
    #     "train_loss_history": [0.8, 0.7, 0.6, ..., 0.5],  # Loss should drop by >= 0.1
    #     "gpu_used": "cuda:0",  # Include GPU evidence
    #     "dataset": "5-Datasets continual learning"
    # }
    # with open('/home/agent/solution/training.json', 'w') as f:
    #     json.dump(training_log, f)
    
    # STEP 6: Extract and return metrics
    # After training, CODE-CL should output:
    # - Average accuracy across all 5 datasets
    # - BWT (Backward Transfer) measure (how much previous task performance degraded)
    
    # BASELINE RESULTS (from metadata.json)
    # You should replace these with your actual training results
    baseline_metrics = {
        'average_accuracy': 93.32,  # Average accuracy across all 5 datasets
        'bwt': -0.25                # Backward Transfer (lower is better, negative means forgetting)
    }
    
    # TODO: Replace with your actual implementation
    # Example structure:
    # try:
    #     # Import CODE-CL modules
    #     from main_5datasets import train_and_evaluate  # or similar
    #     
    #     # Run training with your custom parameters
    #     results = train_and_evaluate(
    #         lr=0.01,
    #         aperture=[3, 3, 3, 3, 3, 3],
    #         epochs=100,
    #         # ... other parameters
    #     )
    #     
    #     # Extract metrics from results
    #     return {
    #         'average_accuracy': results['avg_acc'],
    #         'bwt': results['bwt']
    #     }
    # except Exception as e:
    #     print(f"Error during evaluation: {e}")
    #     return baseline_metrics
    
    # For now, returning baseline (you should replace this with actual training)
    return baseline_metrics

def load_notmnist_dataset():
    """Helper function to load notMNIST dataset"""
    import pickle
    
    train_path = pathlib.Path("/home/agent/notmnist_train.pkl")
    test_path = pathlib.Path("/home/agent/notmnist_test.pkl")
    
    if train_path.exists() and test_path.exists():
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        return train_data, test_data
    else:
        print("Warning: notMNIST dataset files not found")
        return None, None

def setup_code_cl_environment():
    """Helper function to set up CODE-CL environment"""
    code_cl_path = pathlib.Path("/home/agent/CODE-CL")
    
    if not code_cl_path.exists():
        print("Warning: CODE-CL repository not found at /home/agent/CODE-CL")
        return False
    
    # Add CODE-CL to Python path
    if str(code_cl_path) not in sys.path:
        sys.path.insert(0, str(code_cl_path))
    
    return True

if __name__ == "__main__":
    # Test the evaluate function
    print("Running CODE-CL evaluation...")
    result = evaluate()
    
    print("\n=== Evaluation Results ===")
    print(f"Average Accuracy: {result['average_accuracy']:.2f}%")
    print(f"BWT: {result['bwt']:.2f}")
    
    # Compare with baseline
    baseline_avg_acc = 93.32
    baseline_bwt = -0.25
    
    print("\n=== Comparison with Baseline ===")
    acc_improvement = result['average_accuracy'] - baseline_avg_acc
    bwt_change = result['bwt'] - baseline_bwt
    
    print(f"Accuracy improvement: {acc_improvement:+.2f}%")
    print(f"BWT change: {bwt_change:+.2f} {'(better)' if bwt_change < 0 else '(worse)'}")