#!/usr/bin/env python3
"""
Solution template for California Housing Binary Diffusion task
Agent should implement the evaluate() function to train and evaluate their model


**Technical Requirements for Validation:**

Your solution must save these artifacts for automatic validation:

1. Model checkpoint: 'model.pt' (≥5.0 MB)
   ```python
   torch.save(model.state_dict(), 'model.pt')
   ```

2. Training log: 'training.json' with:
   - batches_seen (≥50)
   - train_loss_history (must show ≥0.05 drop)
   - final_metrics (must match your return values exactly)
   - gpu_used field
   
   ```python
   result = evaluate_model()  # Get actual results
   log = {
       "batches_seen": 100,
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

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

def evaluate():
    """
    Train and evaluate a model on the California Housing dataset
    
    Returns:
        dict: Dictionary containing evaluation metrics
              - 'mse': Mean Squared Error on test set
              - 'parameters_m': Model parameters in millions
    """
    # Load California Housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # BASELINE MODEL - Replace this with your Binary Diffusion model!
    # This is just a simple Random Forest baseline to demonstrate the interface
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    
    # Estimate parameters for Random Forest (rough approximation)
    # For your Binary Diffusion model, count actual model parameters
    n_trees = model.n_estimators
    avg_nodes_per_tree = np.mean([tree.tree_.node_count for tree in model.estimators_])
    estimated_params = n_trees * avg_nodes_per_tree * X.shape[1]
    parameters_m = estimated_params / 1e6  # Convert to millions
    
    # TODO: Replace the above with your Binary Diffusion model implementation
    # Example structure:
    # 1. Load/create your Binary Diffusion model
    # 2. Train it on X_train_scaled, y_train
    # 3. Evaluate on X_test_scaled, y_test
    # 4. Return the MSE and parameter count
    
    return {
        'mse': float(mse),
        'parameters_m': float(parameters_m)
    }

if __name__ == "__main__":
    # Test the evaluate function
    result = evaluate()
    print(f"MSE: {result['mse']:.4f}")
    print(f"Parameters: {result['parameters_m']:.2f}M")
