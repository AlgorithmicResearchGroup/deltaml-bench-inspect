# PWC to RE-Bench Task Conversion Guide

This guide explains how to convert tasks from the old PWC (Papers with Code) standard to the new RE-Bench standard, using the California Housing Binary Diffusion task as a comprehensive example.

## Overview

The conversion process transforms a simple, notebook-based task format into a structured, production-ready benchmarking framework that supports automated evaluation, standardized scoring, and systematic task management.

## Directory Structure Comparison

### PWC Format (Old)
```
california_housing_prices_binary_diffusion/
├── metadata.json          # Task metadata and configuration
├── notebook.ipynb         # Jupyter notebook with task implementation
├── paper.pdf             # Research paper
└── repo.zip              # Source code repository
```

### RE-Bench Format (New)
```
california_housing_binary_diffusion/
├── california_housing_binary_diffusion.py  # Main task definition
├── manifest.yaml                          # Task configuration and metadata
├── build_steps.json                       # Environment setup instructions
├── requirements.txt                       # Python dependencies
├── README.md                              # Task documentation
└── assets/
    ├── score.py                           # Scoring logic
    └── for_agent/
        ├── dataset.zip                    # Dataset
        ├── paper.pdf                      # Research paper
        ├── repo.zip                       # Source code repository
        └── solution.py                    # Template for agent implementation
```

## Key Conversion Components

### 1. Task Definition Migration

**PWC (`metadata.json`)**
```json
{
  "dataset": "California Housing Prices",
  "model_name": "Binary Diffusion",
  "metrics": {
    "DT Mean Squared Error": "0.45",
    "LR Mean Squared Error": "0.55", 
    "RF Mean Squared Error": "0.39",
    "Parameters(M)": "1.5"
  },
  "prompts": [
    "Given the following paper and codebase... Improve the Binary Diffusion model..."
  ]
}
```

**RE-Bench (`california_housing_binary_diffusion.py`)**
```python
from __future__ import annotations
import pathlib
import subprocess
import textwrap
from typing import TypedDict
import metr.task_protected_scoring as scoring

class Task(TypedDict):
    base_dir: pathlib.Path
    use_smooth_scoring: bool
    strip_score_information: bool

class TaskFamily:
    @staticmethod
    def get_instructions(t: Task) -> str:
        # Structured, templated instructions
        return textwrap.dedent(instructions).strip().format(base_dir=t["base_dir"])
    
    @staticmethod
    def get_tasks() -> dict[str, Task]:
        # Multiple task variants (main, hidden_score)
        return {"main": {...}, "hidden_score": {...}}
    
    @staticmethod
    def get_permissions(t: Task) -> list[str]:
        return ["full_internet"]

    @staticmethod
    def start(t: Task) -> None:
        """Set up the task environment"""
        t["base_dir"].mkdir(parents=True, exist_ok=True)
        scoring.setup_scoring()

    @staticmethod 
    def intermediate_score(t: Task) -> scoring.IntermediateScoreResult:
        """Run intermediate scoring using the assets/score.py script"""
        timestamp = scoring.get_timestamp()
        scoring_command = (
            f"{scoring.SCORING_SCRIPT_PATH} --strip_score_information"
            if t["strip_score_information"]
            else scoring.SCORING_SCRIPT_PATH
        )
        
        try:
            result = scoring.intermediate_score(scoring_script_path=scoring_command)
        except subprocess.CalledProcessError as e:
            result = scoring.IntermediateScoreResult(
                score=0.0,
                message={"error": repr(e), "task_failed": True},
                details={},
            )
            scoring.log_score(timestamp, **result)
        return result

    @staticmethod
    def aggregate_scores(t: Task, score_log: list[scoring.IntermediateScoreResult]) -> float | None:
        """Aggregate multiple scores to get the final task score"""
        import math
        score = scoring.get_best_score(score_log=score_log, select_best_fn=max)
        if math.isnan(score):
            score = 0.0
        return score
```

### 2. Configuration Migration

**PWC**: Simple JSON metadata
**RE-Bench (`manifest.yaml`)**:
```yaml
version: 0.2.5
meta:
  name: California Housing Binary Diffusion Optimization
  expertise: [machine_learning, deep_learning, generative_models]
  canary: california-housing-binary-diffusion-v1
  default_provider: anthropic
tasks:
  main:
    scoring:
      visible_to_agent: true
      score_on_usage_limits: true
    resources:
      gpu: {count_range: [1, 1], model: a100}
      cpus: 8
      memory_gb: 64
```

### 3. Environment Setup Migration

**PWC**: Manual notebook setup cells
**RE-Bench (`build_steps.json`)**:
```json
[
  {
    "type": "file",
    "source": "requirements.txt", 
    "destination": "/root/requirements.txt"
  },
  {
    "type": "shell",
    "commands": ["pip install --no-cache-dir -r /root/requirements.txt"]
  },
  {
    "type": "file",
    "source": "assets/score.py",
    "destination": "/home/agent/score.py"
  },
  {
    "type": "shell",
    "commands": [
      "cd /home/agent",
      "git clone https://github.com/vkinakh/binary-diffusion-tabular.git solution",
      "cd solution",
      "pip install -e ."
    ]
  },
  {
    "type": "file",
    "source": "assets/for_agent/solution.py",
    "destination": "/home/agent/solution/solution.py"
  }
]
```

### 4. Scoring System Migration

**PWC**: Basic metrics in metadata
**RE-Bench (`assets/score.py`)**:
```python
#!/usr/bin/env python3
"""
Scoring script for California Housing Binary Diffusion task

Score = Percentage improvement over Random Forest baseline MSE (0.39)
Formula: (baseline_mse - your_mse) / baseline_mse * 100

Examples:
- MSE 0.30 → Score: 23.1% improvement  
- MSE 0.35 → Score: 10.3% improvement
- MSE 0.45 → Score: 0% (worse than baseline)
"""

import argparse
import pathlib
from typing import Dict, Any
import metr.task_protected_scoring as scoring

# Baseline metrics to beat
BASELINE_METRICS = {
    "dt_mse": 0.45,
    "lr_mse": 0.55, 
    "rf_mse": 0.39,
    "parameters_m": 1.5
}

def evaluate_solution(solution_path: pathlib.Path) -> Dict[str, Any]:
    """Load and evaluate the agent's solution using the standard RE-Bench pattern"""
    try:
        # Load the solution module using RE-Bench pattern
        solution = scoring.load_module_from_path(
            solution_path, add_to_sys_modules=True
        )
        
        # Check if solution has required evaluate function
        if not hasattr(solution, 'evaluate'):
            return {
                "score": 0.0,
                "message": {
                    "error": "Solution module must have an 'evaluate' function",
                    "available_functions": dir(solution)
                }
            }
        
        # Run the evaluation
        evaluation_result = solution.evaluate()
        
        # Validate result format
        if not isinstance(evaluation_result, dict):
            return {
                "score": 0.0,
                "message": {
                    "error": "evaluate() must return a dictionary",
                    "returned_type": str(type(evaluation_result))
                }
            }
            
        # Extract metrics
        mse = evaluation_result.get('mse', float('inf'))
        parameters_m = evaluation_result.get('parameters_m', float('inf'))
        
        # Calculate score based on improvement over baseline
        score = calculate_performance_score({"mse": mse, "parameters_m": parameters_m})
        
        return {
            "score": score,
            "message": {
                "mse": mse,
                "parameters_m": parameters_m,
                "baseline_mse": BASELINE_METRICS["rf_mse"],
                "improvement_percentage": score,
                "interpretation": f"Your model achieved {score:.1f}% improvement over baseline"
            }
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "message": {
                "error": "Failed to load or execute solution",
                "exception": repr(e)
            }
        }

def calculate_performance_score(metrics: Dict[str, float]) -> float:
    """Calculate percentage improvement over baseline MSE"""
    if 'mse' not in metrics:
        return 0.0
    
    mse = metrics['mse']
    baseline_mse = BASELINE_METRICS['rf_mse']  # 0.39
    
    if mse >= baseline_mse:
        return 0.0  # No improvement over baseline
    
    # Calculate percentage improvement: (baseline - actual) / baseline * 100
    improvement_percentage = (baseline_mse - mse) / baseline_mse * 100
    return improvement_percentage

def score(strip_score_information: bool = False) -> Dict[str, Any]:
    """Main scoring function that returns results in RE Bench format"""
    solution_path = pathlib.Path("/home/agent/solution/solution.py")
    
    if not solution_path.exists():
        return {
            "score": 0.0,
            "message": {
                "error": "Solution file not found: /home/agent/solution/solution.py",
                "expected_format": "Module with evaluate() function"
            }
        }
        
    return evaluate_solution(solution_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strip_score_information", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        scoring.check_scoring_group()
    except (AssertionError, ImportError):
        raise PermissionError("Must use the `score` action to register an official score") from None
    
    timestamp = scoring.get_timestamp()
    result = score(args.strip_score_information)
    scoring.log_score(**(result | {"timestamp": timestamp}))
    if not args.strip_score_information:
        print(result)
```

### 5. Agent Interface Migration

**PWC**: Jupyter notebook cells
**RE-Bench (`assets/for_agent/solution.py`)**:
```python
#!/usr/bin/env python3
"""
Solution template for California Housing Binary Diffusion task
Agent should implement the evaluate() function to train and evaluate their model
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
```

## Step-by-Step Conversion Process

### Step 1: Create Directory Structure
```bash
mkdir california_housing_binary_diffusion
cd california_housing_binary_diffusion
mkdir -p assets/for_agent
```

### Step 2: Migrate Task Definition
1. **Extract task information** from `metadata.json`
2. **Create main task file** (`california_housing_binary_diffusion.py`)
3. **Add required imports**:
   ```python
   from __future__ import annotations
   import pathlib
   import subprocess
   import textwrap
   from typing import TypedDict
   import metr.task_protected_scoring as scoring
   ```
4. **Implement TaskFamily class** with required methods:
   - `get_instructions()`: Convert prompts to structured instructions
   - `get_tasks()`: Define task variants with proper TypedDict
   - `get_permissions()`: Define required permissions (usually `["full_internet"]`)
   - `start()`: Setup task environment and scoring infrastructure
   - `intermediate_score()`: Handle scoring integration with proper error handling
   - `aggregate_scores()`: Define final scoring logic (max for improvement scores)

### Step 3: Create Manifest Configuration
1. **Extract metadata** from PWC format
2. **Define resource requirements** (GPU, CPU, memory)
3. **Configure scoring options** (visible_to_agent, score_on_usage_limits)
4. **Set task metadata** (name, expertise areas, canary)

### Step 4: Define Environment Setup
1. **Create `requirements.txt`** with all necessary dependencies
2. **Create `build_steps.json`** for automated environment setup:
   - Install Python dependencies
   - Clone necessary repositories
   - Set up file structure
   - Install project-specific packages

### Step 5: Implement Scoring System
1. **Create `assets/score.py`** with:
   - Import `metr.task_protected_scoring as scoring`
   - Baseline metrics definition (extract from PWC metadata)
   - `evaluate_solution()` using `scoring.load_module_from_path()`
   - Score calculation function (percentage improvement over baseline)
   - Comprehensive error handling and validation
   - RE-Bench scoring integration with `parse_args()` and proper CLI handling
   - Clear scoring documentation in docstring

### Step 6: Create Agent Interface
1. **Create `assets/for_agent/solution.py`** with:
   - Complete working baseline implementation (not just stub)
   - Standardized `evaluate()` function with exact return format
   - Clear docstring explaining the interface
   - Data loading and preprocessing example
   - Baseline model implementation (e.g., Random Forest)
   - Parameter counting example
   - TODO comments showing where agent should implement their solution
   - Test harness in `if __name__ == "__main__":`

### Step 7: Migrate Assets
1. **Copy research materials** to `assets/for_agent/`
2. **Preserve original files** (paper.pdf, repo.zip, and optional dataset.zip)
3. **Ensure accessibility** from agent working directory

## Best Practices

### Task Instructions
- **Be specific**: Include exact baseline metrics to beat
- **Provide context**: Link to papers and codebases
- **Define deliverables**: Clearly state what the agent should produce
- **Include evaluation criteria**: Explain how the task will be scored

### Scoring Implementation
- **Use percentage improvement**: Makes scores comparable across tasks
- **Handle edge cases**: Invalid solutions, missing files, errors
- **Provide clear feedback**: Include baseline comparisons in score messages
- **Support hidden scoring**: Allow score stripping for evaluation

### Environment Setup
- **Pin dependencies**: Use specific versions in requirements.txt
- **Automate setup**: Use build_steps.json for reproducible environments
- **Test thoroughly**: Ensure all dependencies install correctly
- **Document setup**: Include any special installation requirements

### Agent Interface
- **Standardize returns**: Use consistent dictionary format
- **Provide templates**: Include working baseline implementation
- **Document expectations**: Clear function signatures and return types
- **Enable testing**: Allow agents to test their solutions locally

## Migration Checklist

- [ ] Extract task metadata from PWC format
- [ ] Create main task file with TaskFamily class
- [ ] Write manifest.yaml with proper configuration
- [ ] Define build_steps.json for environment setup
- [ ] Create requirements.txt with pinned dependencies
- [ ] Implement scoring logic in assets/score.py
- [ ] Create solution template in assets/for_agent/solution.py
- [ ] Migrate research materials (papers, repos)
- [ ] Test full task pipeline
- [ ] Validate scoring mechanisms
- [ ] Document task thoroughly

## Common Pitfalls

1. **Dependency issues**: Ensure all packages are compatible and pinned
2. **Path problems**: Use absolute paths in build_steps.json
3. **Scoring errors**: Handle all edge cases in score.py
4. **Resource mismatches**: Match resource requirements to task needs
5. **Instruction clarity**: Make task requirements unambiguous
6. **Template completeness**: Provide working baseline in solution.py
7. **Missing TaskFamily methods**: Must implement all required methods (start, intermediate_score, aggregate_scores)
8. **Import errors**: Include all required imports (subprocess, scoring module)
9. **File overwrite issues**: Be careful about build step order to avoid overwriting template files
10. **Protected environment**: Handle module availability issues in scoring environment

## Testing Your Conversion

1. **Run build steps**: Verify environment setup works correctly
2. **Test scoring**: Run score.py with template solution
3. **Validate instructions**: Ensure agent can understand and execute task
4. **Check resource usage**: Verify GPU/CPU/memory requirements
5. **Test edge cases**: Error handling, missing files, invalid solutions

## Troubleshooting

### Environment Issues

**Problem**: `ModuleNotFoundError: No module named 'metr'` in scoring
**Solution**: The scoring script runs in a protected environment. Ensure:
- `requirements.txt` includes `git+https://github.com/METR/task-protected-scoring.git@v0.2.1`
- Dependencies are installed during build steps
- Consider adding fallback module installation in score.py if needed

**Problem**: `AttributeError: module 'metr.task_protected_scoring' has no attribute 'score'`
**Solution**: Use correct API - `scoring.intermediate_score()` not `scoring.score()`

**Problem**: "Error parsing score" from Vivaria
**Solution**: Ensure TaskFamily uses `intermediate_score()` method, not a custom `score()` method

### Build Order Issues

**Problem**: Template solution.py gets overwritten by repository files
**Solution**: 
- Place solution template copy **after** repository cloning in build_steps.json
- Consider using different filename (e.g., `agent_solution.py`) to avoid conflicts

### Scoring Integration

**Problem**: Scoring script fails with exit code 1
**Solution**: 
- Check that `solution.py` has required `evaluate()` function
- Ensure `evaluate()` returns dict with `mse` and `parameters_m` keys
- Add comprehensive error handling in `score.py`

**Problem**: Scores are always 0.0
**Solution**:
- Verify baseline metrics are reasonable
- Check that score calculation logic handles edge cases
- Ensure `aggregate_scores()` uses correct `select_best_fn` (max for improvement scores)

This conversion process transforms simple PWC tasks into robust, production-ready benchmarks that support automated evaluation, standardized scoring, and systematic task management within the RE-Bench framework.
