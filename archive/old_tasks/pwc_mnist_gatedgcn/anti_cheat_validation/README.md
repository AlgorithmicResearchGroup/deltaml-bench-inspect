# Anti-Cheat Validation Module

## Overview

This is an independent anti-cheat validation module for PWC tasks. It prevents agents from submitting fabricated results and ensures legitimate training occurred.


## Contents

### `anti_cheat_validation.py`

Universal anti-cheat validation module that prevents agents from submitting fabricated results.

**Features:**
- Rule-based code validation (hardcoded values, suspicious patterns)
- Training artifact verification (checkpoints, logs, GPU usage)
- LLM-based semantic analysis (optional)
- Configurable thresholds per task

**Documentation:** See `ANTI_CHEAT_GUIDE_UNIVERSAL.md`

## Usage

### Basic Import

```python
import sys
import pathlib

# Add root directory to path (from task's assets/score.py)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from anti_cheat_validation.anti_cheat_validation import (
    validate_solution_integrity,
    get_validation_error_response
)
```

### In Your Task's `score.py`

```python
def evaluate_solution(solution_path: pathlib.Path) -> Dict[str, Any]:
    try:
        solution = scoring.load_module_from_path(solution_path, add_to_sys_modules=True)
        result = solution.evaluate()
        
        # Anti-cheat validation
        validation_result = validate_solution_integrity(
            solution_path=solution_path,
            baseline_metrics=BASELINE_METRICS,
            returned_metrics=result,
            validate_artifacts=True,
            min_model_size_mb=5.0,      # Adjust for your task
            min_batches=50,             # Adjust for your task
            min_loss_drop=0.05          # Adjust for your task
        )
        
        if not validation_result['is_valid']:
            return get_validation_error_response(validation_result)
        
        # Normal scoring if validation passes
        score = calculate_performance_score(result)
        return {"score": score, "message": {...}}
    
    except Exception as e:
        return {"score": 0.0, "message": {"error": str(e)}}
```

## Configuration Examples

### Lightweight Tasks (MNIST, Simple Classification)
```python
min_model_size_mb=1.0,
min_batches=20,
min_loss_drop=0.01,
check_gpu_usage=False
```

### Standard Tasks (CIFAR, NLP)
```python
min_model_size_mb=5.0,
min_batches=50,
min_loss_drop=0.05,
check_gpu_usage=True
```

### Heavy Tasks (ImageNet, Large Models)
```python
min_model_size_mb=50.0,
min_batches=200,
min_loss_drop=0.1,
check_gpu_usage=True
```

## Environment Variables

```bash
# Enable/disable artifact validation (default: true)
export VALIDATE_TRAINING_ARTIFACTS=true

# Enable LLM validation (default: false, requires API key)
export ENABLE_LLM_VALIDATION=true
export ANTHROPIC_API_KEY=sk-ant-...
```

## Documentation

- **`ANTI_CHEAT_GUIDE_UNIVERSAL.md`** - Complete guide for all tasks
- **`__init__.py`** - Package exports and version info
- **Reference Implementation:** `pwc_astock_srl_factors/` - Full example

## Violation Levels

| Level | Behavior | Examples |
|-------|----------|----------|
| **CRITICAL** | Score = 0.0 | Hardcoded values, missing checkpoint, insufficient batches |
| **WARNING** | Scored but flagged | Missing logs, small loss drop, no GPU evidence |
| **CLEAN** | Normal scoring | All checks pass |

## Testing

Test your integration locally:

```python
# test_anticheat.py
import sys
sys.path.insert(0, '.')

from common.anti_cheat_validation import validate_solution_integrity
from pathlib import Path

result = validate_solution_integrity(
    solution_path=Path("path/to/solution.py"),
    baseline_metrics={"accuracy": 69.0},
    returned_metrics={"accuracy": 75.0},
    validate_artifacts=True
)

print(f"Valid: {result['is_valid']}")
print(f"Violations: {result['violations']}")
```
