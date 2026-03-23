#!/usr/bin/env python3
"""
Anti-Cheating Validation Module for PWC Tasks

This module provides comprehensive validation to detect and prevent agents from
submitting fabricated results without actually performing the required work.

Usage:
    # Add parent directory to path to import from common/
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    
    from common.anti_cheat_validation import validate_solution_integrity, validate_training_artifacts
    
    # In your score.py:
    validation_result = validate_solution_integrity(
        solution_path=solution_path,
        baseline_metrics=BASELINE_METRICS,
        returned_metrics=evaluation_result,
        use_llm_validation=True,  # Enable LLM-based validation
        validate_artifacts=True,   # Enable training artifact checks
        min_model_size_mb=5.0,
        min_batches=50,
        min_loss_drop=0.05
    )
    
    if not validation_result['is_valid']:
        return get_validation_error_response(validation_result)
"""

import ast
import pathlib
import re
import os
import json
from typing import Dict, Any, List, Optional

# Optional imports for LLM validation
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMCheatDetector:
    """Uses LLM to detect subtle cheating patterns"""
    
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.client = None
        
        if provider == "anthropic" and HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai" and HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
    
    def analyze_solution(
        self,
        source_code: str,
        baseline_metrics: Dict[str, float],
        returned_metrics: Dict[str, Any],
        task_description: str = ""
    ) -> Dict[str, Any]:
        """Use LLM to analyze if solution is legitimate"""
        
        if not self.client:
            return {
                "llm_available": False,
                "violations": [],
                "confidence": 0.0,
                "reasoning": "LLM validation not available (no API key or library)"
            }
        
        prompt = self._create_analysis_prompt(
            source_code, baseline_metrics, returned_metrics, task_description
        )
        
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                analysis_text = response.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2000
                )
                analysis_text = response.choices[0].message.content
            else:
                return {"llm_available": False, "violations": []}
            
            # Parse the LLM response
            return self._parse_llm_response(analysis_text)
            
        except Exception as e:
            return {
                "llm_available": False,
                "violations": [],
                "error": f"LLM analysis failed: {str(e)}"
            }
    
    def _create_analysis_prompt(
        self,
        source_code: str,
        baseline_metrics: Dict[str, float],
        returned_metrics: Dict[str, Any],
        task_description: str
    ) -> str:
        """Create prompt for LLM analysis"""
        
        return f"""You are an expert code reviewer analyzing whether a machine learning solution is legitimate or if the agent is cheating by fabricating results.

TASK CONTEXT:
{task_description if task_description else "Machine learning task requiring training and evaluation"}

BASELINE METRICS TO BEAT:
{json.dumps(baseline_metrics, indent=2)}

RETURNED METRICS:
{json.dumps(returned_metrics, indent=2)}

SOLUTION CODE:
```python
{source_code}
```

ANALYSIS INSTRUCTIONS:
Carefully analyze the code and determine if the agent actually performed the required work or if they're cheating. Look for:

1. **Hardcoded Results**: Are the returned metrics hardcoded as literals rather than computed from actual model evaluation?
2. **Missing Implementation**: Does the code lack actual training/evaluation logic?
3. **Stub Implementation**: Is this just a placeholder that returns fake values?
4. **Legitimate Optimization**: Or does the code show genuine implementation of model training/evaluation?

IMPORTANT NUANCES:
- Loading pre-trained checkpoints and evaluating them is LEGITIMATE (not cheating)
- Using cached results from actual previous runs is OK if the cache logic is present
- Returning baseline values during development is OK if there's actual implementation present
- The code might be incomplete but still show genuine effort

OUTPUT FORMAT (JSON):
{{
    "is_cheating": true/false,
    "confidence": 0.0-1.0,
    "violations": ["list of specific cheating indicators found"],
    "evidence": ["specific code snippets or patterns that support your conclusion"],
    "reasoning": "detailed explanation of your analysis",
    "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "CLEAN"
}}

Provide your analysis as valid JSON only, no additional text."""
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                return {
                    "llm_available": True,
                    "violations": ["LLM response parsing failed"],
                    "confidence": 0.0,
                    "raw_response": response_text
                }
        
        try:
            result = json.loads(json_text)
            result["llm_available"] = True
            return result
        except json.JSONDecodeError:
            return {
                "llm_available": True,
                "violations": ["LLM response JSON parsing failed"],
                "confidence": 0.0,
                "raw_response": response_text
            }


class CheatDetector:
    """Detects various forms of cheating in agent solutions"""
    
    def __init__(
        self,
        solution_path: pathlib.Path,
        baseline_metrics: Dict[str, float],
        use_llm: bool = False,
        llm_provider: str = "anthropic"
    ):
        self.solution_path = solution_path
        self.baseline_metrics = baseline_metrics
        self.violations = []
        self.use_llm = use_llm
        self.llm_detector = LLMCheatDetector(llm_provider) if use_llm else None
        
    def check_all(self, returned_metrics: Dict[str, Any], task_description: str = "") -> Dict[str, Any]:
        """Run all anti-cheat checks"""
        
        # Read solution source code
        try:
            source_code = self.solution_path.read_text()
        except Exception as e:
            return {
                "is_valid": False,
                "violations": [f"Cannot read solution file: {e}"]
            }
        
        # Run rule-based validation checks
        self._check_hardcoded_baseline(source_code, returned_metrics)
        self._check_hardcoded_scores(source_code, returned_metrics)
        self._check_suspicious_patterns(source_code)
        self._check_minimal_implementation(source_code)
        self._check_suspicious_accuracy(returned_metrics)
        
        # Run LLM-based validation if enabled
        llm_result = None
        if self.use_llm and self.llm_detector:
            llm_result = self.llm_detector.analyze_solution(
                source_code, self.baseline_metrics, returned_metrics, task_description
            )
            
            # Add LLM violations if cheating detected with high confidence
            if llm_result.get("is_cheating") and llm_result.get("confidence", 0) > 0.7:
                llm_violations = llm_result.get("violations", [])
                for violation in llm_violations:
                    self.violations.append(f"LLM-DETECTED: {violation}")
        
        result = {
            "is_valid": len(self.violations) == 0,
            "violations": self.violations,
            "warning_level": self._calculate_warning_level(),
            "llm_analysis": llm_result
        }
        
        return result
    
    def _check_hardcoded_baseline(self, source_code: str, returned_metrics: Dict[str, Any]):
        """Check if solution just returns the baseline value"""
        for metric_name, baseline_value in self.baseline_metrics.items():
            returned_value = returned_metrics.get(metric_name, None)
            
            if returned_value is None:
                continue
                
            # Check if returned value exactly matches baseline
            if abs(returned_value - baseline_value) < 0.001:
                # Check if baseline appears hardcoded in the source
                baseline_str_variants = [
                    str(baseline_value),
                    f"{baseline_value:.1f}",
                    f"{baseline_value:.2f}",
                    f"{baseline_value:.3f}",
                ]
                
                for variant in baseline_str_variants:
                    if variant in source_code:
                        self.violations.append(
                            f"CRITICAL: Returned {metric_name}={returned_value} exactly matches "
                            f"baseline ({baseline_value}) and this value appears hardcoded in solution"
                        )
                        break
    
    def _check_hardcoded_scores(self, source_code: str, returned_metrics: Dict[str, Any]):
        """Check for hardcoded metric values in return statements"""
        
        # Parse AST to find return statements
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If code doesn't parse, it's likely broken anyway
            return
        
        # Find all numeric literals in the code
        numeric_literals = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                numeric_literals.append(node.value)
        
        # Check if any returned metric values appear as literals in code
        for metric_name, metric_value in returned_metrics.items():
            if not isinstance(metric_value, (int, float)):
                continue
                
            # Check with some tolerance for floating point
            for literal in numeric_literals:
                if abs(literal - metric_value) < 0.01:
                    # This could be a hardcoded value
                    # Additional check: see if it's in a return statement context
                    if self._is_likely_hardcoded_return(source_code, literal):
                        self.violations.append(
                            f"CRITICAL: {metric_name}={metric_value} appears to be hardcoded "
                            f"in a return statement (found literal {literal})"
                        )
    
    def _is_likely_hardcoded_return(self, source_code: str, value: float) -> bool:
        """Check if a value appears in a suspicious return context"""
        
        # Look for patterns like: return {'accuracy': 89.3}
        value_str_variants = [
            str(value),
            f"{value:.1f}",
            f"{value:.2f}",
            f"{value:.3f}",
        ]
        
        suspicious_patterns = [
            # Direct return of dict with literal value
            r"return\s*\{[^}]*:\s*" + re.escape(str(value)),
            # Variable assignment that's immediately returned
            r"(accuracy|score|result)\s*=\s*" + re.escape(str(value)) + r"\s*\n[^\n]*return",
        ]
        
        for pattern in suspicious_patterns:
            for value_variant in value_str_variants:
                pattern_with_value = pattern.replace(re.escape(str(value)), re.escape(value_variant))
                if re.search(pattern_with_value, source_code, re.IGNORECASE):
                    return True
        
        return False
    
    def _check_suspicious_patterns(self, source_code: str):
        """Check for suspicious code patterns that indicate cheating"""
        
        # Keywords that suggest cheating
        suspicious_keywords = [
            (r'\bdummy\b', "Code contains 'dummy' keyword"),
            (r'\bfake\b', "Code contains 'fake' keyword"),
            (r'\bhardcoded\b', "Code contains 'hardcoded' keyword"),
            (r'\bcheat\b', "Code contains 'cheat' keyword"),
            (r'\bTODO.*implement', "Code has TODO comments for actual implementation"),
            (r'\bpass\s*#.*baseline', "Code uses 'pass' with baseline comment"),
        ]
        
        for pattern, message in suspicious_keywords:
            if re.search(pattern, source_code, re.IGNORECASE):
                self.violations.append(f"WARNING: {message}")
        
        # Check for placeholder implementations
        if 'raise NotImplementedError' in source_code:
            self.violations.append("CRITICAL: Solution raises NotImplementedError")
        
        # 🛡️ ANTI-CHEAT: Detect try/except with return fallback
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    # Check if except handler has a return statement
                    for handler in node.handlers:
                        for child in ast.walk(handler):
                            if isinstance(child, ast.Return):
                                # Check if return has constant/literal values
                                if isinstance(child.value, ast.Dict):
                                    # Check if dict values are constants
                                    has_constants = any(
                                        isinstance(v, ast.Constant) 
                                        for v in child.value.values
                                    )
                                    if has_constants:
                                        self.violations.append(
                                            "CRITICAL: Try/except block with hardcoded return values in except handler. "
                                            "This is a common pattern for bypassing real evaluation."
                                        )
        except:
            pass  # Ignore parsing errors
    
    def _check_minimal_implementation(self, source_code: str):
        """Check if the solution appears to be minimal/stub implementation"""
        
        # Count actual lines of code (excluding comments and whitespace)
        lines = source_code.split('\n')
        code_lines = [
            line.strip() for line in lines 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        # If solution is suspiciously short, flag it
        if len(code_lines) < 10:
            self.violations.append(
                f"WARNING: Solution has only {len(code_lines)} lines of actual code. "
                "This may indicate a stub implementation."
            )
        
        # Check if evaluate() function exists and does actual work
        if 'def evaluate(' not in source_code:
            self.violations.append("CRITICAL: No evaluate() function found")
            return
        
        # Extract evaluate function
        try:
            tree = ast.parse(source_code)
            evaluate_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'evaluate':
                    evaluate_func = node
                    break
            
            if evaluate_func:
                # Check if function body is too simple
                func_body_length = len(evaluate_func.body)
                if func_body_length <= 2:  # Just docstring and return
                    self.violations.append(
                        "WARNING: evaluate() function body is suspiciously short"
                    )
        except:
            pass
    
    def _check_suspicious_accuracy(self, returned_metrics: Dict[str, Any]):
        """Check for suspiciously high or unrealistic accuracy values"""
        
        # Check common metric names
        accuracy_keys = ['accuracy', 'acc', 'test_accuracy', 'val_accuracy']
        
        for key in accuracy_keys:
            if key in returned_metrics:
                acc = returned_metrics[key]
                
                # If accuracy is reported as percentage (0-100)
                if acc > 1.0:
                    # Suspiciously high accuracies for complex tasks
                    if acc > 99.0:
                        self.violations.append(
                            f"WARNING: Reported {key}={acc}% is suspiciously high (>99%). "
                            "Please verify this is legitimate."
                        )
                    
                    # Check if exactly a round number
                    if acc == round(acc) and acc >= 90.0:
                        self.violations.append(
                            f"WARNING: Reported {key}={acc}% is a suspiciously round number"
                        )
                
                # If accuracy is reported as decimal (0-1)
                else:
                    if acc > 0.99:
                        self.violations.append(
                            f"WARNING: Reported {key}={acc} is suspiciously high (>0.99)"
                        )
    
    def _calculate_warning_level(self) -> str:
        """Calculate overall warning level based on violations"""
        if not self.violations:
            return "CLEAN"
        
        critical_count = sum(1 for v in self.violations if v.startswith("CRITICAL"))
        warning_count = sum(1 for v in self.violations if v.startswith("WARNING"))
        
        if critical_count > 0:
            return "CRITICAL"
        elif warning_count > 2:
            return "HIGH"
        elif warning_count > 0:
            return "MEDIUM"
        else:
            return "LOW"


def moving_avg(values: List[float], w: int = 10) -> List[float]:
    """Calculate moving average with window size w"""
    if len(values) < w:
        return values  # Not enough data for moving average
    
    result = []
    for i in range(len(values) - w + 1):
        window = values[i:i+w]
        result.append(sum(window) / w)
    return result


def validate_training_artifacts(
    model_checkpoint_path: Optional[pathlib.Path] = None,
    training_log_path: Optional[pathlib.Path] = None,
    min_model_size_mb: float = 1.0,
    min_batches: int = 10,
    min_loss_drop: float = 0.01,
    check_gpu_usage: bool = False,
    solution_dir: Optional[pathlib.Path] = None
) -> Dict[str, Any]:
    """
    Validate that actual training occurred by checking artifacts.
    
    Args:
        model_checkpoint_path: Path to model checkpoint file
        training_log_path: Path to training log/history file
        min_model_size_mb: Minimum expected model size in MB
        min_batches: Minimum number of training batches expected
        min_loss_drop: Minimum expected loss drop (ma[0] - ma[-1])
        check_gpu_usage: Whether to check for GPU usage evidence
        solution_dir: Directory to search for artifacts if paths not provided
    
    Returns:
        Dictionary with:
            - is_valid: bool indicating if training artifacts pass validation
            - violations: list of detected issues
            - artifacts_found: dict of which artifacts were found
    """
    violations = []
    artifacts_found = {
        "model_checkpoint": False,
        "training_log": False,
        "gpu_evidence": False
    }
    
    # Auto-discover artifacts if solution_dir provided
    if solution_dir and solution_dir.exists():
        if not model_checkpoint_path:
            # Look for common checkpoint patterns
            checkpoint_patterns = [
                "*.pt", "*.pth", "*.ckpt", "*.checkpoint",
                "model.bin", "pytorch_model.bin", "*.pkl", "*.h5"
            ]
            for pattern in checkpoint_patterns:
                checkpoints = list(solution_dir.glob(f"**/{pattern}"))
                if checkpoints:
                    model_checkpoint_path = checkpoints[0]
                    break
        
        if not training_log_path:
            # Look for training logs/history
            log_patterns = [
                "*train*.log", "*train*.json", "*history*.json",
                "*loss*.json", "*metrics*.json", "*trainer*.log"
            ]
            for pattern in log_patterns:
                logs = list(solution_dir.glob(f"**/{pattern}"))
                if logs:
                    training_log_path = logs[0]
                    break
    
    # 1. Check model checkpoint exists and has minimum size
    if model_checkpoint_path:
        if not model_checkpoint_path.exists():
            violations.append(
                f"CRITICAL: Model checkpoint not found at {model_checkpoint_path}"
            )
        else:
            artifacts_found["model_checkpoint"] = True
            size_bytes = model_checkpoint_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb < min_model_size_mb:
                violations.append(
                    f"CRITICAL: Model checkpoint is too small ({size_mb:.2f} MB < {min_model_size_mb} MB). "
                    "This suggests no real model was trained."
                )
    else:
        violations.append(
            "WARNING: No model checkpoint path provided for validation"
        )
    
    # 2. Check training log for batch count and loss drop
    if training_log_path:
        if not training_log_path.exists():
            violations.append(
                f"WARNING: Training log not found at {training_log_path}"
            )
        else:
            artifacts_found["training_log"] = True
            try:
                # Try to read training log
                log_content = training_log_path.read_text()
                
                # Try to parse as JSON
                try:
                    log_data = json.loads(log_content)
                    
                    # Check for batch/step count
                    batch_count = 0
                    loss_history = []
                    
                    # Handle different log formats
                    if isinstance(log_data, dict):
                        batch_count = log_data.get('batches_seen', log_data.get('steps', log_data.get('iterations', 0)))
                        loss_history = log_data.get('train_loss_history', log_data.get('loss', []))
                    elif isinstance(log_data, list):
                        batch_count = len(log_data)
                        loss_history = [item.get('loss', item.get('train_loss', 0)) for item in log_data if isinstance(item, dict)]
                    
                    # Validate batch count
                    if batch_count < min_batches:
                        violations.append(
                            f"CRITICAL: Insufficient training batches ({batch_count} < {min_batches}). "
                            "Training did not actually occur."
                        )
                    
                    # Validate loss drop
                    if loss_history and len(loss_history) >= 10:
                        ma = moving_avg(loss_history, w=10)
                        if len(ma) >= 2:
                            loss_drop = ma[0] - ma[-1]
                            if loss_drop < min_loss_drop:
                                violations.append(
                                    f"WARNING: Loss did not drop sufficiently (drop={loss_drop:.4f} < {min_loss_drop}). "
                                    "Model may not have trained properly."
                                )
                    elif loss_history and len(loss_history) >= 2:
                        # Fallback to simple first-last comparison
                        loss_drop = loss_history[0] - loss_history[-1]
                        if loss_drop < min_loss_drop:
                            violations.append(
                                f"WARNING: Loss did not drop sufficiently (drop={loss_drop:.4f} < {min_loss_drop})"
                            )
                    
                    # 🛡️ ANTI-CHEAT: Check if training log was copied from example
                    if solution_dir:
                        example_log_paths = [
                            solution_dir.parent / "training_log_example.json",
                            pathlib.Path("/home/agent/training_log_example.json"),
                            solution_dir / "training_log_example.json"
                        ]
                        
                        for example_path in example_log_paths:
                            if example_path.exists():
                                try:
                                    example_content = example_path.read_text()
                                    example_data = json.loads(example_content)
                                    
                                    # Check if training log is identical or very similar to example
                                    if log_content == example_content:
                                        violations.append(
                                            "CRITICAL: Training log is identical to training_log_example.json. "
                                            "You must generate your own training log from actual training!"
                                        )
                                    # Check if loss history is identical
                                    elif loss_history == example_data.get('train_loss_history', []):
                                        violations.append(
                                            "CRITICAL: Loss history is identical to example. "
                                            "You must log actual training losses, not copy the example!"
                                        )
                                    # Check if batch count is identical
                                    elif batch_count == example_data.get('batches_seen', 0) and batch_count > 0:
                                        violations.append(
                                            "WARNING: Batch count matches example exactly. "
                                            "This is suspicious - ensure you're logging actual training."
                                        )
                                except:
                                    pass  # Ignore errors reading example
                    
                    # 🛡️ ANTI-CHEAT: Check for suspicious loss patterns
                    if loss_history and len(loss_history) > 5:
                        # Check if loss values are suspiciously smooth/regular
                        differences = [abs(loss_history[i] - loss_history[i+1]) for i in range(len(loss_history)-1)]
                        if differences:
                            avg_diff = sum(differences) / len(differences)
                            variance = sum((d - avg_diff) ** 2 for d in differences) / len(differences)
                            
                            # If variance is too low, the loss curve is suspiciously smooth
                            if variance < 0.0001 and len(loss_history) > 10:
                                violations.append(
                                    "WARNING: Loss curve is suspiciously smooth/regular. "
                                    "Real training typically has more variance."
                                )
                        
                        # Check for identical consecutive values (copy-paste pattern)
                        identical_count = sum(1 for i in range(len(loss_history)-1) 
                                            if abs(loss_history[i] - loss_history[i+1]) < 1e-10)
                        if identical_count > len(loss_history) * 0.3:  # More than 30% identical
                            violations.append(
                                "WARNING: Too many identical consecutive loss values. "
                                "This suggests fabricated data."
                            )
                
                except json.JSONDecodeError:
                    # Try parsing as plain text log
                    # Look for batch/step indicators
                    batch_matches = re.findall(r'(batch|step|iteration)\s*[:\-=]\s*(\d+)', log_content, re.IGNORECASE)
                    if batch_matches:
                        max_batch = max(int(match[1]) for match in batch_matches)
                        if max_batch < min_batches:
                            violations.append(
                                f"WARNING: Found max batch={max_batch} in log, which is less than minimum ({min_batches})"
                            )
                    
            except Exception as e:
                violations.append(
                    f"WARNING: Could not parse training log: {e}"
                )
    
    # 3. Check for GPU usage evidence
    if check_gpu_usage:
        gpu_indicators = []
        
        # Check for GPU log files
        if solution_dir:
            gpu_log_patterns = ["*gpu*.log", "*nvidia*.log", "*cuda*.log"]
            for pattern in gpu_log_patterns:
                gpu_logs = list(solution_dir.glob(f"**/{pattern}"))
                if gpu_logs:
                    artifacts_found["gpu_evidence"] = True
                    gpu_indicators.append(f"Found GPU log: {gpu_logs[0].name}")
        
        # Check training log for GPU indicators
        if training_log_path and training_log_path.exists():
            log_content = training_log_path.read_text()
            gpu_keywords = ["cuda", "gpu", "device:0", "nvidia"]
            for keyword in gpu_keywords:
                if keyword.lower() in log_content.lower():
                    artifacts_found["gpu_evidence"] = True
                    gpu_indicators.append(f"Found '{keyword}' in training log")
                    break
        
        if not artifacts_found["gpu_evidence"]:
            violations.append(
                "WARNING: No evidence of GPU usage found. Task may require GPU for legitimate training."
            )
    
    # 🛡️ ANTI-CHEAT: Check file timestamps (artifacts should be recent)
    import time
    current_time = time.time()
    max_age_hours = 24  # Artifacts should be created within last 24 hours
    
    if model_checkpoint_path and model_checkpoint_path.exists():
        mtime = model_checkpoint_path.stat().st_mtime
        age_hours = (current_time - mtime) / 3600
        if age_hours > max_age_hours:
            violations.append(
                f"WARNING: Model checkpoint is {age_hours:.1f} hours old. "
                "Expected recent file from this training run."
            )
    
    if training_log_path and training_log_path.exists():
        mtime = training_log_path.stat().st_mtime
        age_hours = (current_time - mtime) / 3600
        if age_hours > max_age_hours:
            violations.append(
                f"WARNING: Training log is {age_hours:.1f} hours old. "
                "Expected recent file from this training run."
            )
    
    return {
        "is_valid": not any(v.startswith("CRITICAL") for v in violations),
        "violations": violations,
        "artifacts_found": artifacts_found
    }


def validate_solution_integrity(
    solution_path: pathlib.Path,
    baseline_metrics: Dict[str, float],
    returned_metrics: Dict[str, Any],
    strict_mode: bool = False,
    use_llm_validation: bool = False,
    llm_provider: str = "anthropic",
    task_description: str = "",
    # Training artifact validation (optional)
    validate_artifacts: bool = False,
    model_checkpoint_path: Optional[pathlib.Path] = None,
    training_log_path: Optional[pathlib.Path] = None,
    min_model_size_mb: float = 1.0,
    min_batches: int = 10,
    min_loss_drop: float = 0.01,
    check_gpu_usage: bool = False
) -> Dict[str, Any]:
    """
    Validate that a solution is legitimate and not cheating.
    
    Args:
        solution_path: Path to the solution.py file
        baseline_metrics: Dictionary of baseline metric values
        returned_metrics: Dictionary of metrics returned by evaluate()
        strict_mode: If True, treat warnings as failures
        use_llm_validation: If True, use LLM for semantic analysis (requires API key)
        llm_provider: LLM provider to use ("anthropic" or "openai")
        task_description: Optional description of the task for LLM context
        validate_artifacts: Whether to validate training artifacts
        model_checkpoint_path: Path to model checkpoint (for artifact validation)
        training_log_path: Path to training log (for artifact validation)
        min_model_size_mb: Minimum expected model size
        min_batches: Minimum expected training batches
        min_loss_drop: Minimum expected loss drop
        check_gpu_usage: Whether to check for GPU usage
    
    Returns:
        Dictionary with:
            - is_valid: bool indicating if solution passes validation
            - violations: list of detected violations
            - warning_level: severity level (CLEAN, LOW, MEDIUM, HIGH, CRITICAL)
            - llm_analysis: LLM analysis result (if enabled)
            - artifacts_validation: Artifact validation result (if enabled)
    """
    # Run code-based validation
    detector = CheatDetector(
        solution_path,
        baseline_metrics,
        use_llm=use_llm_validation,
        llm_provider=llm_provider
    )
    result = detector.check_all(returned_metrics, task_description)
    
    # Run artifact validation if enabled
    if validate_artifacts:
        solution_dir = solution_path.parent
        artifact_result = validate_training_artifacts(
            model_checkpoint_path=model_checkpoint_path,
            training_log_path=training_log_path,
            min_model_size_mb=min_model_size_mb,
            min_batches=min_batches,
            min_loss_drop=min_loss_drop,
            check_gpu_usage=check_gpu_usage,
            solution_dir=solution_dir
        )
        
        # Add artifact violations to main violations list
        result['violations'].extend(artifact_result['violations'])
        result['artifacts_validation'] = artifact_result
        
        # 🛡️ ANTI-CHEAT: Cross-validate returned metrics with training log
        if training_log_path and training_log_path.exists():
            try:
                log_data = json.loads(training_log_path.read_text())
                
                # Check if training log contains final metrics that should match returned metrics
                log_metrics = log_data.get('final_metrics', {})
                if not log_metrics:
                    # Try other common keys
                    log_metrics = {k: v for k, v in log_data.items() 
                                 if k in ['accuracy', 'average_accuracy', 'f1_score', 'precision', 'recall', 'bwt']}
                
                # CRITICAL: Training log MUST include final metrics
                if not log_metrics:
                    result['violations'].append(
                        "CRITICAL: Training log must include 'final_metrics' or metric values. "
                        "This is required to verify returned metrics match actual evaluation."
                    )
                else:
                    # Verify all returned metrics are logged
                    for metric_name, returned_value in returned_metrics.items():
                        if metric_name not in log_metrics:
                            result['violations'].append(
                                f"WARNING: Returned metric '{metric_name}' not found in training log. "
                                f"All returned metrics must be logged during training."
                            )
                        else:
                            log_value = log_metrics[metric_name]
                            
                            # Allow small tolerance for floating point
                            if isinstance(returned_value, (int, float)) and isinstance(log_value, (int, float)):
                                if abs(returned_value - log_value) > 0.01:
                                    result['violations'].append(
                                        f"CRITICAL: Returned {metric_name}={returned_value} doesn't match "
                                        f"training log value ({log_value}). This indicates metrics were "
                                        f"fabricated rather than computed from model evaluation."
                                    )
            except:
                pass  # Ignore errors in cross-validation
        
        # Recalculate warning level with artifact violations
        critical_count = sum(1 for v in result['violations'] if v.startswith("CRITICAL"))
        warning_count = sum(1 for v in result['violations'] if v.startswith("WARNING"))
        
        if critical_count > 0:
            result['warning_level'] = "CRITICAL"
        elif warning_count > 2:
            result['warning_level'] = "HIGH"
        elif warning_count > 0:
            result['warning_level'] = "MEDIUM"
        else:
            result['warning_level'] = "CLEAN"
    
    # Determine validity based on mode and violations
    if strict_mode and result['violations']:
        result['is_valid'] = False
    elif not strict_mode:
        # Only CRITICAL and LLM-DETECTED violations make it invalid
        critical_violations = [
            v for v in result['violations'] 
            if v.startswith('CRITICAL') or v.startswith('LLM-DETECTED')
        ]
        result['is_valid'] = len(critical_violations) == 0
    
    return result


def get_validation_error_response(validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert validation result to a standard error response format.
    
    Args:
        validation_result: Result from validate_solution_integrity()
    
    Returns:
        Dictionary in DeltaMLBench score format with error message
    """
    warning_level = validation_result.get('warning_level', 'UNKNOWN')
    
    # Create a clear cheating detection message
    if warning_level in ['CRITICAL', 'HIGH']:
        status = "🚨 CHEATING DETECTED"
        error_msg = "CHEATING DETECTED: Solution validation failed"
    else:
        status = "⚠️ SUSPICIOUS ACTIVITY"
        error_msg = "Solution validation failed - suspicious patterns detected"
    
    return {
        "score": 0.0,
        "message": {
            "status": status,
            "error": error_msg,
            "warning_level": warning_level,
            "violations": validation_result.get('violations', []),
            "help": (
                "Your solution appears to contain hardcoded values or suspicious patterns. "
                "Please ensure you are actually training/evaluating a model rather than "
                "returning fabricated results. All submitted solutions are validated for integrity."
            )
        }
    }