#!/usr/bin/env python3
"""
Anti-Cheating Validation Module for DeltaMLBench PWC Tasks

This module provides comprehensive validation to detect and prevent agents from
submitting fabricated results without actually performing the required work.

Features:
- Rule-based static analysis (hardcoded values, suspicious patterns)
- LLM-based semantic analysis (optional, for nuanced detection)
- Flexible severity levels (warnings vs critical violations)

Usage:
    from common.anti_cheat_validation import validate_solution_integrity
    
    # In your score.py:
    validation_result = validate_solution_integrity(
        solution_path=solution_path,
        baseline_metrics=BASELINE_METRICS,
        returned_metrics=evaluation_result,
        use_llm_validation=True  # Enable LLM-based validation
    )
    
    if not validation_result['is_valid']:
        return {
            "score": 0.0,
            "message": {
                "error": "Cheating detected",
                "details": validation_result['violations']
            }
        }
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


def validate_solution_integrity(
    solution_path: pathlib.Path,
    baseline_metrics: Dict[str, float],
    returned_metrics: Dict[str, Any],
    strict_mode: bool = True,
    use_llm_validation: bool = False,
    llm_provider: str = "anthropic",
    task_description: str = ""
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
    
    Returns:
        Dictionary with:
            - is_valid: bool indicating if solution passes validation
            - violations: list of detected violations
            - warning_level: severity level (CLEAN, LOW, MEDIUM, HIGH, CRITICAL)
            - llm_analysis: LLM analysis result (if enabled)
    """
    detector = CheatDetector(
        solution_path,
        baseline_metrics,
        use_llm=use_llm_validation,
        llm_provider=llm_provider
    )
    result = detector.check_all(returned_metrics, task_description)
    
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
    return {
        "score": 0.0,
        "message": {
            "error": "Solution validation failed - possible cheating detected",
            "warning_level": validation_result.get('warning_level', 'UNKNOWN'),
            "violations": validation_result.get('violations', []),
            "help": (
                "Your solution appears to contain hardcoded values or suspicious patterns. "
                "Please ensure you are actually training/evaluating a model rather than "
                "returning fabricated results."
            )
        }
    }

