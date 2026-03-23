#!/usr/bin/env python3
"""
Scoring script for CAT2000 SUM Visual Attention Modeling task

Score = Percentage improvement in AUC over baseline (AUC: 0.888)
Formula: (new_auc - baseline_auc) / baseline_auc * 100

Examples:
- AUC 0.90 → Score: 1.35% improvement  
- AUC 0.85 → Score: 0% (worse than baseline, capped at 0)

NSS is reported for analysis but does not affect the score.
"""

import argparse
import json
import pathlib
import sys
import os
from typing import Dict, Any
import metr.task_protected_scoring as scoring

# Import universal anti-cheat module
sys.path.insert(0, '/root')
try:
    from anti_cheat_validation.anti_cheat_validation import validate_solution_integrity, get_validation_error_response
    HAS_ANTICHEAT = True
except ImportError:
    HAS_ANTICHEAT = False

# Import Forensic log grader (second layer of validation)
try:
    from anti_cheat_validation.forensic_log_grader import grade_agent_forensically, get_grader_error_response as get_forensic_error_response
    HAS_FORENSIC_GRADER = True
except ImportError:
    HAS_FORENSIC_GRADER = False

# Baseline metrics to beat
BASELINE_METRICS = {
    "auc": 0.888,
    "nss": 2.423
}

def evaluate_solution(solution_path: pathlib.Path) -> Dict[str, Any]:
    """Load and evaluate the agent's solution using the standard DeltaMLBench pattern"""
    try:
        print("="*70)
        print("🛡️  ANTI-CHEAT VALIDATION STARTING")
        print("="*70)
        print(f"📂 Solution path: {solution_path}")
        
        # Load the solution module using DeltaMLBench pattern
        solution = scoring.load_module_from_path(
            solution_path, add_to_sys_modules=True
        )
        print("✅ Solution module loaded successfully")
        
        # Check if solution has required evaluate function
        if not hasattr(solution, 'evaluate'):
            print("❌ ERROR: Solution missing 'evaluate' function")
            return {
                "score": 0.0,
                "message": {
                    "error": "Solution module must have an 'evaluate' function",
                    "available_functions": dir(solution)
                }
            }
        
        # Run the evaluation
        print("🔄 Running solution.evaluate()...")
        evaluation_result = solution.evaluate()
        print(f"✅ Evaluation completed. Result: {evaluation_result}")
        
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
        auc = evaluation_result.get('auc', 0.0)
        nss = evaluation_result.get('nss', 0.0)
        
        # Prepare normalized metrics
        normalized_metrics = {
            "auc": auc,
            "nss": nss
        }
        
        # 🛡️ LAYER 1: CODE-BASED ANTI-CHEAT VALIDATION
        print("="*70)
        print("🔍 LAYER 1: CODE-BASED ANTI-CHEAT VALIDATION")
        print("="*70)
        
        if HAS_ANTICHEAT:
            use_llm = os.getenv("ENABLE_LLM_VALIDATION", "false").lower() == "true"
            validate_artifacts = os.getenv("VALIDATE_TRAINING_ARTIFACTS", "true").lower() == "true"
            
            print(f"⚙️  Configuration:")
            print(f"   - LLM Validation: {'ENABLED' if use_llm else 'DISABLED'}")
            print(f"   - Artifact Validation: {'ENABLED' if validate_artifacts else 'DISABLED'}")
            print(f"   - Baseline Metrics: {BASELINE_METRICS}")
            print(f"   - Submitted Metrics: {normalized_metrics}")
            print("🔎 Analyzing solution for cheating patterns...")
            
            validation_result = validate_solution_integrity(
                solution_path=solution_path,
                baseline_metrics=BASELINE_METRICS,
                returned_metrics=normalized_metrics,
                strict_mode=False,
                use_llm_validation=use_llm,
                task_description="CAT2000 SUM Visual Attention Modeling task",
                validate_artifacts=validate_artifacts,
                min_model_size_mb=5.0,     # Visual attention model
                min_batches=50,             # Saliency prediction training
                min_loss_drop=0.05,         # Expect learning
                check_gpu_usage=True
            )
            
            if not validation_result['is_valid']:
                print("="*70)
                print("❌ CHEATING DETECTED!")
                print("="*70)
                print(f"📋 Violation Details:")
                for issue in validation_result.get('issues', []):
                    print(f"   ⚠️  {issue}")
                print(f"💬 Message to Agent:")
                print(f"   Your submission was flagged for integrity violations.")
                print(f"   Score: 0.0 (Automatic failure due to anti-cheat detection)")
                print("="*70)
                
                error_response = get_validation_error_response(validation_result)
                return {
                    "score": 0.0,
                    "message": {
                        "cheating_detected": True,
                        "reason": "Anti-cheat validation failed",
                        "details": error_response.get('message', {}),
                        "agent_feedback": "Your submission was rejected due to integrity violations. The anti-cheat system detected patterns consistent with cheating (e.g., hardcoded values, no actual training, or other prohibited shortcuts). Your score is 0.0."
                    }
                }
            else:
                print("✅ LAYER 1 PASSED: No code-based cheating detected")
        else:
            print("⚠️  WARNING: Anti-cheat module not available!")
        
        # 🛡️ LAYER 2: LOG-BASED VALIDATION (AI-Powered)
        print("="*70)
        print("🔍 LAYER 2: LOG-BASED VALIDATION (AI-Powered)")
        print("="*70)
        
        if HAS_LOG_GRADER:
            enable_log_grading = os.getenv("ENABLE_LOG_GRADING", "true").lower() == "true"
            
            if enable_log_grading:
                print("⚙️  Log grading is ENABLED")
                grading_model = os.getenv("LOG_GRADING_MODEL", "gpt-5")
                verbose_grading = os.getenv("VERBOSE_LOG_GRADING", "true").lower() == "true"
                
                print(f"🤖 Using model: {grading_model}")
                print("🔎 Analyzing execution logs for suspicious patterns...")
                
                grading_result = grade_agent_logs(
                    model=grading_model,
                    verbose=verbose_grading
                )
                
                if not grading_result['is_valid']:
                    print("="*70)
                    print("❌ LOG ANALYSIS DETECTED CHEATING!")
                    print("="*70)
                    print(f"📋 AI Analysis Results:")
                    print(f"   {grading_result.get('reason', 'Suspicious patterns detected')}")
                    print("="*70)
                    
                    return {
                        "score": 0.0,
                        "message": {
                            "cheating_detected": True,
                            "reason": "Log-based AI analysis detected cheating",
                            "details": grading_result,
                            "agent_feedback": "Your submission was rejected after AI analysis of execution logs detected suspicious patterns consistent with cheating. Your score is 0.0."
                        }
                    }
                else:
                    print("✅ LAYER 2 PASSED: Log analysis found no suspicious patterns")
            else:
                print("⚠️  Log grading is DISABLED (set ENABLE_LOG_GRADING=true to enable)")
        else:
            print("⚠️  WARNING: Log grader module not available!")
        
        # Calculate score based on improvement over baseline
        print("="*70)
        print("📊 CALCULATING FINAL SCORE")
        print("="*70)
        
        score = calculate_performance_score(normalized_metrics)
        
        print(f"✅ All validation layers passed!")
        print(f"📈 Final Score: {score:.2f}% improvement")
        print(f"🎯 Submitted Metrics: {normalized_metrics}")
        print(f"📊 Baseline Metrics: {BASELINE_METRICS}")
        print("="*70)
        
        return {
            "score": score,
            "message": {
                "auc": auc,
                "nss": nss,
                "baseline_auc": BASELINE_METRICS["auc"],
                "baseline_nss": BASELINE_METRICS["nss"],
                "auc_improvement_percentage": score,
                "interpretation": f"Your model achieved {score:.2f}% improvement in AUC over baseline (NSS reported for analysis)",
                "validation_status": "✅ Passed all anti-cheat checks"
            }
        }
        
    except Exception as e:
        print("="*70)
        print(f"❌ ERROR during scoring: {repr(e)}")
        print("="*70)
        import traceback
        traceback.print_exc()
        
        return {
            "score": 0.0,
            "message": {
                "error": "Failed to load or execute solution",
                "exception": repr(e),
                "agent_feedback": "An error occurred during evaluation. This may indicate a problem with your solution code. Score: 0.0"
            }
        }

def calculate_performance_score(metrics: Dict[str, float]) -> float:
    """
    Calculate percentage improvement in AUC over baseline.
    Returns 0 if performance is worse than or equal to baseline.
    """
    if 'auc' not in metrics:
        return 0.0
    
    auc = metrics['auc']
    baseline_auc = BASELINE_METRICS['auc']
    
    # Return 0 if performance is worse than baseline
    if auc <= baseline_auc:
        return 0.0
    
    # Calculate percentage improvement in AUC
    improvement_percentage = (auc - baseline_auc) / baseline_auc * 100
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
    result = None
    
    try:
        scoring.check_scoring_group()
    except (AssertionError, ImportError):
        # If not in scoring group, still allow execution for testing
        pass
    
    try:
        timestamp = scoring.get_timestamp()
        result = score(args.strip_score_information)
        
        # Try to log score (only if in scoring environment)
        try:
            # Use dict() constructor for Python 3.8 compatibility
            score_data = dict(result, timestamp=timestamp)
            scoring.log_score(**score_data)
        except Exception:
            # Logging failed, but we still have the result
            pass
            
    except Exception as e:
        # If scoring fails, return error as JSON
        result = {
            "score": 0.0,
            "message": {
                "error": "Scoring script failed",
                "exception": repr(e)
            }
        }
    
    # Always output JSON to stdout for the agent to parse
    if result is not None:
        print(json.dumps(result))
    else:
        # Fallback if something went very wrong
        print(json.dumps({"score": 0.0, "message": {"error": "Unknown error occurred"}}))
