"""
Anti-cheat validation module for RE-Bench tasks
"""

from .anti_cheat_validation import (
    validate_solution_integrity,
    get_validation_error_response,
    CheatDetector
)

try:
    from .forensic_log_grader import (
        grade_agent_forensically,
        get_grader_error_response as get_forensic_error_response
    )
    __all__ = [
        'validate_solution_integrity',
        'get_validation_error_response',
        'CheatDetector',
        'grade_agent_forensically',
        'get_forensic_error_response'
    ]
except ImportError:
    __all__ = [
        'validate_solution_integrity',
        'get_validation_error_response',
        'CheatDetector'
    ]
