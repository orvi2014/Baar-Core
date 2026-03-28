"""
benchmarks/evaluators.py — Specialized evaluators for standard datasets.
Extracts and compares answers for accuracy metrics.
"""
import re
from typing import Optional

def evaluate_mmlu(response: str, ground_truth: str) -> bool:
    """Check if the MMLU response (A/B/C/D) matches ground truth."""
    # Pattern: Look for isolated A, B, C, or D (possibly with periods or parens)
    # e.g., "The answer is A.", "Option (B)", etc.
    pattern = r"\b([A-D])\b"
    matches = re.findall(pattern, response.upper())
    if not matches:
        return False
    # If multiple letters found, take the last one as the final answer
    return matches[-1] == ground_truth.upper()

def evaluate_gsm8k(response: str, ground_truth_answer: str) -> bool:
    """Check if the GSM8K math response matches the final answer."""
    # GSM8K ground_truth typically ends with #### <number>
    # The response should also follow this pattern in our prompt.
    pattern = r"####\s*(\-?\d+(?:\.\d+)?)"
    
    # Extract from response
    target_pattern = r"####\s*(\-?\d+(?:\.\d+)?)"
    resp_matches = re.findall(target_pattern, response)
    
    # Extract from ground truth (if it's the full gsm8k 'answer' field)
    gt_matches = re.findall(pattern, ground_truth_answer)
    gt_val = gt_matches[-1] if gt_matches else ground_truth_answer.strip()
    
    if not resp_matches:
        return False
    
    # Compare strings or float values
    try:
        return float(resp_matches[-1]) == float(gt_val)
    except ValueError:
        return resp_matches[-1].strip() == gt_val.strip()

def evaluate_code(response: str, prompt: str, canonical_solution: str) -> bool:
    """
    Very simplified HumanEval evaluator.
    Combines prompt + response and checks for valid Python compilation.
    """
    # 1. HumanEval response typically doesn't include the prompt.
    # We combine them to see if they form a valid function.
    full_code = prompt + "\n" + response
    try:
        compile(full_code, "<string>", "exec")
        return True
    except Exception:
        # Some models might repeat the prompt. Try compiling response itself.
        try:
            compile(response, "<string>", "exec")
            return True
        except Exception:
            return False
