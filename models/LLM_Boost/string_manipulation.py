import re
import os
import textwrap
import math
import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Tuple
import logging



def make_func(body_str):
    def feature_func(row):
        try:
            safe_context = {
                "row": row,
                "log": lambda x: np.log(max(x, 1e-10)),
                "exp": lambda x: np.exp(min(x, 700)),
                "sqrt": lambda x: np.sqrt(max(x, 0)),
                "abs": np.abs,
                "math": math,
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "min": min,
                "max": max,
                "sum": sum,
            }
            result = eval(body_str, safe_context)
            if pd.isna(result) or np.isinf(result):
                return 0.0
            return float(result)
        except (ZeroDivisionError, OverflowError, Exception):
            return 0.0
    return feature_func

def build_feature_prompt(saved_features: List[str], features: List[str]) -> str:
    saved_features_text = "\n".join(saved_features)
    next_func_num = len(saved_features) + 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "feature_extraction.txt")
    try:
        with open(prompt_path, "r") as f:
            prompt = f.read()
    except FileNotFoundError:
        prompt = """Generate a new polynomial feature function based on {{features}}.

Available features: {{features}}

Previously generated features:
{{saved_features}}

Create a function that takes a row and returns a numeric value.

IMPORTANT GUIDELINES:
1. Always handle division by zero: use (row['genus'] + 1e-10) instead of row['genus']
2. For conditional logic: use 'if row['genus'] > 0 else 0' patterns
3. Keep expressions readable and meaningful
4. Avoid complex nested operations that might cause overflow
5. Use mathematical functions like log, sqrt, exp carefully with bounds

Example format:
def feature_name(row):
    return row['cusps'] * (row['rank'] + 1) ** (1/(row['genus'] + 1e-10)) if row['genus'] > 0 else 0

Reason: Brief explanation of why this feature might be useful for Q-gonality prediction."""
    
    # Replace placeholders
    prompt = prompt.replace("{{features}}", ", ".join(features))
    prompt = prompt.replace("{{saved features}}", saved_features_text)
    
    # Add instruction for unique function name to ensure variety
    if "def f1(row):" in prompt:
        prompt = prompt.replace("def f1(row):", f"def f{next_func_num}(row):")
    else:
        prompt += f"\n\nIMPORTANT: Please name your function f{next_func_num} to avoid naming conflicts with existing features."
    
    return prompt

def parse_feature_response(response_text: str, logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[Callable], Optional[str]]:
    try:
        function_match = re.search(
            r"def\s+(\w+)\(row\):\s*(?:return\s*)?([\s\S]+?)(?:\n\s*\n|Reason:|$)",
            response_text
        )
        reason_match = re.search(r"Reason:\s*(.*)", response_text, re.DOTALL)
        
        if not function_match:
            logger.warning("No valid function found in LLM response")
            return None, None, None, None
        
        func_name = function_match.group(1)
        body = function_match.group(2)
        body = textwrap.dedent(body).strip()
        
        if body.startswith('return'):
            body = body[6:].strip()
        
        reason = reason_match.group(1).strip() if reason_match else "No reason provided."
        
        func = make_func(body)
        
        # Return full function definition like working version
        full_code_str = f"def {func_name}(row): return {body}"
        
        return full_code_str, reason, func, body
        
    except Exception as e:
        logger.error(f"Failed to parse feature response: {e}\nRaw response:\n{response_text}")
        return None, None, None, None

def generate_readable_feature_name(feature_body: str, max_length: int = 60) -> str:
    clean_body = feature_body.strip()
    replacements = [
        (r'row\[[\'"](.*?)[\'"]\]', r'\1'),
        (r'\*\*', '_pow_'),
        (r'\+', '_plus_'),
        (r'\-', '_minus_'),
        (r'\*', '_times_'),
        (r'\/', '_div_'),
        (r'\(', '_'),
        (r'\)', '_'),
        (r'\s+', '_'),
        (r'1e-10', 'eps'),
        (r'np\.', ''),
        (r'math\.', ''),
        (r'if.*else.*', 'cond'),
    ]
    for pattern, replacement in replacements:
        clean_body = re.sub(pattern, replacement, clean_body)
    clean_body = re.sub(r'[^\w]', '_', clean_body)
    clean_body = re.sub(r'_+', '_', clean_body)
    clean_body = clean_body.strip('_')
    if len(clean_body) > max_length:
        clean_body = clean_body[:max_length-3] + "..."
    if not clean_body or not clean_body[0].isalpha():
        clean_body = "feat_" + clean_body if clean_body else "custom_feature"
    return clean_body
