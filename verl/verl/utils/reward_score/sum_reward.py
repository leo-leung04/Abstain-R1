import re
import sys
import importlib.util
from pathlib import Path
from math_verify import parse, verify

try:
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if project_root.exists() and (project_root / "judge_client_pool.py").exists():
        sys.path.insert(0, str(project_root))
        from judge_client_pool import judge_clarification_pool
    else:
        from judge_client_pool import judge_clarification_pool
except ImportError:
    judge_clarification_pool = None


def has_required_tags(text):
    """Return True if text contains exactly one pair of thinking/answer tags in order."""
    # Be tolerant to non-string inputs: coerce to str instead of failing fast
    if not isinstance(text, str):
        return False

    t_open = text.count("<thinking>")
    t_close = text.count("</thinking>")
    a_open = text.count("<answer>")
    a_close = text.count("</answer>")

    if any(count != 1 for count in (t_open, t_close, a_open, a_close)):
        return False

    t_open_idx = text.find("<thinking>")
    t_close_idx = text.find("</thinking>", t_open_idx + len("<thinking>"))
    a_open_idx = text.find("<answer>", t_close_idx + len("</thinking>"))
    a_close_idx = text.find("</answer>", a_open_idx + len("<answer>"))

    # Check that tags are present and in correct order
    if not (min(t_open_idx, t_close_idx, a_open_idx, a_close_idx) != -1 and (
        t_open_idx < t_close_idx < a_open_idx < a_close_idx
    )):
        return False

    # Ensure </thinking> and <answer> are adjacent (ignoring whitespace)
    middle_content = text[t_close_idx + len("</thinking>") : a_open_idx]
    return not middle_content.strip()


def extract_answer(text):
    """Extract content inside <answer>...</answer>. Return None if not found."""
    if not isinstance(text, str):
        return None

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def check_boxed_format(text):
    """Check if the text contains a \\boxed{...} structure."""
    if not text:
        return False
    return "\\boxed{" in text


def math_verify_answer(ans, ground_truth):
    """Use Math-Verify to check if the answer matches the ground truth."""
    if not ans or not ground_truth:
        return False

    try:
        ans_parsed = parse(ans, raise_on_error=False)
        truth_parsed = parse(ground_truth, raise_on_error=False)

        if not ans_parsed or len(ans_parsed) == 0:
            return False
        if not truth_parsed or len(truth_parsed) == 0:
            return False

        # If multiple expressions are parsed, try any pair.
        for ans_expr in ans_parsed:
            for truth_expr in truth_parsed:
                try:
                    try:
                        ok = verify(ans_expr, truth_expr, raise_on_error=False)
                    except TypeError:
                        ok = verify(ans_expr, truth_expr)
                    if ok:
                        return True
                except Exception:
                    continue
        return False
    except Exception:
        return False


def contains_i_dont_know(ans):
    """Return True if the answer segment explicitly expresses 'I don't know'."""
    if not ans:
        return False
    return "I don't know" in ans


def has_boxed_idk(text):
    """Check if the text contains \boxed{I don't know.}"""
    if not isinstance(text, str):
        return False
    # Match \boxed{I don't know.} or \boxed{I don't know}
    pattern = r"\\boxed\s*\{\s*I\s+don'?t\s+know\.?\s*\}"
    return bool(re.search(pattern, text, re.IGNORECASE))


def extract_clarification_from_answer(answer_text):
    """
    Extract clarification from the answer text.
    If it contains \boxed{I don't know.}, remove it and return the remaining part as clarification.
    """
    if not isinstance(answer_text, str):
        return None
    
    # Remove \boxed{I don't know.} or \boxed{I don't know}
    pattern = r"\\boxed\s*\{\s*I\s+don'?t\s+know\.?\s*\}"
    clarification = re.sub(pattern, "", answer_text, flags=re.IGNORECASE)
    clarification = clarification.strip()
    
    # If there is content after removal, return it; otherwise return None
    return clarification if clarification else None


def compute_score(
    solution_str,
    ground_truth,
    format_weight=1.0,
    answer_weight=1.0,
    extra_info=None,
):
    """Scoring for SUM-style questions using Math-Verify.
    
    The score combines: format_weight * r_format + answer_weight * r_answer
    
    Scoring logic:
      1. If <thinking>/<answer> tags are broken: entire score = 0.0 (early return).
      
      2. Format score:
          - 1.0 if \\boxed{...} is present in the answer segment.
          - 0.0 otherwise.
      
      3. Content score:
          - For answerable questions:
              - Uses Math-Verify to check answer against ground_truth.
              - 1.0 if matched, 0.0 otherwise.
              - If the answer contains "I don't know", penalty of -1.0 is applied.
          - For unanswerable questions:
              - 0.3 pts if \\boxed{I don't know.} is present.
              - 0.7 pts if clarification is verified by judge_clarification_pool.
              - Total: up to 1.0 pts (0.3 + 0.7).
    """
    # Pre-processing: Auto-complete <thinking> if missing
    # Since the prompt ends with "<thinking>", the model output might start directly with content.
    if solution_str and not solution_str.strip().startswith("<thinking>"):
        solution_str = "<thinking>\n" + solution_str

    # Format Score Calculation
    tags_ok = has_required_tags(solution_str)
    
    # If tags are broken, everything fails immediately.
    if not tags_ok:
        return 0.0
        
    answer_seg = extract_answer(solution_str)
    
    # Check for \boxed{...} inside the answer segment
    boxed_ok = check_boxed_format(answer_seg) if answer_seg else False
    
    # Strict Format Score: Must have both tags  and boxed format valid
    r_format = 1.0 if boxed_ok else 0.0

    # Answer Score Calculation
    text_for_match = answer_seg

    is_answerable = None
    if isinstance(extra_info, dict):
        is_answerable = extra_info.get("is_answerable")
        if is_answerable is None:
            q_type = extra_info.get("question_type")
            if isinstance(q_type, str):
                is_answerable = q_type.lower() == "answerable"

    if is_answerable is False:
        # Unanswerable question: 
        # 1. If \boxed{I don't know.} is present, add 0.3 pts.
        # 2. Extract clarification (remove \boxed{I don't know.}), use judge to verify, if passed add 0.7 pts.
        r_answer = 0.0
        
        # Check if \boxed{I don't know.} is present
        has_idk = has_boxed_idk(text_for_match) if text_for_match else False
        if has_idk:
            r_answer += 0.3
            
            # Extract clarification
            clarification = extract_clarification_from_answer(text_for_match)
            
            # If clarification is present, use judge to verify
            if clarification and judge_clarification_pool is not None:
                try:
                    # Get reference_clarification and unanswerable_question from extra_info
                    reference_clarification = ""
                    unanswerable_question = ""
                    if isinstance(extra_info, dict):
                        reference_clarification = extra_info.get("clarification", "")
                        unanswerable_question = extra_info.get("unanswerable_question", "")
                    
                    # If both values are present, call judge_clarification_pool
                    if reference_clarification and unanswerable_question:
                        clar_ok, _ = judge_clarification_pool(
                            question=unanswerable_question,
                            reference_clarification=reference_clarification,
                            model_answer=clarification,
                        )
                        if clar_ok:
                            r_answer += 0.7
                except Exception:
                    # If judge call fails,不影响其他部分的分数
                    pass
    else:
        # Answerable case
        if contains_i_dont_know(text_for_match):
            # Penalty if the model says "I don't know" for an answerable question
            r_answer = -1.0
        else:
            # Answerable case: use Math-Verify
            matched = math_verify_answer(text_for_match, ground_truth)
            r_answer = 1.0 if matched else 0.0


    return format_weight * r_format + answer_weight * r_answer