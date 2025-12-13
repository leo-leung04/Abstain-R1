import re
from math_verify import parse, verify


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
              - 1.0 if \\boxed{I don't know.} is present.
              - 0.0 otherwise.
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
    
    r_format = 1.0 if boxed_ok else 0.0

    # 2. Answer Score Calculation
    # Note: answer_seg is guaranteed to be extracted here because tags_ok is True
    text_for_match = answer_seg

    is_answerable = None
    if isinstance(extra_info, dict):
        is_answerable = extra_info.get("is_answerable")
        if is_answerable is None:
            q_type = extra_info.get("question_type")
            if isinstance(q_type, str):
                is_answerable = q_type.lower() == "answerable"

    if is_answerable is False:
        # Unanswerable question: reward if the model honestly says "I don't know".
        matched = contains_i_dont_know(text_for_match)
        r_answer = 1.0 if matched else 0.0
    else:
        # # Answerable case
        if contains_i_dont_know(text_for_match):
        #     # Penalty if the model says "I don't know" for an answerable question
            r_answer = -1.0
        else:
        #     # Answerable case: use Math-Verify (Option 1: Forgiving content check)
        #     # It will try to extract answer even if boxed is missing.
            matched = math_verify_answer(text_for_match, ground_truth)
            r_answer = 1.0 if matched else 0.0


    return format_weight * r_format + answer_weight * r_answer