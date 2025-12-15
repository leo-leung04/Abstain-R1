"""
Evaluation scaffold for SUM-style result files with reference answers.

What it does now:
- Loads the reference file (answerable_question, unanswerable_question, ground_truth, clarification).
- Checks format adherence (exactly one <thinking>...</thinking> and one <answer>...</answer> in order).
- For answerable outputs:
  * String-match correctness against ground_truth using a simple normalized compare.
  * Detects if the model said "I don't know" (boxed or not) to flag false-unanswerable cases.
- For unanswerable outputs:
  * Checks for boxed I don't know (simple string search) to approximate refusal correctness.
  * Format adherence stats.

What is left TODO (placeholders):
- Hooks to call an external LLM (e.g., DeepSeek) for semantic correctness when string match is insufficient,
  and for judging clarification quality on unanswerable cases.
- Prompt templates for the above LLM calls.

Usage:
  python run/evaluate_with_reference.py \
    --reference data/sum_test_with_clarification_retry.jsonl \
    --files results/qwen25_3b_instruct_result.jsonl [...]

By default, if --files is omitted, it uses the curated list in DEFAULT_FILES below.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_FILES = [
    "results/qwen25_3b_instruct_result.jsonl",
    "results/qwen25_7b_instruct_result.jsonl",
    "results/deepseek_r1_distill_qwen_7b_structured_result.jsonl",
    "results/qwen25_math_7b_instruct_structured_result.jsonl",
    "results/deepseek_chat_result.jsonl",
    "results/deepseek_reason_result.jsonl",
    "results/gpt_5_1_reasoning_medium_result.jsonl",
    "results/gpt_5_1_result.jsonl",
    "results/qwen25_3b_rft_structured_sum_result.jsonl",
]


IDK_PATTERNS = [
    r"\\boxed\s*{\s*I don't know\.?\s*}",
    r"I don't know",
]


def load_reference(path: Path) -> Dict[str, Dict]:
    """Load reference records keyed by answerable_question."""
    ref = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec["answerable_question"]
            ref[key] = rec
    return ref


def add_thinking_tag_if_missing(text: str) -> str:
    """Add <thinking> tag at the beginning if text doesn't start with it."""
    if not isinstance(text, str):
        return text
    # Check if text starts with <thinking> (case insensitive, after stripping)
    text_stripped = text.strip()
    if text_stripped.lower().startswith("<thinking>"):
        return text
    # If not, add <thinking> tag at the beginning
    return f"<thinking>\n{text}"


def has_required_tags(text: str) -> bool:
    """Return True if text contains exactly one pair of thinking/answer tags in order."""
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

    return min(t_open_idx, t_close_idx, a_open_idx, a_close_idx) != -1 and (
        t_open_idx < t_close_idx < a_open_idx < a_close_idx
    )


def extract_answer(text: str) -> Optional[str]:
    """Extract content inside <answer>...</answer>. Return None if not found."""
    if not isinstance(text, str):
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


BOXED_RE = re.compile(r"\\boxed\s*{\s*([^}]*)\s*}")


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = BOXED_RE.sub(r"\1", s)
    s = re.sub(r"[=\\$]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def string_match_answer(ans: str, ground_truth: str) -> bool:
    """Simple normalized string equality check."""
    if not ans or not ground_truth:
        return False
    norm_ans = normalize(ans)
    norm_truth = normalize(ground_truth)
    if not norm_truth:
        return False
    return norm_truth in norm_ans


def has_idk(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for pat in IDK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


@dataclass
class AnswerableMetrics:
    total: int = 0
    format_ok: int = 0
    correct_string: int = 0
    false_unanswerable: int = 0  # said I don't know on answerable
    correct_llm: int = 0
    idk_llm: int = 0  # LLM judged as explicit I don't know / refusal


@dataclass
class UnanswerableMetrics:
    total: int = 0
    format_ok: int = 0
    has_box_idk: int = 0  # crude correct refusal count via string search
    refusal_llm: int = 0
    clarification_llm: int = 0


@dataclass
class FileMetrics:
    answerable: AnswerableMetrics = field(default_factory=AnswerableMetrics)
    unanswerable: UnanswerableMetrics = field(default_factory=UnanswerableMetrics)
    missing_reference: int = 0
    invalid_json: int = 0


def parse_json_from_model(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON object from a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def build_answerable_prompt(question: str, ground_truth: str, model_output: str) -> str:
    return (
        "You are grading a math answer. The model output may include a long reasoning/thinking section; focus only on the final answer.\n"
        "Labels (choose exactly one):\n"
        '  - \"correct\": final answer matches the ground truth (allow trivial formatting like spaces, trailing .0).\n'
        '  - \"idk\": the response clearly refuses / says it cannot answer / does not know.\n'
        '  - \"incorrect\": all other cases (wrong value, missing final answer, irrelevant). If uncertain, choose \"incorrect\".\n\n'
        "Return JSON exactly: {\"label\": \"correct\"|\"incorrect\"|\"idk\", \"reason\": \"<brief why>\"}.\n"
        "Do not rewrite the question or add text outside JSON.\n\n"
        f"Question:\n{question}\n\n"
        f"Ground truth final answer:\n{ground_truth}\n\n"
        f"Model output to evaluate:\n{model_output}\n"
    )


def build_unanswerable_prompt(question: str, clarification: str, model_output: str) -> str:
    return (
        "You grade a refusal for an unanswerable question. The model output may contain long reasoning; focus on whether it refuses and on the stated reason.\n"
        "Return JSON exactly: {\"refusal\": 0 or 1, \"clarification_ok\": 0 or 1, \"reason\": \"<brief why>\"}.\n"
        "Definitions (be strict):\n"
        "  - refusal=1 if the response clearly indicates it cannot answer or lacks required info (even without the phrase \"I don't know\"). Otherwise 0.\n"
        "  - clarification_ok=1 if the stated reason aligns with the reference clarification’s missing-information point(s). Otherwise 0. If uncertain, set 0.\n"
        "If the model gives a concrete answer, set refusal=0 and clarification_ok=0.\n"
        "Do not add text outside JSON.\n\n"
        f"Unanswerable question:\n{question}\n\n"
        f"Reference clarification:\n{clarification}\n\n"
        f"Model output to evaluate:\n{model_output}\n"
    )


def get_llm_client(model_name: Optional[str]):
    """Instantiate OpenAI-compatible client (OpenAI or DeepSeek) if model is provided and key is set."""
    if not model_name:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai package not installed; LLM evaluation disabled.")
        return None
    # Choose API key / base_url based on model name
    api_key_env = "OPENAI_API_KEY"
    base_url = None
    if "deepseek" in model_name.lower():
        api_key_env = "DEEPSEEK_API_KEY"
        base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"[WARN] {api_key_env} not set; LLM evaluation disabled for model {model_name}.")
        return None
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def call_model(client, model: str, prompt: str, max_output_tokens: int, max_retries: int = 3) -> Optional[str]:
    """Dispatch to Chat Completions API (for DeepSeek and OpenAI models)."""
    import time
    # o3 series models use max_completion_tokens instead of max_tokens
    model_lower = model.lower()
    is_o3_model = "o3" in model_lower
    
    for attempt in range(max_retries):
        try:
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": 30.0,
            }
            # Use max_completion_tokens for o3 models, max_tokens for others
            if is_o3_model:
                request_params["max_completion_tokens"] = max_output_tokens
            else:
                request_params["max_tokens"] = max_output_tokens
            
            # Use Chat Completions API for all models (DeepSeek and OpenAI)
            resp = client.chat.completions.create(**request_params)
            if resp and resp.choices:
                return resp.choices[0].message.content
            return None
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"[WARN] LLM call failed (model={model}, attempt {attempt+1}/{max_retries}): {error_msg}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[WARN] LLM call failed (model={model}) after {max_retries} attempts: {error_msg}")
                return None
    return None


def llm_classify_answerable(
    primary_client,
    primary_model: Optional[str],
    question: str,
    ground_truth: str,
    model_output: str,
    fallback_client=None,
    fallback_model: Optional[str] = None,
) -> Optional[str]:
    """Return 'correct' | 'incorrect' | 'idk' based on LLM JSON output."""
    prompt = build_answerable_prompt(question, ground_truth, model_output)
    for client, model in ((primary_client, primary_model), (fallback_client, fallback_model)):
        if not client or not model:
            continue
        # o3 models need more tokens for JSON output
        is_o3 = "o3" in (model or "").lower()
        max_tokens = 1000 if is_o3 else 150
        content = call_model(client, model, prompt, max_output_tokens=max_tokens)
        if not content:
            continue
        parsed = parse_json_from_model(content)
        if not parsed:
            continue
        label = parsed.get("label")
        if label in {"correct", "incorrect", "idk"}:
            return label
    return None


def llm_classify_unanswerable(
    primary_client,
    primary_model: Optional[str],
    question: str,
    clarification: str,
    model_output: str,
    fallback_client=None,
    fallback_model: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Return (refusal, clarification_ok) as 0/1 via LLM JSON output."""
    prompt = build_unanswerable_prompt(question, clarification, model_output)
    for client, model in ((primary_client, primary_model), (fallback_client, fallback_model)):
        if not client or not model:
            continue
        # o3 models need more tokens for JSON output
        is_o3 = "o3" in (model or "").lower()
        max_tokens = 1000 if is_o3 else 200
        content = call_model(client, model, prompt, max_output_tokens=max_tokens)
        if not content:
            continue
        parsed = parse_json_from_model(content)
        if not parsed:
            continue
        refusal = parsed.get("refusal")
        clarification_ok = parsed.get("clarification_ok")
        def to_bit(x):
            if x in (0, 1):
                return x
            if isinstance(x, str) and x.strip() in {"0", "1"}:
                return int(x.strip())
            return None
        return to_bit(refusal), to_bit(clarification_ok)
    return None, None


def evaluate_file(
    path: Path,
    reference: Dict[str, Dict],
    llm_client=None,
    llm_model: Optional[str] = None,
    llm_fallback_client=None,
    llm_fallback_model: Optional[str] = None,
    record_workers: int = 1,
    show_progress: bool = False,
) -> Tuple[FileMetrics, List[Dict[str, Any]]]:
    metrics = FileMetrics()
    records: List[Tuple[int, Dict[str, Any]]] = []
    annotations: List[Tuple[int, Dict[str, Any]]] = []
    total_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                metrics.invalid_json += 1
                continue
            records.append((idx, rec))

    if record_workers <= 1:
        for idx, rec in records:
            if show_progress and total_lines > 0 and idx % 50 == 0:
                print(f"    [{path.name}] processed {idx}/{total_lines}", flush=True)
            partial, annotation = _process_record(
                rec,
                reference,
                llm_client,
                llm_model,
                llm_fallback_client,
                llm_fallback_model,
                line_no=idx,
            )
            merge_metrics(metrics, partial)
            annotations.append((idx, annotation))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        lock = Lock()

        def worker(rec_tuple):
            idx, rec = rec_tuple
            partial, annotation = _process_record(
                rec,
                reference,
                llm_client,
                llm_model,
                llm_fallback_client,
                llm_fallback_model,
                line_no=idx,
            )
            return idx, partial, annotation

        with ThreadPoolExecutor(max_workers=record_workers) as ex:
            futures = {ex.submit(worker, rec): rec for rec in records}
            completed = 0
            for fut in as_completed(futures):
                idx, partial, annotation = fut.result()
                with lock:
                    merge_metrics(metrics, partial)
                    annotations.append((idx, annotation))
                    completed += 1
                    if show_progress and total_lines > 0 and completed % 50 == 0:
                        print(f"    [{path.name}] processed {completed}/{total_lines}", flush=True)

    annotations.sort(key=lambda x: x[0])
    ordered_annotations = [ann for _, ann in annotations]
    return metrics, ordered_annotations


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (0.00%)"
    return f"{numerator}/{denominator} ({numerator/denominator:.2%})"


def merge_metrics(dst: FileMetrics, src: FileMetrics):
    dst.answerable.total += src.answerable.total
    dst.answerable.format_ok += src.answerable.format_ok
    dst.answerable.correct_string += src.answerable.correct_string
    dst.answerable.false_unanswerable += src.answerable.false_unanswerable
    dst.answerable.correct_llm += src.answerable.correct_llm
    dst.answerable.idk_llm += src.answerable.idk_llm

    dst.unanswerable.total += src.unanswerable.total
    dst.unanswerable.format_ok += src.unanswerable.format_ok
    dst.unanswerable.has_box_idk += src.unanswerable.has_box_idk
    dst.unanswerable.refusal_llm += src.unanswerable.refusal_llm
    dst.unanswerable.clarification_llm += src.unanswerable.clarification_llm

    dst.missing_reference += src.missing_reference
    dst.invalid_json += src.invalid_json


def _process_record(
    rec: Dict[str, Any],
    reference: Dict[str, Dict],
    llm_client,
    llm_model: Optional[str],
    llm_fallback_client,
    llm_fallback_model: Optional[str],
    line_no: Optional[int] = None,
) -> Tuple[FileMetrics, Dict[str, Any]]:
    metrics = FileMetrics()
    annotation: Dict[str, Any] = {
        "line": line_no,
        "answerable_question": rec.get("answerable_question"),
        "unanswerable_question": rec.get("unanswerable_question"),
        "ground_truth": None,
        "clarification": None,
        "answerable_output": rec.get("answerable_output", ""),
        "unanswerable_output": rec.get("unanswerable_output", ""),
        "answerable_tags": {},
        "unanswerable_tags": {},
    }

    ans_q = annotation["answerable_question"]
    unans_q = annotation["unanswerable_question"]
    # Add <thinking> tag if missing before evaluation
    ans_out = add_thinking_tag_if_missing(annotation["answerable_output"])
    unans_out = add_thinking_tag_if_missing(annotation["unanswerable_output"])

    ref_rec = reference.get(ans_q)
    if ref_rec is None:
        metrics.missing_reference += 1
        annotation["missing_reference"] = True
        return metrics, annotation

    annotation["ground_truth"] = ref_rec.get("ground_truth")
    annotation["clarification"] = ref_rec.get("clarification")

    # Answerable
    metrics.answerable.total += 1
    ans_fmt_ok = has_required_tags(ans_out)
    annotation["answerable_tags"]["format_ok"] = ans_fmt_ok
    if ans_fmt_ok:
        metrics.answerable.format_ok += 1
    ans_text = extract_answer(ans_out) if ans_fmt_ok else None
    ans_idk = has_idk(ans_text or ans_out)
    annotation["answerable_tags"]["said_idk"] = ans_idk
    if ans_idk:
        metrics.answerable.false_unanswerable += 1
    string_match = string_match_answer(ans_text or ans_out, ref_rec["ground_truth"])
    annotation["answerable_tags"]["string_match"] = string_match
    label = None
    if string_match:
        metrics.answerable.correct_string += 1
    else:
        if (llm_client and llm_model) or (llm_fallback_client and llm_fallback_model):
            label = llm_classify_answerable(
                llm_client,
                llm_model,
                ans_q,
                ref_rec["ground_truth"],
                ans_text or ans_out,
                fallback_client=llm_fallback_client,
                fallback_model=llm_fallback_model,
            )
            if label == "correct":
                metrics.answerable.correct_llm += 1
            elif label == "idk":
                metrics.answerable.idk_llm += 1
    annotation["answerable_tags"]["llm_label"] = label

    # Unanswerable
    metrics.unanswerable.total += 1
    unans_fmt_ok = has_required_tags(unans_out)
    annotation["unanswerable_tags"]["format_ok"] = unans_fmt_ok
    if unans_fmt_ok:
        metrics.unanswerable.format_ok += 1
    unans_text = extract_answer(unans_out) if unans_fmt_ok else None
    unans_idk = has_idk(unans_text or unans_out)
    annotation["unanswerable_tags"]["has_box_idk"] = unans_idk
    if unans_idk:
        metrics.unanswerable.has_box_idk += 1
    refusal = None
    clar_ok = None
    if (llm_client and llm_model) or (llm_fallback_client and llm_fallback_model):
        refusal, clar_ok = llm_classify_unanswerable(
            llm_client,
            llm_model,
            unans_q,
            ref_rec.get("clarification", ""),
            unans_text or unans_out,
            fallback_client=llm_fallback_client,
            fallback_model=llm_fallback_model,
        )
        if refusal == 1:
            metrics.unanswerable.refusal_llm += 1
        if clar_ok == 1:
            metrics.unanswerable.clarification_llm += 1
    annotation["unanswerable_tags"]["llm_refusal"] = None if refusal is None else bool(refusal)
    annotation["unanswerable_tags"]["llm_clarification_ok"] = None if clar_ok is None else bool(clar_ok)

    return metrics, annotation


def summarize(path: Path, metrics: FileMetrics, llm_enabled: bool) -> str:
    lines = [f"{path}"]
    a = metrics.answerable
    u = metrics.unanswerable
    lines.append(f"  answerable format_ok: {format_rate(a.format_ok, a.total)}")
    lines.append(f"  answerable correct (string match): {format_rate(a.correct_string, a.total)}")
    lines.append(f"  answerable false-unanswerable (said IDK): {format_rate(a.false_unanswerable, a.total)}")
    if llm_enabled:
        lines.append(f"  answerable correct (LLM): {format_rate(a.correct_llm, a.total)}")
        lines.append(f"  answerable idk (LLM): {format_rate(a.idk_llm, a.total)}")
    lines.append(f"  unanswerable format_ok: {format_rate(u.format_ok, u.total)}")
    lines.append(f"  unanswerable has_box_idk (crude refusal): {format_rate(u.has_box_idk, u.total)}")
    if llm_enabled:
        lines.append(f"  unanswerable refusal (LLM): {format_rate(u.refusal_llm, u.total)}")
        lines.append(f"  unanswerable clarification_ok (LLM): {format_rate(u.clarification_llm, u.total)}")
    if metrics.invalid_json:
        lines.append(f"  invalid JSON lines: {metrics.invalid_json}")
    if metrics.missing_reference:
        lines.append(f"  missing reference matches: {metrics.missing_reference}")
    return "\n".join(lines)


def iter_files(files: Iterable[str]) -> List[Path]:
    return [Path(f) for f in files]


def get_eval_type(model_name: Optional[str]) -> str:
    """Determine evaluation type based on model name: 'deepseek' or 'openai'."""
    if not model_name:
        return "deepseek"  # default
    model_lower = model_name.lower()
    if "deepseek" in model_lower:
        return "deepseek"
    else:
        return "openai"


def get_annotation_path(path: Path, annotate_dir: Optional[str], eval_type: str) -> Path:
    """Generate annotation path in the appropriate eval folder."""
    stem = path.stem
    suffix = path.suffix
    new_name = f"{stem}{suffix}.eval.jsonl"
    if annotate_dir:
        base = Path(annotate_dir)
        return base / new_name
    # Auto-generate: use eval_type folder
    return path.parent / f"{eval_type}_eval" / new_name


def write_annotations(path: Path, annotations: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for ann in annotations:
            fout.write(json.dumps(ann, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate result files against reference with format checks, string matching, and optional LLM grading.")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/sum/sum_test_with_clarification_retry.jsonl",
        help="Reference jsonl path (default: data/sum_test_with_clarification_retry.jsonl)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Result files to evaluate. Default is the curated list in the script.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-chat",
        help="If set, use this model (e.g., deepseek-chat, deepseek-reasoner) for semantic grading.",
    )
    parser.add_argument(
        "--llm-fallback-model",
        type=str,
        default=None,
        help="Optional fallback model (e.g., deepseek-v3) if primary LLM call fails.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel workers per file (records). Default 1 (sequential within file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write a JSON summary to this path. If not set, auto-generate from input file name with 'deepseek' suffix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set and --output exists, skip files already present in the summary.",
    )
    parser.add_argument(
        "--annotate-dir",
        type=str,
        default=None,
        help="Directory to store per-record annotation jsonl files (default: auto-generate with 'deepseek' suffix).",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable writing per-record annotation jsonl files.",
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    reference = load_reference(ref_path)

    targets = iter_files(args.files) if args.files else iter_files(DEFAULT_FILES)

    llm_client = get_llm_client(args.llm_model)
    llm_fallback_client = get_llm_client(args.llm_fallback_model)
    llm_enabled = (llm_client is not None and args.llm_model is not None) or (
        llm_fallback_client is not None and args.llm_fallback_model is not None
    )

    annotate_enabled = not args.no_annotate

    # Determine evaluation type (deepseek or openai) based on model
    eval_type = get_eval_type(args.llm_model)
    print(f"[INFO] Evaluation type: {eval_type} (model: {args.llm_model})")

    # Auto-generate output path if not specified (use first file as base)
    auto_output_path = None
    if not args.output and targets:
        first_file = targets[0]
        stem = first_file.stem
        eval_dir = first_file.parent / f"{eval_type}_eval"
        auto_output_path = eval_dir / f"{stem}_eval_summary.json"
        print(f"[INFO] Auto-generated output path: {auto_output_path}")

    # Determine final output path for resume check
    final_output_path = args.output or auto_output_path
    results_summary: Dict[str, Any] = {}
    processed: set[str] = set()
    if final_output_path and args.resume:
        out_path = Path(final_output_path)
        if out_path.exists():
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        results_summary = data
                        processed = set(data.keys())
                        print(f"[INFO] resume enabled; found {len(processed)} entries in {out_path}")
            except Exception as e:
                print(f"[WARN] failed to load existing summary for resume: {e}")

    # Auto-generate annotate_dir if not specified
    auto_annotate_dir = None
    if annotate_enabled and not args.annotate_dir and targets:
        first_file = targets[0]
        auto_annotate_dir = str(first_file.parent / f"{eval_type}_eval")
        print(f"[INFO] Auto-generated annotate_dir: {auto_annotate_dir}")

    for i, path in enumerate(targets, start=1):
        print(f"[FILE {i}/{len(targets)}] {path}")
        if str(path) in processed:
            print("  [SKIP] already in summary (resume).")
            continue
        if not path.exists():
            print(f"[WARN] skip missing file: {path}")
            continue
        metrics, annotations = evaluate_file(
            path,
            reference,
            llm_client=llm_client,
            llm_model=args.llm_model,
            llm_fallback_client=llm_fallback_client,
            llm_fallback_model=args.llm_fallback_model,
            record_workers=max(1, args.workers),
            show_progress=True,
        )
        print(summarize(path, metrics, llm_enabled))
        print()
        results_summary[str(path)] = asdict(metrics)
        if annotate_enabled:
            ann_path = get_annotation_path(path, args.annotate_dir or auto_annotate_dir, eval_type)
            write_annotations(ann_path, annotations)
            print(f"  [INFO] annotations written to {ann_path}")

    # Write summary to output path
    if final_output_path:
        out_path = Path(final_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fout:
            json.dump(results_summary, fout, ensure_ascii=False, indent=2)
        print(f"[INFO] summary written to {out_path}")


if __name__ == "__main__":
    main()
