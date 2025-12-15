"""
Evaluation script for Abstain-style result files.

Input format (result file):
- question: The original question
- answer: Ground truth answer (null for unanswerable)
- should_abstain: true/false
- clarification: Clarification for unanswerable questions
- response: Model output

Evaluation:
- For answerable (should_abstain=false): LLM judges correctness (no string match)
- For unanswerable (should_abstain=true): LLM judges refusal and clarification quality

Usage:
  python run/evaluate_abstain.py \
    --reference data/Abstain/abstain-Test-withoutsum.jsonl \
    --files results/abstain-test/deepseek_reasoner_result.jsonl [...]
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
    "results/abstain-test/deepseek_reasoner_result.jsonl",
    "results/abstain-test/deepseek_chat_result.jsonl",
    "results/abstain-test/qwen25_3b_instruct_result.jsonl",
    "results/abstain-test/qwen25_7b_instruct_result.jsonl",
    "results/abstain-test/abstain-r1_result.jsonl",
]


IDK_PATTERNS = [
    r"\\boxed\s*{\s*I don't know\.?\s*}",
    r"I don't know",
]


def load_reference(path: Path) -> Dict[str, Dict]:
    """Load reference records keyed by question."""
    ref = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec["question"]
            ref[key] = rec
    return ref


def add_thinking_tag_if_missing(text: str) -> str:
    """Add <thinking> tag at the beginning if text doesn't start with it."""
    if not isinstance(text, str):
        return text
    text_stripped = text.strip()
    if text_stripped.lower().startswith("<thinking>"):
        return text
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
    false_unanswerable: int = 0  # said I don't know on answerable
    correct_llm: int = 0
    incorrect_llm: int = 0
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
        "  - clarification_ok=1 if the stated reason aligns with the reference clarification's missing-information point(s). Otherwise 0. If uncertain, set 0.\n"
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
    dst.answerable.false_unanswerable += src.answerable.false_unanswerable
    dst.answerable.correct_llm += src.answerable.correct_llm
    dst.answerable.incorrect_llm += src.answerable.incorrect_llm
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
        "question": rec.get("question"),
        "should_abstain": rec.get("should_abstain"),
        "ground_truth": None,
        "clarification": None,
        "response": rec.get("response", ""),
        "tags": {},
    }

    question = annotation["question"]
    should_abstain = rec.get("should_abstain", False)
    # Add <thinking> tag if missing before evaluation
    response = add_thinking_tag_if_missing(annotation["response"])

    ref_rec = reference.get(question)
    if ref_rec is None:
        metrics.missing_reference += 1
        annotation["missing_reference"] = True
        return metrics, annotation

    annotation["ground_truth"] = ref_rec.get("answer")
    annotation["clarification"] = ref_rec.get("clarification")

    if not should_abstain:
        # Answerable question
        metrics.answerable.total += 1
        fmt_ok = has_required_tags(response)
        annotation["tags"]["format_ok"] = fmt_ok
        if fmt_ok:
            metrics.answerable.format_ok += 1
        
        # Extract answer content: if format OK, extract from <answer> tag; otherwise use whole response
        ans_text = extract_answer(response) if fmt_ok else response
        
        # Check if model incorrectly said "I don't know"
        ans_idk = has_idk(ans_text or response)
        annotation["tags"]["said_idk"] = ans_idk
        if ans_idk:
            metrics.answerable.false_unanswerable += 1
        
        # LLM classification (always, no string match)
        label = None
        if (llm_client and llm_model) or (llm_fallback_client and llm_fallback_model):
            label = llm_classify_answerable(
                llm_client,
                llm_model,
                question,
                ref_rec["answer"] or "",
                ans_text or response,
                fallback_client=llm_fallback_client,
                fallback_model=llm_fallback_model,
            )
            if label == "correct":
                metrics.answerable.correct_llm += 1
            elif label == "incorrect":
                metrics.answerable.incorrect_llm += 1
            elif label == "idk":
                metrics.answerable.idk_llm += 1
        annotation["tags"]["llm_label"] = label

    else:
        # Unanswerable question
        metrics.unanswerable.total += 1
        fmt_ok = has_required_tags(response)
        annotation["tags"]["format_ok"] = fmt_ok
        if fmt_ok:
            metrics.unanswerable.format_ok += 1
        
        # Extract answer text for evaluation
        ans_text = extract_answer(response) if fmt_ok else response
        
        # Check for "I don't know" pattern
        has_idk_flag = has_idk(ans_text or response)
        annotation["tags"]["has_box_idk"] = has_idk_flag
        if has_idk_flag:
            metrics.unanswerable.has_box_idk += 1
        
        # LLM classification for refusal and clarification
        refusal = None
        clar_ok = None
        if (llm_client and llm_model) or (llm_fallback_client and llm_fallback_model):
            refusal, clar_ok = llm_classify_unanswerable(
                llm_client,
                llm_model,
                question,
                ref_rec.get("clarification", "") or "",
                ans_text or response,
                fallback_client=llm_fallback_client,
                fallback_model=llm_fallback_model,
            )
            if refusal == 1:
                metrics.unanswerable.refusal_llm += 1
            if clar_ok == 1:
                metrics.unanswerable.clarification_llm += 1
        annotation["tags"]["llm_refusal"] = None if refusal is None else bool(refusal)
        annotation["tags"]["llm_clarification_ok"] = None if clar_ok is None else bool(clar_ok)

    return metrics, annotation


def summarize(path: Path, metrics: FileMetrics, llm_enabled: bool) -> str:
    lines = [f"{path}"]
    a = metrics.answerable
    u = metrics.unanswerable
    lines.append(f"  answerable format_ok: {format_rate(a.format_ok, a.total)}")
    lines.append(f"  answerable false-unanswerable (said IDK): {format_rate(a.false_unanswerable, a.total)}")
    if llm_enabled:
        lines.append(f"  answerable correct (LLM): {format_rate(a.correct_llm, a.total)}")
        lines.append(f"  answerable incorrect (LLM): {format_rate(a.incorrect_llm, a.total)}")
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
        return "openai"  # default to openai for o3-mini
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
    parser = argparse.ArgumentParser(description="Evaluate Abstain-style result files with LLM grading.")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/Abstain/abstain-Test-withoutsum.jsonl",
        help="Reference jsonl path",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Result files to evaluate. Default is the curated list in the script.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="o3-mini",
        help="Model for semantic grading (default: o3-mini).",
    )
    parser.add_argument(
        "--llm-fallback-model",
        type=str,
        default=None,
        help="Optional fallback model if primary LLM call fails.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel workers per file (records). Default 12.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write a JSON summary to this path. If not set, auto-generate.",
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
        help="Directory to store per-record annotation jsonl files.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable writing per-record annotation jsonl files.",
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    reference = load_reference(ref_path)
    print(f"[INFO] Loaded {len(reference)} reference records from {ref_path}")

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

