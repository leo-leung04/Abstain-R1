"""
使用 math-verify 库验证 LLM 生成的数学表达式是否与真值等价。

主要功能：
- 从 <answer>...</answer> 标签中提取答案
- 使用 math-verify 的 parse 和 verify 函数验证答案与 ground_truth 是否等价
- 仅针对可回答问题（answerable questions）进行验证
- 生成 summary JSON 和详细结果的 jsonl 文件

Usage:
  python run/evaluate_with_math_verify.py \
    --reference data/sum_test_with_clarification_retry.jsonl \
    --files results/qwen25_3b_instruct_result.jsonl [...]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("[ERROR] math-verify library not installed. Please install it with:")
    print("  pip install math-verify[antlr4_13_2]")
    sys.exit(1)


def load_reference(path: Path) -> Dict[str, Dict]:
    """加载参考文件，以 answerable_question 为键."""
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
    """如果文本开头没有 <thinking> 标签，则在开头添加."""
    if not isinstance(text, str):
        return text
    # 检查文本是否以 <thinking> 开头（不区分大小写，去除首尾空白后）
    text_stripped = text.strip()
    if text_stripped.lower().startswith("<thinking>"):
        return text
    # 如果没有，则在开头添加 <thinking> 标签
    return f"<thinking>\n{text}"


def extract_answer(text: str) -> Optional[str]:
    """从 <answer>...</answer> 标签中提取内容."""
    if not isinstance(text, str):
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


IDK_PATTERNS = [
    r"\\boxed\s*{\s*I don't know\.?\s*}",
    r"I don't know",
]


def is_abstain_answer(answer_text: str) -> bool:
    """检测 answer tag 中的内容是否包含 'I don't know'."""
    if not isinstance(answer_text, str):
        return False
    for pat in IDK_PATTERNS:
        if re.search(pat, answer_text, re.IGNORECASE):
            return True
    return False


def has_required_tags(text: str) -> bool:
    """检查文本是否包含正确格式的 <thinking> 和 <answer> 标签."""
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


def wrap_boxed_if_needed(text: str) -> str:
    """如果文本没有 \boxed{}，则包裹它."""
    if not text:
        return text
    # 检查是否已经包含 \boxed
    if "\\boxed" in text:
        return text
    # 如果没有，则包裹
    return f"\\boxed{{{text}}}"


def verify_with_math_verify(answer_text: str, ground_truth: str) -> Tuple[bool, Optional[str]]:
    """
    使用 math-verify 验证答案是否与 ground_truth 等价.
    
    返回: (is_correct, error_message)
    """
    if not answer_text or not ground_truth:
        return False, "Empty answer or ground_truth"
    
    try:
        # 给 ground_truth 包裹 \boxed{}（如果还没有）
        ground_truth_wrapped = wrap_boxed_if_needed(ground_truth)
        
        # 解析答案和真值
        answer_parsed = parse(answer_text, raise_on_error=False)
        truth_parsed = parse(ground_truth_wrapped, raise_on_error=False)
        
        # 检查解析是否成功
        if not answer_parsed or len(answer_parsed) == 0:
            return False, f"Failed to parse answer: {answer_text[:100]}"
        
        if not truth_parsed or len(truth_parsed) == 0:
            return False, f"Failed to parse ground_truth: {ground_truth[:100]}"
        
        # 如果解析出多个表达式，尝试匹配任何一个
        # 通常答案应该只有一个表达式，但为了健壮性，我们检查所有可能的匹配
        # print("==========================") 
        # print(answer_parsed)
        # print(truth_parsed)
        # print("==========================")
        for ans_expr in answer_parsed:
            for truth_expr in truth_parsed:
                try:
                    # print(ans_expr, truth_expr)
                    is_equivalent = verify(ans_expr, truth_expr, raise_on_error=False)
                    if is_equivalent:
                        return True, None
                except Exception as e:
                    # 如果验证过程出错，继续尝试下一个组合
                    continue
        # is_equivalent = verify(answer_text, ground_truth_wrapped, raise_on_error=False)
        return False, "Expressions are not equivalent"
        
    except Exception as e:
        return False, f"Error during verification: {str(e)}"


@dataclass
class AnswerableMetrics:
    total: int = 0
    format_ok: int = 0
    correct_math_verify: int = 0
    parse_error_answer: int = 0  # 无法解析答案
    parse_error_truth: int = 0  # 无法解析真值
    verification_error: int = 0  # 验证过程出错
    abstain: int = 0  # 拒答（在 answer tag 中检测到 "I don't know" 等）


@dataclass
class FileMetrics:
    answerable: AnswerableMetrics = field(default_factory=AnswerableMetrics)
    missing_reference: int = 0
    invalid_json: int = 0


def evaluate_file(
    path: Path,
    reference: Dict[str, Dict],
    show_progress: bool = False,
) -> Tuple[FileMetrics, List[Dict[str, Any]]]:
    """评估文件，返回指标和详细注释."""
    metrics = FileMetrics()
    annotations: List[Dict[str, Any]] = []
    total_lines = 0
    
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            
            if show_progress and total_lines > 0 and idx % 50 == 0:
                print(f"    [{path.name}] processed {idx}/{total_lines}", flush=True)
            
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                metrics.invalid_json += 1
                continue
            
            annotation = _process_record(rec, reference, idx)
            annotations.append(annotation)
            
            # 更新指标
            metrics.answerable.total += 1
            if annotation.get("format_ok"):
                metrics.answerable.format_ok += 1
            if annotation.get("math_verify_correct"):
                metrics.answerable.correct_math_verify += 1
            if annotation.get("parse_error_answer"):
                metrics.answerable.parse_error_answer += 1
            if annotation.get("parse_error_truth"):
                metrics.answerable.parse_error_truth += 1
            if annotation.get("verification_error"):
                metrics.answerable.verification_error += 1
            if annotation.get("abstain"):
                metrics.answerable.abstain += 1
            if annotation.get("missing_reference"):
                metrics.missing_reference += 1
    
    return metrics, annotations


def _process_record(
    rec: Dict[str, Any],
    reference: Dict[str, Dict],
    line_no: Optional[int] = None,
) -> Dict[str, Any]:
    """处理单条记录."""
    annotation: Dict[str, Any] = {
        "line": line_no,
        "answerable_question": rec.get("answerable_question"),
        "ground_truth": None,
        "answerable_output": rec.get("answerable_output", ""),
        "format_ok": False,
        "extracted_answer": None,
        "abstain": False,  # 是否拒答
        "math_verify_correct": False,
        "parse_error_answer": False,
        "parse_error_truth": False,
        "verification_error": False,
        "error_message": None,
        "missing_reference": False,
    }
    
    ans_q = annotation["answerable_question"]
    # 在处理前添加 <thinking> 标签（如果缺失）
    ans_out = add_thinking_tag_if_missing(annotation["answerable_output"])
    
    # 检查参考数据
    ref_rec = reference.get(ans_q)
    if ref_rec is None:
        annotation["missing_reference"] = True
        return annotation
    
    annotation["ground_truth"] = ref_rec.get("ground_truth")
    
    # 检查格式
    ans_fmt_ok = has_required_tags(ans_out)
    annotation["format_ok"] = ans_fmt_ok
    
    if not ans_fmt_ok:
        return annotation
    
    # 提取答案
    ans_text = extract_answer(ans_out)
    annotation["extracted_answer"] = ans_text
    
    if not ans_text:
        return annotation
    
    # 检测是否拒答（即使是对可回答问题的拒答）
    is_abstain = is_abstain_answer(ans_text)
    annotation["abstain"] = is_abstain
    
    # 如果检测到拒答，仍然可以继续验证（但标记为 abstain）
    # 使用 math-verify 验证
    ground_truth = ref_rec.get("ground_truth", "")
    is_correct, error_msg = verify_with_math_verify(ans_text, ground_truth)
    
    annotation["math_verify_correct"] = is_correct
    annotation["error_message"] = error_msg
    
    # 记录错误类型
    if error_msg:
        if "Failed to parse answer" in error_msg:
            annotation["parse_error_answer"] = True
        elif "Failed to parse ground_truth" in error_msg:
            annotation["parse_error_truth"] = True
        else:
            annotation["verification_error"] = True
    
    return annotation


def format_rate(numerator: int, denominator: int) -> str:
    """格式化比率."""
    if denominator == 0:
        return "0/0 (0.00%)"
    return f"{numerator}/{denominator} ({numerator/denominator:.2%})"


def summarize(path: Path, metrics: FileMetrics) -> str:
    """生成摘要文本."""
    lines = [f"{path}"]
    a = metrics.answerable
    lines.append(f"  answerable total: {a.total}")
    lines.append(f"  answerable format_ok: {format_rate(a.format_ok, a.total)}")
    lines.append(f"  answerable abstain (拒答率): {format_rate(a.abstain, a.total)}")
    lines.append(f"  answerable correct (math-verify): {format_rate(a.correct_math_verify, a.total)}")
    lines.append(f"  answerable parse_error_answer: {format_rate(a.parse_error_answer, a.total)}")
    lines.append(f"  answerable parse_error_truth: {format_rate(a.parse_error_truth, a.total)}")
    lines.append(f"  answerable verification_error: {format_rate(a.verification_error, a.total)}")
    if metrics.invalid_json:
        lines.append(f"  invalid JSON lines: {metrics.invalid_json}")
    if metrics.missing_reference:
        lines.append(f"  missing reference matches: {metrics.missing_reference}")
    return "\n".join(lines)


def write_annotations(path: Path, annotations: List[Dict[str, Any]]):
    """写入详细注释到 jsonl 文件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for ann in annotations:
            fout.write(json.dumps(ann, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="使用 math-verify 库验证数学表达式是否与真值等价"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/sum/sum_test_with_clarification_retry.jsonl",
        help="参考 jsonl 文件路径 (default: data/sum_test_with_clarification_retry.jsonl)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        required=True,
        help="要评估的结果文件列表",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 summary JSON 文件路径 (default: 自动生成)",
    )
    parser.add_argument(
        "--annotate-dir",
        type=str,
        default=None,
        help="存储详细注释 jsonl 文件的目录 (default: 自动生成)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="不写入详细注释 jsonl 文件",
    )
    args = parser.parse_args()
    
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}")
        sys.exit(1)
    
    reference = load_reference(ref_path)
    print(f"[INFO] Loaded {len(reference)} reference records")
    
    targets = [Path(f) for f in args.files]
    
    annotate_enabled = not args.no_annotate
    
    # 自动生成输出路径
    auto_output_path = None
    if not args.output and targets:
        first_file = targets[0]
        stem = first_file.stem
        eval_dir = first_file.parent / "math_verify_eval"
        auto_output_path = eval_dir / f"{stem}_math_verify_summary.json"
        print(f"[INFO] Auto-generated output path: {auto_output_path}")
    
    # 自动生成 annotate_dir
    auto_annotate_dir = None
    if annotate_enabled and not args.annotate_dir and targets:
        first_file = targets[0]
        auto_annotate_dir = str(first_file.parent / "math_verify_eval")
        print(f"[INFO] Auto-generated annotate_dir: {auto_annotate_dir}")
    
    results_summary: Dict[str, Any] = {}
    
    for i, path in enumerate(targets, start=1):
        print(f"[FILE {i}/{len(targets)}] {path}")
        if not path.exists():
            print(f"[WARN] skip missing file: {path}")
            continue
        
        metrics, annotations = evaluate_file(
            path,
            reference,
            show_progress=True,
        )
        print(summarize(path, metrics))
        print()
        
        results_summary[str(path)] = asdict(metrics)
        
        if annotate_enabled:
            ann_dir = Path(args.annotate_dir or auto_annotate_dir)
            ann_path = ann_dir / f"{path.stem}_math_verify_eval.jsonl"
            write_annotations(ann_path, annotations)
            print(f"  [INFO] annotations written to {ann_path}")
    
    # 写入 summary
    final_output_path = args.output or auto_output_path
    if final_output_path:
        out_path = Path(final_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fout:
            json.dump(results_summary, fout, ensure_ascii=False, indent=2)
        print(f"[INFO] summary written to {out_path}")


if __name__ == "__main__":
    main()
