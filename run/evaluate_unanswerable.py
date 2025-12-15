"""
使用本地小模型验证 unanswerable 输出的 clarification 是否与参考答案一致。

主要功能：
- 从 <answer>...</answer> 标签中提取答案
- 检查格式是否正确（<thinking> 和 <answer> 标签）
- 检查 answer tag 之间是否有 \boxed{I don't know.}
- 如果有，提取 clarification 部分（去掉 \boxed{I don't know.}）
- 使用本地 judge_server_vllm 服务判断 clarification 是否一致
- 生成 summary JSON 和详细结果的 jsonl 文件

Usage:
  python run/evaluate_unanswerable.py \
    --reference data/sum_test_with_clarification_retry.jsonl \
    --files results/qwen25_3b_instruct_result.jsonl [...]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# 导入 judge_client_pool
sys.path.insert(0, str(Path(__file__).parent.parent))
from judge_client_pool import judge_clarification_pool


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


def has_boxed_idk(text: str) -> bool:
    """检查文本中是否包含 \boxed{I don't know.}"""
    if not isinstance(text, str):
        return False
    # 匹配 \boxed{I don't know.} 或 \boxed{I don't know}（注意可能有或没有句号）
    pattern = r"\\boxed\s*\{\s*I\s+don'?t\s+know\.?\s*\}"
    return bool(re.search(pattern, text, re.IGNORECASE))


def extract_clarification_from_answer(answer_text: str) -> Optional[str]:
    """
    从 answer 文本中提取 clarification 部分。
    如果包含 \boxed{I don't know.}，则去掉它，返回剩余部分作为 clarification。
    """
    if not isinstance(answer_text, str):
        return None
    
    # 去掉 \boxed{I don't know.} 或 \boxed{I don't know}
    pattern = r"\\boxed\s*\{\s*I\s+don'?t\s+know\.?\s*\}"
    clarification = re.sub(pattern, "", answer_text, flags=re.IGNORECASE)
    clarification = clarification.strip()
    
    # 如果去掉后还有内容，返回它；否则返回 None
    return clarification if clarification else None


@dataclass
class UnanswerableMetrics:
    total: int = 0
    format_ok: int = 0
    has_boxed_idk: int = 0
    has_clarification: int = 0  # 去掉 \boxed{I don't know.} 后还有内容
    clarification_ok: int = 0  # 使用模型判断 clarification 是否一致
    judge_error: int = 0  # 调用 judge 服务时出错


@dataclass
class FileMetrics:
    unanswerable: UnanswerableMetrics = field(default_factory=UnanswerableMetrics)
    missing_reference: int = 0
    invalid_json: int = 0


def evaluate_file(
    path: Path,
    reference: Dict[str, Dict],
    show_progress: bool = False,
) -> tuple[FileMetrics, List[Dict[str, Any]]]:
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
            metrics.unanswerable.total += 1
            if annotation.get("format_ok"):
                metrics.unanswerable.format_ok += 1
            if annotation.get("has_boxed_idk"):
                metrics.unanswerable.has_boxed_idk += 1
            if annotation.get("has_clarification"):
                metrics.unanswerable.has_clarification += 1
            if annotation.get("clarification_ok"):
                metrics.unanswerable.clarification_ok += 1
            if annotation.get("judge_error"):
                metrics.unanswerable.judge_error += 1
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
        "unanswerable_question": rec.get("unanswerable_question"),
        "reference_clarification": None,
        "unanswerable_output": rec.get("unanswerable_output", ""),
        "format_ok": False,
        "extracted_answer": None,
        "has_boxed_idk": False,
        "extracted_clarification": None,
        "has_clarification": False,
        "clarification_ok": False,
        "judge_error": False,
        "judge_raw_output": None,
        "error_message": None,
        "missing_reference": False,
    }
    
    ans_q = annotation["answerable_question"]
    unans_q = annotation["unanswerable_question"]
    # 在处理前添加 <thinking> 标签（如果缺失）
    unans_out = add_thinking_tag_if_missing(annotation["unanswerable_output"])
    
    # 检查参考数据
    ref_rec = reference.get(ans_q)
    if ref_rec is None:
        annotation["missing_reference"] = True
        return annotation
    
    annotation["reference_clarification"] = ref_rec.get("clarification")
    
    # 检查格式
    unans_fmt_ok = has_required_tags(unans_out)
    annotation["format_ok"] = unans_fmt_ok
    
    if not unans_fmt_ok:
        return annotation
    
    # 提取答案
    ans_text = extract_answer(unans_out)
    annotation["extracted_answer"] = ans_text
    
    if not ans_text:
        return annotation
    
    # 检查是否有 \boxed{I don't know.}
    has_idk = has_boxed_idk(ans_text)
    annotation["has_boxed_idk"] = has_idk
    
    if not has_idk:
        return annotation
    
    # 提取 clarification（去掉 \boxed{I don't know.}）
    clarification = extract_clarification_from_answer(ans_text)
    annotation["extracted_clarification"] = clarification
    
    if clarification:
        annotation["has_clarification"] = True
        
        # 使用 judge_clarification_pool 判断 clarification 是否一致
        try:
            reference_clarification = ref_rec.get("clarification", "")
            clar_ok, raw_output = judge_clarification_pool(
                question=unans_q,
                reference_clarification=reference_clarification,
                model_answer=clarification,
            )
            annotation["clarification_ok"] = bool(clar_ok)
            annotation["judge_raw_output"] = raw_output
        except Exception as e:
            annotation["judge_error"] = True
            annotation["error_message"] = str(e)
    
    return annotation


def format_rate(numerator: int, denominator: int) -> str:
    """格式化比率."""
    if denominator == 0:
        return "0/0 (0.00%)"
    return f"{numerator}/{denominator} ({numerator/denominator:.2%})"


def summarize(path: Path, metrics: FileMetrics) -> str:
    """生成摘要文本."""
    lines = [f"{path}"]
    u = metrics.unanswerable
    lines.append(f"  unanswerable total: {u.total}")
    lines.append(f"  unanswerable format_ok: {format_rate(u.format_ok, u.total)}")
    lines.append(f"  unanswerable has_boxed_idk: {format_rate(u.has_boxed_idk, u.total)}")
    lines.append(f"  unanswerable has_clarification: {format_rate(u.has_clarification, u.total)}")
    lines.append(f"  unanswerable clarification_ok: {format_rate(u.clarification_ok, u.total)}")
    lines.append(f"  unanswerable judge_error: {format_rate(u.judge_error, u.total)}")
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
        description="使用本地小模型验证 unanswerable 输出的 clarification 是否一致"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/sum/sum_test_with_clarification_retry.jsonl",
        help="参考 jsonl 文件路径 (default: data/sum/sum_test_with_clarification_retry.jsonl)",
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
        eval_dir = first_file.parent / "unanswerable_eval"
        auto_output_path = eval_dir / f"{stem}_unanswerable_summary.json"
        print(f"[INFO] Auto-generated output path: {auto_output_path}")
    
    # 自动生成 annotate_dir
    auto_annotate_dir = None
    if annotate_enabled and not args.annotate_dir and targets:
        first_file = targets[0]
        auto_annotate_dir = str(first_file.parent / "unanswerable_eval")
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
            ann_path = ann_dir / f"{path.stem}_unanswerable_eval.jsonl"
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
