import argparse
import json
import os
from typing import Optional
import errno

from vllm import LLM, SamplingParams

# 默认的本地缓存目录
DEFAULT_BASE_DIR = "/scratch.global/haoti002/models"

# 可选的模型 key -> HF 模型名
MODELS = {
    "qwen25_3b_instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek_r1_distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen25_math_7b_instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
}

# Prompt 模板，可按参数选择
TEMPLATES = {
    "structured": (
        "Question:\n"
        "{question}\n\n"
        "Respond using the following structure without repeating the original question:\n"
        "<thinking>\n"
        "Reasoning process here\n"
        "</thinking>\n"
        "<answer>\n"
        "Answer here\n"
        "If the question is answerable, provide the final answer.\n"
        "If you don't know the answer or the question is unanswerable, reply with\n"
        "\\boxed{{I don't know.}} and explain why you cannot answer it."
        "</answer>\n"
    ),
    "structured_strict_answer": (
        "Question:\n"
        "{question}\n\n"
        "Respond using the following structure without repeating the original question.\n"
        "The reasoning process must be written inside <thinking> </thinking> tags, and the final answer must be written inside <answer> </answer> tags."
        "\n"
        "You must follow this structure:\n"
        "<thinking>\n"
        "Reasoning process here\n"
        "</thinking>\n"
        "<answer>\n"
        "Final answer here, for example: \\boxed{{42}}.\n"
        "If the question is answerable, provide the final answer wrapped in \\boxed{{}}.\n"
        "If you find the question is unanswerable, reply with \\boxed{{I don't know.}} and then explain why you cannot answer it.\n"
        "</answer>\n"
        "Let's think step by step, <thinking>"
    ),
}

DEFAULT_TEMPLATE_ID = "structured"


def build_prompt(question: str, template_id: str = DEFAULT_TEMPLATE_ID) -> str:
    template = TEMPLATES.get(template_id)
    if template is None:
        raise ValueError(f"Unknown template_id: {template_id}")
    return template.format(question=question)


def load_model(
    model_key: str,
    tensor_parallel_size: int = 1,
    base_dir: str = DEFAULT_BASE_DIR,
    dtype: str = "float16",
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = True,
):
    """
    使用 vLLM 加载模型
    
    Args:
        model_key: 比如 'qwen25_3b_instruct' 或直接 HF 模型名
        tensor_parallel_size: 张量并行大小，即使用的 GPU 数量（默认: 1）
        base_dir: 模型缓存目录
        dtype: 模型数据类型，'float16' 或 'bfloat16'（默认: float16）
        max_model_len: 最大序列长度（默认: None，使用模型默认值）
        gpu_memory_utilization: GPU 内存利用率（默认: 0.9）
        trust_remote_code: 是否信任远程代码（默认: True）
    """
    if model_key in MODELS:
        hf_name = MODELS[model_key]
    else:
        hf_name = model_key

    print(f"[INFO] Loading model with vLLM: {model_key} -> {hf_name}")
    print(f"[INFO] Tensor parallel size: {tensor_parallel_size}")
    print(f"[INFO] Dtype: {dtype}, GPU memory utilization: {gpu_memory_utilization}")

    os.makedirs(base_dir, exist_ok=True)

    # vLLM 的 LLM 类会自动处理 tokenizer
    llm = LLM(
        model=hf_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        download_dir=base_dir,
    )

    return llm


def generate_batch(
    llm,
    questions,
    max_new_tokens: int = 2024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    template_id: str = DEFAULT_TEMPLATE_ID,
    stop: Optional[list] = None,
):
    """
    对一批 question 做一次 batched generation，利用 vLLM 的并行能力。
    返回列表 [str, str, ...]，长度 = len(questions)
    """
    prompts = [build_prompt(q, template_id) for q in questions]

    # 设置停止词：遇到 </answer> 就停止生成
    if stop is None:
        stop = ["</answer>"]

    # 创建采样参数
    # include_stop_str_in_output=True 确保停止词 </answer> 被包含在输出中
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop,
        include_stop_str_in_output=True,  # 包含停止词在输出中
    )

    # vLLM 会自动进行批量推理和并行处理
    outputs = llm.generate(prompts, sampling_params)

    # 提取生成的文本
    texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        texts.append(generated_text.strip())

    return texts


def iter_jsonl(path):
    """简单的 jsonl 迭代器，返回 (idx, record)。"""
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {idx}: JSON decode error, skip. {e}")
                continue
            yield idx, record


def process_file(
    model_key: str,
    input_path: str,
    output_path: str,
    batch_size: int,
    max_new_tokens: int,
    limit: Optional[int] = None,
    tensor_parallel_size: int = 1,
    base_dir: str = DEFAULT_BASE_DIR,
    dtype: str = "float16",
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    temperature: float = 1.0,
    top_p: float = 1.0,
    template_id: str = DEFAULT_TEMPLATE_ID,
    stop: Optional[list] = None,
    question_field: str = "question",
    response_field: str = "response",
):
    """
    统一的入口：
    - 读取包含 question 字段的 jsonl 文件（Abstain 数据集格式）
    - 对每个问题生成答案
    - 输出到 jsonl 文件，添加 response 字段
    """
    llm = load_model(
        model_key,
        tensor_parallel_size=tensor_parallel_size,
        base_dir=base_dir,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fout = open(output_path, "w", encoding="utf-8")

    # 初始化批处理缓冲区和计数器，供内部函数使用
    buffer_records = []
    buffer_questions = []
    total = 0

    def flush_batch():
        nonlocal buffer_records, buffer_questions, total, fout

        if not buffer_records:
            return False

        print(f"[INFO] Running batch of size {len(buffer_records)} ...")

        # 批量生成输出
        outputs = generate_batch(
            llm,
            buffer_questions,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            template_id=template_id,
            stop=stop,
        )

        reached_limit = False

        for rec, output in zip(buffer_records, outputs):
            rec[response_field] = output
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

            if limit is not None and total >= limit:
                reached_limit = True
                break

        # 批处理结束后再统一 flush + fsync 一次
        try:
            fout.flush()
            os.fsync(fout.fileno())
        except OSError as e:
            # 116 一般就是 ESTALE；有的系统 errno.ESTALE 不一定有，就两种都兜一下
            stale_err = getattr(errno, "ESTALE", 116)
            if e.errno == stale_err:
                print(f"[WARN] fsync got ESTALE (stale file handle): {e}. "
                      "Skip hard sync this time, data will still be flushed on close.")
            else:
                raise

        # 清空缓冲
        buffer_records = []
        buffer_questions = []

        return reached_limit

    # 主循环：读取输入文件并填充批处理缓冲区
    for idx, record in iter_jsonl(input_path):
        if limit is not None and total >= limit:
            break

        question = record.get(question_field, "")

        if not question:
            print(
                f"[WARN] Line {idx}: missing '{question_field}' field, skip."
            )
            continue

        if limit is not None:
            remaining = limit - total - len(buffer_records)
            if remaining <= 0:
                break

        buffer_records.append(record)
        buffer_questions.append(question)

        if len(buffer_records) >= batch_size:
            reached_limit = flush_batch()
            if reached_limit:
                break

    # flush 最后一批
    if buffer_records and (limit is None or total < limit):
        flush_batch()

    fout.close()
    print(f"[DONE] Model {model_key}: processed {total} lines, written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single model on Abstain-style test.jsonl with batched generation using vLLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key (e.g., qwen25_3b_instruct) or HF model name."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2024,
        help="Max new tokens to generate for each sample (default: 2024).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/users/9/haoti002/nlpproject/data/abstain/abstain_test.jsonl",
        help="Input jsonl file path (default: test.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output jsonl file path (default: {model}_abstain_result.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N valid records (default: all).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Local cache directory for models (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model data type (default: float16).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model length (default: None, use model default).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, use !=1.0 to enable sampling).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value (default: 1.0).",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=list(TEMPLATES.keys()),
        default="structured_strict_answer",
        help="Prompt template to use (default: structured).",
    )
    parser.add_argument(
        "--stop",
        type=str,
        nargs="+",
        default=None,
        help="Stop strings (default: ['</answer>']).",
    )
    parser.add_argument(
        "--question_field",
        type=str,
        default="question",
        help="Field name for the question in input JSON (default: question).",
    )
    parser.add_argument(
        "--response_field",
        type=str,
        default="response",
        help="Field name for the model response in output JSON (default: response).",
    )
    args = parser.parse_args()

    model_key = args.model
    input_path = args.input
    output_path = args.output or f"{model_key}_abstain_result.jsonl"

    limit = args.limit if args.limit is None or args.limit > 0 else None

    # 设置停止词
    stop = args.stop if args.stop is not None else ["</answer>"]

    process_file(
        model_key,
        input_path,
        output_path,
        args.batch_size,
        args.max_new_tokens,
        limit,
        args.tensor_parallel_size,
        args.base_dir,
        args.dtype,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.temperature,
        args.top_p,
        args.template,
        stop,
        args.question_field,
        args.response_field,
    )


if __name__ == "__main__":
    main()

