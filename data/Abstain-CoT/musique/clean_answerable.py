import argparse
import concurrent.futures as futures
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Please install `openai>=1.0.0` to call the DeepSeek API."
    ) from exc
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


BASE_DIR = Path(__file__).resolve().parent


PROMPT_TEMPLATE = """You are an expert reasoning engine tasked with generating high-quality
training data for small language models on multi-paragraph reading comprehension.

You will be given:
- A [Question], which includes:
  - Multiple paragraphs of context (Paragraph 1, Paragraph 2, etc.)
  - A question that requires information from one or more paragraphs to answer
- A [Ground Truth Answer], which is the correct answer extracted from the context.

Your objective is to generate a clear, step-by-step Chain of Thought (CoT)
that logically leads from the context paragraphs to the correct answer.

You MUST behave as if you are solving the problem from scratch and MUST
NOT reveal or hint that you were given the answer.

--------------------------------------------------
DOMAIN & DATASET BEHAVIOR (MUSIQUE-STYLE MULTI-PARAGRAPH READING COMPREHENSION)
--------------------------------------------------
1. The MuSiQue dataset contains reading comprehension questions that:
   - Provide multiple paragraphs of context (typically 2-4 paragraphs)
   - Require extracting information from one or more paragraphs to answer
   - May involve multi-hop reasoning across different paragraphs
   - Test the ability to locate and synthesize information from multiple sources

2. When reasoning, you MUST:
   - Read each paragraph carefully and identify key information
   - Determine which paragraph(s) contain the information needed to answer the question
   - For multi-hop questions, identify how information from different paragraphs connects
   - Extract the relevant information accurately from the context
   - Synthesize information from multiple paragraphs if needed

3. Reading comprehension approach:
   - Identify what the question is asking for
   - Scan through each paragraph to locate relevant information
   - For questions requiring information from multiple paragraphs, trace the connections
   - Extract the answer directly from the context, ensuring accuracy
   - Verify that the answer is supported by the provided context

--------------------------------------------------
CRITICAL REASONING GUIDELINES
--------------------------------------------------
1. No Hindsight Bias:
   - Reason forward from the [Question] and context paragraphs.
   - Do NOT say things like "since the correct answer is {ground_truth}",
     or "working backwards from the answer...".
   - The CoT should look like a natural reading comprehension process.

2. Evidence-Centered Reasoning:
   - Start by summarizing what the question is asking.
   - Systematically go through each paragraph and identify relevant information.
   - For multi-hop questions, explain how information from different paragraphs connects.
   - Quote or paraphrase the specific text from the context that supports the answer.
   - Clearly connect the extracted information to the final answer.

3. Granularity:
   - Avoid big logical jumps. Break the reasoning into small, checkable steps.
   - For each paragraph, identify what information it contains.
   - For multi-paragraph questions, make it clear how information from different
     paragraphs is combined.
   - Show the specific text or phrases from the context that lead to the answer.

4. Self-Checking:
   - Before concluding, verify that:
     - The answer is directly supported by the context
     - All relevant paragraphs have been considered
     - The answer accurately reflects the information in the context
   - Add a brief check such as "This answer is correct because it is stated in
     Paragraph X that...".

5. Format Compliance:
   - Do NOT repeat the full original [Question] verbatim.
   - Do NOT change the final answer: it MUST exactly match [Ground Truth Answer].
   - Do NOT introduce alternative final answers or extra options.
   - Strictly follow the output structure below.

--------------------------------------------------
Input Data:
[Question]:
{question}

[Ground Truth Answer]:
{ground_truth}

--------------------------------------------------
Output Structure:
<thinking>
Begin by clarifying what the question is asking.
Then, step by step, go through each paragraph, identify relevant information,
and explain how you extract or synthesize the answer from the context.
For multi-paragraph questions, show how information from different paragraphs connects.
Finish by verifying that the answer is supported by the context.
</thinking>
<answer>
{ground_truth}
</answer>
"""


def strip_answer_suffix(question: str) -> str:
    cleaned = question.rstrip()
    cleaned = re.sub(r"\s*Answer:\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def load_answerable(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split_name = next(iter(data))
    return data[split_name]["answerable"]


def build_prompt(question: str, ground_truth: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
    )


def call_deepseek(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_single(
    client: OpenAI,
    model: str,
    temperature: float,
    question: str,
    ground_truth: str,
) -> Dict[str, str]:
    prompt = build_prompt(question, ground_truth)
    answer = call_deepseek(client, model, prompt, temperature)
    return {"question": question, "answer": answer}


def iter_items(
    raw_items: List[Dict[str, Any]],
    limit: int | None,
) -> Iterable[Tuple[int, str, str]]:
    count = len(raw_items) if limit is None else min(limit, len(raw_items))
    for idx in range(count):
        item = raw_items[idx]
        raw_question = item["question"]
        cleaned_question = strip_answer_suffix(raw_question)
        refs = item.get("reference_answers") or []
        if not refs:
            continue
        ground_truth = refs[0]
        yield idx, cleaned_question, ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean MuSiQue answerable data and generate DeepSeek CoT outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "musique_answerable_100_forward.json"),
        help="Path to the input JSON with answerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "musique_answerable_clean_deepseek.jsonl"),
        help="Where to write the JSONL results.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var).",
    )
    parser.add_argument(
        "--api-base",
        default="https://api.deepseek.com/v1",
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name, e.g., deepseek-chat (DeepSeek-V3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for parallel generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only process the first N items.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("DeepSeek API key is required (set --api-key or DEEPSEEK_API_KEY).")

    raw_items = load_answerable(args.input_path)
    todo = list(iter_items(raw_items, args.limit))
    if not todo:
        raise SystemExit("No items found to process.")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    total = len(todo)
    progress = tqdm(total=total, desc="deepseek", unit="item") if tqdm else None
    processed = 0
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    generate_single,
                    client,
                    args.model,
                    args.temperature,
                    question,
                    ground_truth,
                ): (idx, question)
                for idx, question, ground_truth in todo
            }

            for future in futures.as_completed(future_map):
                idx, question = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"[WARN] item {idx} failed: {exc}")
                    continue
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1
                if progress:
                    progress.update(1)
                elif processed % 10 == 0 or processed == total:
                    print(f"processed {processed}/{total}")

    if progress:
        progress.close()
    print(f"Finished. Wrote {processed} items to {args.output_path}")


if __name__ == "__main__":
    main()

