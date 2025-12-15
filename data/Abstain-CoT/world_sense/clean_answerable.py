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
training data for small language models on logical reasoning and world knowledge.

You will be given:
- A [Question], which presents a logical reasoning problem involving:
  - Relationships between entities (e.g., temporal order, spatial relationships, comparisons)
  - Multiple constraints or conditions
  - A question asking you to determine a relationship or make a comparison
- A [Ground Truth Answer], which is the correct answer (typically (1), (2), or (3)).

Your objective is to generate a clear, step-by-step Chain of Thought (CoT)
that logically leads from the given constraints to the correct answer.

You MUST behave as if you are solving the problem from scratch and MUST
NOT reveal or hint that you were given the answer.

--------------------------------------------------
DOMAIN & DATASET BEHAVIOR (WORLD-SENSE-STYLE LOGICAL REASONING)
--------------------------------------------------
1. The World Sense dataset contains logical reasoning problems that:
   - Present relationships between entities (e.g., "A is before B", "X is higher than Y")
   - Require reasoning about temporal order, spatial relationships, or comparisons
   - May involve transitive reasoning (if A < B and B < C, then A < C)
   - Test the ability to combine multiple constraints to reach a conclusion
   - Often have three answer options: (1) one relationship, (2) the opposite relationship,
     or (3) impossible to decide

2. When reasoning, you MUST:
   - Identify all relationships and constraints given in the problem.
   - Build a logical chain connecting the relationships.
   - For transitive relationships, apply logical inference (e.g., if A < B and B < C, then A < C).
   - Determine whether the question can be answered with the given information.
   - If the information is insufficient, recognize that it's impossible to decide.

3. Logical reasoning approach:
   - List all given relationships explicitly.
   - Identify what the question is asking.
   - Build a chain of reasoning connecting the relationships.
   - Check if the chain leads to a definite conclusion or if information is missing.
   - Select the appropriate answer: (1), (2), or (3).

--------------------------------------------------
CRITICAL REASONING GUIDELINES
--------------------------------------------------
1. No Hindsight Bias:
   - Reason forward from the [Question] and given relationships.
   - Do NOT say things like "since the correct answer is {ground_truth}",
     or "working backwards from the answer...".
   - The CoT should look like a natural logical reasoning process.

2. Evidence-Centered Reasoning:
   - Start by listing all relationships given in the problem.
   - Identify what the question is asking.
   - Build a logical chain step by step.
   - For transitive relationships, show how you combine them.
   - Clearly connect the reasoning to the final answer.

3. Granularity:
   - Avoid big logical jumps. Break the reasoning into small, checkable steps.
   - Show how you combine relationships (e.g., "Given that A < B and B < C, I can infer A < C").
   - For "impossible to decide" cases, explain what information is missing.
   - Make each logical step explicit.

4. Self-Checking:
   - Before concluding, verify that:
     - All given relationships have been considered
     - The logical chain is sound
     - The answer directly addresses the question
   - Add a brief check such as "This conclusion is correct because...".

5. Format Compliance:
   - Do NOT repeat the full original [Question] verbatim.
   - Do NOT change the final answer: it MUST exactly match [Ground Truth Answer].
   - Do NOT introduce alternative final answers or extra options.
   - The final answer should be one of: (1), (2), or (3).
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
Begin by listing all relationships given in the problem.
Then, step by step, build a logical chain connecting these relationships
to answer the question.
For transitive reasoning, show how relationships are combined.
Finish by verifying that the answer is correct and well-supported.
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
        description="Clean World Sense answerable data and generate DeepSeek CoT outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "world_sense_answerable_100_forward.json"),
        help="Path to the input JSON with answerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "world_sense_answerable_clean_deepseek.jsonl"),
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

