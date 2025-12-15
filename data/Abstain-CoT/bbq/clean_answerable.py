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
training data for small language models on social-bias-sensitive
question answering (the BBQ benchmark).

I will provide you with:
- A [Question], which describes a short scenario with two or more people
  and then asks a multiple-choice-style question in natural language.
- A [Ground Truth Answer], which is the correct short answer to the
  question.

Your objective is to generate a clear, step-by-step Chain of Thought (CoT)
that logically leads from the scenario to the correct answer, while
explicitly avoiding social stereotypes.

You MUST behave as if you are solving the problem from scratch and MUST
NOT reveal or hint that you were given the answer.

--------------------------------------------------
DOMAIN & DATASET BEHAVIOR (BBQ-STYLE SOCIAL BIAS QA)
--------------------------------------------------
1. The questions are taken from the Bias Benchmark for Question Answering
   (BBQ), which is designed to test whether models rely on harmful social
   stereotypes rather than the explicit information in the scenario.

2. Each question:
   - Mentions attributes like race, gender, age, SES, disability, etc.
     These are sensitive attributes.
   - Describes a brief situation (who did what, who experienced what).
   - Asks a question such as "Who did X?" or "Who was Y?".

3. When reasoning, you MUST:
   - Base your answer ONLY on the explicit facts given in the scenario.
   - Track which person did which action or had which property.
   - NEVER infer an answer purely from group membership (e.g., race,
     gender, age, SES). Do NOT use social stereotypes to decide.

4. Even if a stereotypical answer might seem plausible, you must ignore
   stereotypes and instead follow the literal information in the text.

--------------------------------------------------
CRITICAL REASONING GUIDELINES
--------------------------------------------------
1. No Hindsight Bias:
   - Do NOT say things like "since the correct answer is {ground_truth}"
     or "working backwards from the answer...".
   - The reasoning must look like a natural forward solution based on the
     scenario.

2. Evidence-Centered Reasoning:
   - Begin by restating in your own words what the question is asking
     (e.g., "The question asks which person spoke calmly").
   - Explicitly identify the key parts of the scenario that determine the
     answer (e.g., "Latoya asked what could be done, which is calm
     behavior; Alejandra yelled at the principal").
   - Clearly connect those textual clues to the final choice.

3. Granularity:
   - Avoid big jumps such as "It must be X" without explanation.
   - Break the reasoning into small, checkable steps:
     - Identify each person and their attributes/actions.
     - Match the question to the relevant actions or properties.
     - Rule out the wrong option(s) explicitly.

4. Self-Checking:
   - Before concluding, briefly verify that no alternative answer also
     fits the scenario and that you did not rely on any stereotype.
   - For example: "There is no statement that the other person did this,
     so only [correct person] fits."

5. Fairness & Safety:
   - Do NOT invent or repeat harmful stereotypes.
   - If a stereotype would favor one option, you must explicitly ignore it
     and rely only on the written description of the situation.

6. Format Compliance:
   - Do NOT repeat the full [Question] verbatim.
   - Do NOT change the final answer: it MUST exactly match the
     [Ground Truth Answer].
   - Do NOT add extra alternative answers.
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
Start by briefly clarifying what the question is asking.
Then, step by step, link the described actions or properties in the scenario
to the options, making it clear why the [Ground Truth Answer] is correct
and the other option(s) are not.
Explicitly avoid using stereotypes and base your conclusion only on the
given text.
Finish with a short internal check that the answer is uniquely supported
by the scenario.
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
        description="Clean BBQ answerable data and generate DeepSeek CoT outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "bbq_answerable_100_forward.json"),
        help="Path to the input JSON with answerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "bbq_answerable_clean_deepseek.jsonl"),
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

