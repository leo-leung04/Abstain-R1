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


PROMPT_TEMPLATE = """You are an expert reasoning engine tasked with generating high-quality training data
for small language models on biology-style knowledge tasks.

You will be given:
- A [Question], which already includes:
  - Instructions to a biology question-answering system,
  - Structured biological information about an artificial taxon (a “made-up” organism),
  - And a natural-language question to answer.
- A [Ground Truth Answer], which is the correct short answer for this question.

Your goal is to produce a clear, step-by-step Chain of Thought (CoT) that
solves the problem as if you were seeing it for the first time.

You MUST NOT reveal or hint that you were given the answer.

--------------------------------------------------
DOMAIN & DATASET BEHAVIOR (ALCUNA-STYLE BIOLOGY)
--------------------------------------------------
1. The biological information describes an artificial taxon constructed from
   modified attributes and relations. It may conflict with real-world biology.
2. When reasoning, you MUST:
   - Treat the provided properties as ground truth for this question,
     even if they contradict real-world knowledge.
   - Explicitly identify which properties are relevant (e.g., trophic guild,
     photosynthetic pathway, habitat, life cycle, etc.) and restate them
     in your own words before using them.
   - Use general biological knowledge only to interpret these properties
     (e.g., “brown-rot fungus is a decomposer, not a photoautotroph”),
     NOT to override them.

3. Questions are typically short-answer (a few words) and may involve:
   - Direct lookup from a single property (knowledge understanding),
   - Distinguishing this artificial taxon from other entities (knowledge differentiation),
   - Multi-hop reasoning that links several properties or entities together (knowledge association).

--------------------------------------------------
CRITICAL REASONING GUIDELINES
--------------------------------------------------
1. No Hindsight Bias:
   - Reason forward from the information in the [Question].
   - Do NOT say things like “since the correct answer is {ground_truth}”,
     or “working backwards from the answer…”.
   - The CoT should look like a natural solution process.

2. Evidence-Centered Reasoning:
   - Start by summarizing what the question is asking in your own words.
   - Then, carefully read the taxon’s properties and explicitly quote or paraphrase
     the key fields you will use (e.g., trophic guild, photosynthetic pathway).
   - For multi-hop questions, make each hop explicit:
     (a) identify the first relevant fact, 
     (b) derive an intermediate conclusion,
     (c) use it together with another fact, and so on.

3. Granularity:
   - Avoid big logical jumps. Break the reasoning into small, checkable steps.
   - For logical/linguistic questions, clearly track what each phrase or option means
     and why some are ruled out.
   - For biological questions, explain briefly how the properties imply the conclusion
     (e.g., why a certain trophic guild is not a photoautotroph).

4. Self-Checking:
   - Before concluding, add a short internal check such as
     “This is consistent with the given properties because …”.
   - If the reasoning would lead to more than one plausible answer, explain
     why the [Ground Truth Answer] is the best choice under the given data.

5. Format Compliance:
   - Do NOT repeat the full original [Question] verbatim.
   - Do NOT change the final answer: it MUST exactly match [Ground Truth Answer].
   - Do NOT introduce alternative final answers or extra options.
   - Strictly follow the output structure below.

--------------------------------------------------
Input Data:
[Question]:
{question_string}   # this should already include the biology preprompt + properties + user question

[Ground Truth Answer]:
{ground_truth}

--------------------------------------------------
Output Structure:
<thinking>
Begin by clarifying what the question is asking.
Then, step by step, read and interpret the taxon’s properties, selecting the relevant ones.
Use these properties (and only light, supportive biological knowledge) to derive the answer.
Make multi-hop reasoning explicit if needed.
Finish by briefly checking that the derived answer is consistent with all the key properties.
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
        question_string=question,
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
        description="Clean ALCUNA answerable data and generate DeepSeek CoT outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "alcuna_answerable_100_forward.json"),
        help="Path to the input JSON with answerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "alcuna_answerable_clean_deepseek.jsonl"),
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
