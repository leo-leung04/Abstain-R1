import argparse
import concurrent.futures as futures
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import time
import traceback

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Please install `openai>=1.0.0` to call the DeepSeek API.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


BASE_DIR = Path(__file__).resolve().parent

UNANSWERABLE_PROMPT = """You are an expert reasoning engine tasked with generating high-quality
training data for small language models on logical reasoning and world knowledge
with abstention and clarification.

You will be given:
- A [Question], which presents a logical reasoning problem involving:
  - Relationships between entities (e.g., temporal order, spatial relationships, comparisons)
  - Multiple constraints or conditions
  - A question asking you to determine a relationship or make a comparison
- A label indicating that this question is UNANSWERABLE because the given
  relationships and constraints are insufficient to determine the answer.

Your goal is to:
1) Reason carefully about why the relationship CANNOT be determined from the
   given constraints, and
2) Produce a user-facing response that:
   - Explicitly abstains from answering, and
   - Briefly explains what information is missing or why the constraints
     are insufficient.

You MUST NOT guess or infer relationships not supported by the given constraints.

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

2. When judging answerability, you MUST:
   - Identify all relationships and constraints given in the problem
   - Build a logical chain connecting the relationships
   - For transitive relationships, apply logical inference (e.g., if A < B and B < C, then A < C)
   - Determine whether the question can be answered with the given information
   - If the information is insufficient to establish the relationship, you must abstain

3. Common reasons for unanswerable problems:
   - Missing relationships needed to establish a connection between entities
   - Insufficient constraints to determine the relationship
   - Ambiguous or conflicting information
   - The relationship cannot be inferred through transitive reasoning
   - Critical intermediate relationships are missing

--------------------------------------------------
ABSTENTION & CLARIFICATION BEHAVIOR
--------------------------------------------------
For this task, the problem is GUARANTEED to be UNANSWERABLE from the
given relationships and constraints. Therefore, you must:

1. In your reasoning (<thinking>):
   - Restate what relationship the question is asking you to determine.
   - List all relationships and constraints given in the problem.
   - Attempt to build a logical chain connecting the relationships.
   - Explicitly identify which relationships or constraints are missing
     that prevent determining the answer.
   - Explain why the given information is insufficient to establish
     the relationship.
   - Optionally, state what additional relationships would be required
     to make the problem answerable.

2. In your final response (<answer>):
   - You MUST begin with exactly:
     \\boxed{I don't know.}
   - After that, give a brief, user-facing clarification in natural
     language that:
       (a) Explains that the given relationships are insufficient to
           determine the answer, or
       (b) States what additional relationships or constraints would be
           needed.
   - Do NOT guess or infer relationships not supported by the constraints.
   - Do NOT contradict the abstention (you must not later commit to
     a specific relationship).

--------------------------------------------------
CRITICAL REASONING GUIDELINES
--------------------------------------------------
1. No Hindsight Bias:
   - Reason forward from the [Question] and given relationships.
   - Do NOT act as if you magically know what the relationship should be.
   - Your reasoning should show how you discovered that the relationships
     are insufficient.

2. Evidence-Centered Analysis:
   - Make it clear which relationships are given in the problem.
   - Explicitly identify which relationships are missing.
   - Explain why the missing relationships are necessary to determine
     the answer.
   - Show your attempt to build a logical chain and where it breaks.

3. Granularity:
   - Avoid big jumps like "there is not enough information, so I abstain".
   - Instead, show the intermediate checks: what relationships are given,
     what chain you tried to build, where it breaks, and what's missing.

4. Self-Checking:
   - Before concluding, add a short internal check, such as:
     "I have identified that [specific relationship] is missing, which
     is necessary to connect [entity A] to [entity B], so the problem
     remains unanswerable."

5. Format Compliance:
   - Do NOT repeat the full [Question] verbatim.
   - Do NOT output any relationship not supported by the constraints.
   - Strictly follow the output structure below.

--------------------------------------------------
Input Data:
[Question]:
{question_string}

[Label]:
UNANSWERABLE_FROM_GIVEN_INFORMATION = True

--------------------------------------------------
Output Structure:
<thinking>
Begin by clarifying what relationship the question is asking you to determine.
Then list all relationships and constraints given in the problem.
Attempt to build a logical chain and identify where it breaks.
Explain, step by step, which relationships are missing and why they
prevent determining the answer.
Optionally, say what additional relationships would make the problem answerable.
Conclude that you must abstain.
</thinking>
<answer>
\\boxed{I don't know.}
[Write a short, user-facing explanation (1–3 sentences) that either:
 - explains that the given relationships are insufficient to determine
   the answer, or
 - states what additional relationships or constraints would be needed.]
</answer>
"""


def strip_answer_suffix(question: str) -> str:
    cleaned = question.rstrip()
    cleaned = re.sub(r"\s*Answer:\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def load_unanswerable(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split_name = next(iter(data))
    return data[split_name]["unanswerable"]


def build_prompt(question: str) -> str:
    # Avoid str.format on user text (question contains braces); simple replace is safer.
    return UNANSWERABLE_PROMPT.replace("{question_string}", question, 1)


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
    max_retries: int,
    retry_delay: float,
) -> Dict[str, str]:
    prompt = build_prompt(question)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            answer = call_deepseek(client, model, prompt, temperature)
            return {"question": question, "answer": answer}
        except Exception as exc:  # pragma: no cover - runtime guard
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay * attempt)
    # If still failing, propagate the last exception to be logged by the caller.
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown failure without exception.")


def iter_items(
    raw_items: List[Dict[str, Any]],
    limit: int | None,
) -> Iterable[Tuple[int, str]]:
    count = len(raw_items) if limit is None else min(limit, len(raw_items))
    for idx in range(count):
        item = raw_items[idx]
        raw_question = item["question"]
        cleaned_question = strip_answer_suffix(raw_question)
        yield idx, cleaned_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean World Sense unanswerable data and generate DeepSeek abstention outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "world_sense_unanswerable_100_forward.json"),
        help="Path to the input JSON with unanswerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "world_sense_unanswerable_clean_deepseek.jsonl"),
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
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per item on API errors.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.5,
        help="Base delay (seconds) between retries; multiplied by attempt number.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only process the first N items.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Abort on the first failed item instead of writing a fallback abstention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("DeepSeek API key is required (set --api-key or DEEPSEEK_API_KEY).")

    raw_items = load_unanswerable(args.input_path)
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
                    args.max_retries,
                    args.retry_delay,
                ): (idx, question)
                for idx, question in todo
            }

            for future in futures.as_completed(future_map):
                idx, question = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - runtime guard
                    err_cls = type(exc).__name__
                    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
                    err_code = getattr(exc, "code", None)
                    print(
                        f"[WARN] item {idx} failed: type={err_cls}, status={status}, code={err_code}, message={exc}"
                    )
                    if args.fail_on_error:
                        traceback.print_exc()
                        raise
                    # fallback abstention to avoid losing the item
                    fallback_answer = (
                        "<thinking>Model call failed; abstaining to avoid hallucination.</thinking>\n"
                        "<answer>\n\\boxed{I don't know.}\nI could not complete the request for this question.\n</answer>"
                    )
                    result = {"question": question, "answer": fallback_answer}
                    print(f"[INFO] item {idx}: wrote fallback abstention.")
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

