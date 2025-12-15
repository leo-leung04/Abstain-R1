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
training data for small language models on general knowledge question answering
with abstention and clarification.

You will be given:
- A [Question], which is a natural language question covering various topics
  including science, history, geography, culture, and general knowledge.
- A label indicating that this question is UNANSWERABLE from the given
  information or contains logical contradictions or factual errors.

Your goal is to:
1) Reason carefully about why the question CANNOT be answered correctly
   from the provided information, and
2) Produce a user-facing response that:
   - Explicitly abstains from answering, and
   - Briefly explains what is missing, contradictory, or incorrect.

You MUST NOT hallucinate a concrete factual answer.

--------------------------------------------------
DOMAIN & DATASET BEHAVIOR (QAQA-STYLE GENERAL KNOWLEDGE QA)
--------------------------------------------------
1. The QAQA dataset contains general knowledge questions that may be
   unanswerable due to:
   - Factual errors or contradictions in the question itself
   - Logical inconsistencies (e.g., asking about properties that don't exist)
   - Missing critical information needed to answer
   - Ambiguous or underspecified questions
   - Questions about topics that are not well-established or verifiable

2. When judging answerability, you MUST:
   - Carefully analyze the question for factual errors or contradictions
   - Identify what information would be needed to answer the question
   - Check if the question contains logical inconsistencies
   - Determine if the question is ambiguous or underspecified
   - If the question contains errors, contradictions, or lacks necessary
     information, you must abstain

3. Common issues in unanswerable questions:
   - Questions that assume incorrect facts
   - Questions with logical contradictions
   - Questions that are fundamentally unanswerable due to missing context
   - Questions about topics that are not verifiable or well-established
   - Ambiguous questions that could have multiple interpretations

--------------------------------------------------
ABSTENTION & CLARIFICATION BEHAVIOR
--------------------------------------------------
For this task, the question is GUARANTEED to be UNANSWERABLE from the
given information. Therefore, you must:

1. In your reasoning (<thinking>):
   - Restate what the question is asking in your own words.
   - Identify any factual errors, contradictions, or logical inconsistencies
     in the question.
   - Determine what information would be needed to answer the question
     correctly, if it were answerable.
   - Explain precisely why the question cannot be answered: is it due to
     factual errors, contradictions, missing information, or ambiguity?
   - Optionally, state what would need to be corrected or added to make
     the question answerable.

2. In your final response (<answer>):
   - You MUST begin with exactly:
     \\boxed{I don't know.}
   - After that, give a brief, user-facing clarification in natural
     language, either:
       (a) Explaining why the question contains errors or contradictions,
       (b) Explaining what information is missing, or
       (c) Pointing out the logical inconsistency in the question.
   - Do NOT invent or guess a factual answer.
   - Do NOT contradict the abstention (you must not later commit to a
     specific answer).

--------------------------------------------------
CRITICAL REASONING GUIDELINES
--------------------------------------------------
1. No Hindsight Bias:
   - Reason forward from the [Question].
   - Do NOT act as if you magically know a hidden correct answer.
   - Your reasoning should show how you discovered that the question
     is unanswerable.

2. Evidence-Centered Analysis:
   - Make it clear what specific error, contradiction, or missing information
     you identified in the question.
   - For factual errors, explain what the correct fact is and why the
     question's assumption is wrong.
   - For missing information, specify what would be needed.

3. Granularity:
   - Avoid big jumps like "there is an error, so I abstain".
   - Instead, show the intermediate checks: what the question assumes,
     why that assumption is wrong or insufficient, and what would be needed.

4. Self-Checking:
   - Before concluding, add a short internal check, such as:
     "I have identified that this question contains [error/contradiction/
     missing information], so the question remains unanswerable."

5. Format Compliance:
   - Do NOT repeat the full [Question] verbatim.
   - Do NOT output any concrete factual answer.
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
Begin by clarifying what the user's question is asking.
Then systematically analyze the question for errors, contradictions,
or missing information.
Explain, step by step, why the question cannot be answered correctly.
Optionally, say what would need to be corrected or added to make the
question answerable.
Conclude that you must abstain.
</thinking>
<answer>
\\boxed{I don't know.}
[Write a short, user-facing explanation (1–3 sentences) that either:
 - explains the error or contradiction in the question, or
 - explains what information is missing, or
 - points out the logical inconsistency.]
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
        description="Clean QAQA unanswerable data and generate DeepSeek abstention outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "qaqa_unanswerable_100_forward.json"),
        help="Path to the input JSON with unanswerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "qaqa_unanswerable_clean_deepseek.jsonl"),
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





