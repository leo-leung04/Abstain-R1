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
training data for small language models on social-bias-sensitive
question answering.

You will be given a [Question] that:
- Describes a short scenario involving two or more people, often with
  attributes like age, race, gender, disability, or socioeconomic status.
- Asks a follow-up question such as "Who did X?" or "Who was Y?".

For this task, you should assume that the safe, correct behavior is to
ABSTAIN from choosing any person as the answer, because the scenario
does not contain enough evidence to justify a choice without relying
on social stereotypes.

You MUST NOT mention that you were told to abstain or that this is a
special benchmark example. Instead, you must behave as if you carefully
analyzed the scenario and discovered by yourself that the question is
not answerable from the given information.

--------------------------------------------------
BEHAVIOR REQUIREMENTS (UNANSWERABLE + FAIRNESS)
--------------------------------------------------
1. Evidence-based reasoning:
   - Treat the scenario text as the only valid source of evidence.
   - Identify each person mentioned and summarize what is actually said
     about them in the text.
   - Determine whether the property asked about (e.g., "who forgets
     things", "who is a poor communicator", "who is careless at work")
     is explicitly linked to any person.

2. Detecting under-specification:
   - If the scenario never states or clearly implies which person has
     the requested property, then the question is under-specified.
   - In that case, any concrete choice would be a guess, not a
     text-based inference.

3. Avoiding stereotypes:
   - Many questions are constructed so that a biased model might pick
     an answer based on a social attribute (e.g., age, race, gender)
     rather than the scenario.
   - You must explicitly recognize that selecting a specific person
     would rely on such a stereotype (for example, assumptions about
     older people, specific racial groups, or other protected groups),
     and that this is not acceptable.

4. Abstention & clarification:
   - In your reasoning (<thinking>), explain step by step:
     - What the question is asking.
     - What information is given about each person.
     - Why none of that information is enough to uniquely answer the
       question.
     - How answering would require relying on a biased assumption
       about one group versus another.
   - In your final response (<answer>):
     - You MUST start with exactly:
       \\boxed{{I don't know.}}
     - Then add a short, user-facing explanation (1–3 sentences) in
       natural language that:
         * states that the scenario does not provide enough information
           to choose an answer, and
         * mentions that guessing would risk relying on stereotypes
           about the groups described in the text.
   - Do NOT name any person as the answer.
   - Do NOT describe specific stereotype content; refer to it only in
     general terms (e.g., "stereotypes about older people").

5. Format and safety:
   - Do NOT repeat the full [Question] verbatim.
   - Do NOT mention datasets, benchmarks, metadata, or labels.
   - Do NOT say "this is unanswerable by construction" or refer to an
     external label.
   - Strictly follow the output structure below.

--------------------------------------------------
Input Data:
[Question]:
{question}

--------------------------------------------------
Output Structure:
<thinking>
Begin by clarifying what the user's question is asking.
Then systematically analyze the scenario and show that the text does not
provide enough information to decide between the people mentioned.
Explain, in general terms, that choosing one person would rely on
stereotypes rather than evidence.
Conclude that you must abstain.
</thinking>

<answer>
\\boxed{{I don't know.}}
[Write a short, user-facing explanation (1–3 sentences) that says the
scenario does not give enough information to answer and that answering
would risk relying on stereotypes about the groups described in the text,
so you prefer not to guess.]
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
    # Use format with named placeholders to avoid issues with braces in question
    return UNANSWERABLE_PROMPT.format(
        question=question,
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
        description="Clean BBQ unanswerable data and generate DeepSeek abstention outputs (JSONL).",
    )
    parser.add_argument(
        "--input-path",
        default=str(BASE_DIR / "bbq_unanswerable_100_forward.json"),
        help="Path to the input JSON with unanswerable entries.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "bbq_unanswerable_clean_deepseek.jsonl"),
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
        default=10,
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

