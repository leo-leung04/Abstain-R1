import json
import random
from pathlib import Path

import pandas as pd


INSTRUCTION_TEMPLATE = (
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
    "If you find the question is unanswerable, reply with \\boxed{{I don't know.}} and then ask the user for the necessary information by phrasing the request as a question, "
    "or explain why you cannot answer it.\n"
    "</answer>\n"
    "Let's think step by step, <thinking>"
)


def jsonl_to_parquet(in_path, out_path):

    if not in_path.exists():
        print(f"Skip {in_path} (not found)")
        return 0

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            answerable_q = obj.get("answerable_question")
            ground_truth = obj.get("ground_truth")
            if answerable_q is None or ground_truth is None:
                # Skip malformed records
                continue

            row = {
                "data_source": "sum_math",
                "prompt":[{
                    "role": "user",
                    "content": INSTRUCTION_TEMPLATE.format(question=answerable_q)
                }],
                "reward_model": {
                    "style": "rule",
                    # Wrap ground_truth in \boxed{} if not already present
                    "ground_truth": f"\\boxed{{{ground_truth}}}" if "\\boxed{" not in ground_truth else ground_truth,
                    "clarification": obj.get("clarification"),
                },
                # Optional metadata for analysis
                "answerable_question": answerable_q,
                "unanswerable_question": obj.get("unanswerable_question"),
                "clarification": obj.get("clarification"),
            }

            rows.append(
                {
                    **row,
                }
            )

    if not rows:
        print(f"No data found in {in_path}")
        return 0

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    return len(df)


def jsonl_to_parquet_mixed_answerable_unanswerable(
    in_path,
    out_path,
    answerable_ratio=0.7,
    seed=42,
):
    
    if not in_path.exists():
        print(f"Skip {in_path} (not found)")
        return 0

    answerable_rows = []
    unanswerable_rows = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            answerable_q = obj.get("answerable_question")
            unanswerable_q = obj.get("unanswerable_question")
            ground_truth = obj.get("ground_truth")
            if ground_truth is None:
                continue

            # Answerable variant
            if answerable_q:
                row_ans = {
                    "data_source": "sum_math",
                    "prompt": [
                        {
                            "role": "user",
                            "content": INSTRUCTION_TEMPLATE.format(question=answerable_q),
                        }
                    ],
                    "reward_model": {
                        "style": "rule",
                        # Wrap ground_truth in \boxed{} if not already present
                        "ground_truth": f"\\boxed{{{ground_truth}}}" if "\\boxed{" not in ground_truth else ground_truth,
                    },
                    "extra_info": {
                        "is_answerable": True,
                    },
                    "answerable_question": answerable_q,
                    "unanswerable_question": unanswerable_q,
                    "clarification": obj.get("clarification"),
                }
                answerable_rows.append(row_ans)

            # Unanswerable variant
            if unanswerable_q:
                row_unans = {
                    "data_source": "sum_math",
                    "prompt": [
                        {
                            "role": "user",
                            "content": INSTRUCTION_TEMPLATE.format(question=unanswerable_q),
                        }
                    ],
                    "reward_model": {
                        "style": "rule",
                        # For unanswerable questions we do not rely on ground_truth;
                        # keep a placeholder to satisfy the schema.
                        "ground_truth": ground_truth,
                    },
                    "extra_info": {
                        "is_answerable": False,
                    },
                    "answerable_question": answerable_q,
                    "unanswerable_question": unanswerable_q,
                    "clarification": obj.get("clarification"),
                }
                unanswerable_rows.append(row_unans)

    if not answerable_rows or not unanswerable_rows:
        print(
            f"Not enough data to build mixed dataset from {in_path} "
            f"(answerable={len(answerable_rows)}, unanswerable={len(unanswerable_rows)}); skip."
        )
        return 0

    rng = random.Random(seed)
    rng.shuffle(answerable_rows)
    rng.shuffle(unanswerable_rows)

    # Compute how many samples we can take while respecting the desired ratio.
    total_by_ans = int(len(answerable_rows) / max(answerable_ratio, 1e-6))
    total_by_unans = int(len(unanswerable_rows) / max(1.0 - answerable_ratio, 1e-6))
    max_total = min(total_by_ans, total_by_unans)
    if max_total <= 0:
        print(f"Cannot build mixed dataset from {in_path}: computed max_total={max_total}")
        return 0

    n_answerable = int(round(max_total * answerable_ratio))
    n_unanswerable = max_total - n_answerable

    answerable_subset = answerable_rows[:n_answerable]
    unanswerable_subset = unanswerable_rows[:n_unanswerable]

    mixed_rows = answerable_subset + unanswerable_subset
    rng.shuffle(mixed_rows)

    df = pd.DataFrame(mixed_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(
        f"Saved {len(df)} mixed rows to {out_path} "
        f"(answerable={n_answerable}, unanswerable={n_unanswerable})"
    )
    return len(df)


def main():
    base = Path(__file__).resolve().parent

    # Answerable questions only
    pairs = [
        (
            base / "sum_train.jsonl",
            base / "sum_train_verl_rl.parquet",
        ),
        (
            base / "sum_test.jsonl",
            base / "sum_test_verl_rl.parquet",
        ),
    ]

    total = 0
    for in_path, out_path in pairs:
        total += jsonl_to_parquet(in_path, out_path)

    # 70% answerable, 30% unanswerable
    mixed_out = base / "sum_train_verl_rl_mixed_70_30.parquet"
    total_mixed = jsonl_to_parquet_mixed_answerable_unanswerable(
        base / "sum_train_with_clarification.jsonl",
        # base / "sum_train.jsonl",
        mixed_out,
        answerable_ratio=0.7,
        seed=42,
    )

if __name__ == "__main__":
    main()


