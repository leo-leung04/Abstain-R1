from datasets import load_dataset
import argparse
import json

DATASET_NAME = "known_unknown_questions"


def to_clean_dict(item):
    return {
        "question": item["question"],
        "reference_answers": item["reference_answers"],
        "should_abstain": item["should_abstain"],
        "metadata_json": item["metadata_json"],
    }


def select_examples(direction: str, count: int, subset: str):
    dataset = load_dataset(
        "facebook/AbstentionBench",
        split=DATASET_NAME,
        trust_remote_code=True,
    )

    if subset == "answerable":
        filtered = [row for row in dataset if not row["should_abstain"]]
    else:
        filtered = [row for row in dataset if row["should_abstain"]]

    if direction == "backward":
        filtered = list(reversed(filtered))

    return [to_clean_dict(row) for row in filtered[:count]]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a subset of the AbstentionBench split in the same format "
            "as abstentionbench_samples.json"
        )
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "backward"],
        default="forward",
        help="Take examples from the start (forward) or from the end (backward) of the split",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of examples to extract",
    )
    parser.add_argument(
        "--subset",
        choices=["answerable", "unanswerable"],
        default="answerable",
        help="Extract answerable or unanswerable examples",
    )

    args = parser.parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")

    results = {DATASET_NAME: {"answerable": [], "unanswerable": []}}
    results[DATASET_NAME][args.subset] = select_examples(
        args.direction,
        args.count,
        args.subset,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
