# judge_client_pool.py
import itertools
import requests

# 4 FastAPI + vLLM Instance
ENDPOINTS = [
    "http://localhost:8001/judge",
    "http://localhost:8002/judge",
    "http://localhost:8003/judge",
    "http://localhost:8004/judge",
]

_rr = itertools.cycle(range(len(ENDPOINTS)))

def _get_endpoint():
    idx = next(_rr)
    return ENDPOINTS[idx]

def judge_clarification_pool(question: str,
                             reference_clarification: str,
                             model_answer: str) -> tuple[int, str]:
    url = _get_endpoint()
    payload = {
        "question": question,
        "reference_clarification": reference_clarification,
        "model_answer": model_answer,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return int(data["clarification_ok"]), data["raw_output"]


if __name__ == "__main__":
    q = "Find the sum of all angles x that satisfy sin^5 x - cos^5 x = 1/cos x - 1/sin x."
    ref = (
        "The question is unanswerable because the range [0°, 360°] is omitted, "
        "so there can be infinitely many solutions and the sum is not uniquely defined."
    )
    ans = (
        "\boxed{I don't know.} The problem does not specify the interval for x, "
        "so there might be infinitely many solutions and the sum is not well-defined."
    )

    clar_ok, raw = judge_clarification_pool(q, ref, ans)
    print("clarification_ok:", clar_ok)
    print("raw_output:", raw)
