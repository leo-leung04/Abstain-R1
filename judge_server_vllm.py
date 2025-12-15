# judge_server_vllm.py
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import re

# Load the vLLM model

MODEL_NAME = "/workspace/models/general_verifier"

print(">>> [INIT] loading vLLM model:", MODEL_NAME)

# Note: Which GPU is used is determined by CUDA_VISIBLE_DEVICES.
llm = LLM(
    model="/workspace/models/general_verifier",
    tokenizer=MODEL_NAME,
    dtype="float16",          # float16 for small model
    tensor_parallel_size=1,   # tensor parallel，1 is enough
    gpu_memory_utilization=0.3,
)

# Use a fixed sampling configuration: deterministic + generate a maximum of 256 tokens.
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    top_p=1.0,
    n=1,
)

print(">>> [INIT] model loaded and ready.")

# FastAPI

app = FastAPI()

class JudgeRequest(BaseModel):
    question: str
    reference_clarification: str
    model_answer: str

class JudgeResponse(BaseModel):
    clarification_ok: int
    raw_output: str


# Prompt Construction: Aligning with the official usage of general-verifier

def build_prompt(question: str,
                 reference_clarification: str,
                 model_answer: str) -> str:

    meta_question = (
        "The following problem is known to be unanswerable, ill-posed, or logically flawed as stated.\n\n"
        f"Problem:\n{question}\n\n"
        "Question: What is the MAIN reason why this problem cannot be reliably answered as stated?\n"
        "The core reason can be, for example:\n"
        " - some crucial information or conditions are missing or unspecified;\n"
        " - the assumptions or premises are mutually inconsistent or logically contradictory;\n"
        " - the setup is impossible or violates the stated constraints;\n"
        " - key terms or concepts are undefined, ambiguous, or too vague to apply;\n"
        " - the problem is underdetermined or non-unique with the given data;\n"
        " - or any other structural defect in the problem statement that makes a reliable answer impossible.\n"
        "Your answer should clearly describe this main defect in natural language."
    )

    return (
        "You are a diligent and precise assistant tasked with evaluating the correctness of responses. "
        "You will receive a question, an output sentence, and the correct answer. Your task is to determine "
        "if the output sentence accurately answers the question based on the provided correct answer. "
        "Respond with either [Correct] or [Incorrect].\n\n"
        "Special considerations:\n"
        "1. If the output contains multiple statements, compare its final or main statement with the correct answer. "
        "   If the final statement is unclear or does not match the correct answer in meaning, respond with [Incorrect].\n"
        "2. For mathematical or logical problems, if the formats differ but the meanings are equivalent, "
        "   respond with [Correct].\n"
        "3. If the question does not provide options, the output must match the correct answer in content and "
        "   meaning to be considered [Correct].\n\n"
        f'Question: """{meta_question}"""\n'
        f'Output sentence: """{model_answer}"""\n'
        f'Correct answer: """{reference_clarification}"""\n'
        "Judgement:"
    )
    

# Parse Final Decision: Yes / No

def parse_final_decision(text: str) -> int:
    lower = text.lower()
    if "correct" in lower and "incorrect" not in lower:
        return 1
    if "incorrect" in lower and "correct" not in lower:
        return 0
    # Sometimes the output will be something like "[Correct]" or "Judgement: [Correct]".
    if "[correct]" in lower and "[incorrect]" not in lower:
        return 1
    if "[incorrect]" in lower and "[correct]" not in lower:
        return 0
    # Treat ambiguity as 0.
    return 0

# HTTP Interface: /judge

@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    prompt = build_prompt(
        req.question,
        req.reference_clarification,
        req.model_answer,
    )

    # vLLM can process multiple prompts in a single batch; here, it's simply written as a single-element list.
    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
    )

    # `outputs` is a list, with a length equal to the number of prompts; each element contains multiple candidates (here, n=1).
    out = outputs[0].outputs[0].text # The first generated response to the first request

    clar_ok = parse_final_decision(out)
    return JudgeResponse(clarification_ok=clar_ok, raw_output=out)


@app.get("/health")
def health():
    return {"status": "ok"}
