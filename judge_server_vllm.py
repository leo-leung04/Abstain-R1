# judge_server_vllm.py
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import re

# ------------ 1. 全局加载 vLLM 模型，只加载一次 ------------

MODEL_NAME = "/workspace/models/general_verifier"

print(">>> [INIT] loading vLLM model:", MODEL_NAME)

# 注意：哪块 GPU 是由 CUDA_VISIBLE_DEVICES 决定的
llm = LLM(
    model="/workspace/models/general_verifier",
    tokenizer=MODEL_NAME,
    dtype="float16",          # 小模型 float16 足够
    tensor_parallel_size=1,   # 小模型不需要 tensor parallel，1 就行
    gpu_memory_utilization=0.3,
)

# 固定一个采样配置：deterministic + 最多生成 256 token
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    top_p=1.0,
    n=1,
)

print(">>> [INIT] model loaded and ready.")

# ------------ 2. FastAPI 定义 ------------

app = FastAPI()

class JudgeRequest(BaseModel):
    question: str
    reference_clarification: str
    model_answer: str

class JudgeResponse(BaseModel):
    clarification_ok: int
    raw_output: str


# ------------ 3. prompt 构造：对齐 general-verifier 官方用法 ------------

def build_prompt(question: str,
                 reference_clarification: str,
                 model_answer: str) -> str:
    """
    用于 xVerify-3B-Ia 的 prompt 构造：
    - 把“澄清是否匹配”改写成一个 QA 任务：
      Question = 这道题为什么不可答/病态？
      Correct answer = reference_clarification
      Output sentence = student_clarification
    """

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
    

# ------------ 4. 解析 Final Decision: Yes / No ------------

def parse_final_decision(text: str) -> int:
    lower = text.lower()
    if "correct" in lower and "incorrect" not in lower:
        return 1
    if "incorrect" in lower and "correct" not in lower:
        return 0
    # 有时候会输出像 "[Correct]" 或 "Judgement: [Correct]"
    if "[correct]" in lower and "[incorrect]" not in lower:
        return 1
    if "[incorrect]" in lower and "[correct]" not in lower:
        return 0
    # 模糊就当 0
    return 0

# ------------ 5. HTTP 接口：/judge ------------

@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest):
    prompt = build_prompt(
        req.question,
        req.reference_clarification,
        req.model_answer,
    )

    # vLLM 一次可以 batch 多个 prompt，这里简单写成单元素 list
    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
    )

    # outputs 是一个 list，长度 = prompt 数量；每个元素里还有多个候选（这里 n=1）
    out = outputs[0].outputs[0].text  # 第一条请求的第一条生成

    clar_ok = parse_final_decision(out)
    return JudgeResponse(clarification_ok=clar_ok, raw_output=out)


@app.get("/health")
def health():
    return {"status": "ok"}
