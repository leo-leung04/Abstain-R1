# Abstain-R1

We fine-tune the Qwen2.5-3B-Instruct model using Group Relative Policy Optimization (GRPO) based on [verl](https://github.com/volcengine/verl) and a composite reward function to optimize the delicate balance between refusal and clarification when facing unanswerable queries, all while ensuring high accuracy on answerable queries.

## 📦 Resources

- **Models**: [Abstain-R1 Collection](https://huggingface.co/collections/leoleung04/abstain-r1)
- **Datasets**: [Abstain-CoT & Abstain-Test](https://huggingface.co/collections/zhaihaotian/abstain-test-and-abstain-cot)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Abstain-R1.git
cd Abstain-R1

# Setup environment
source setup.sh

conda create -n verl python=3.10 -y
conda activate verl

pip install vllm==0.8.2
pip install tensordict==0.6.2
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install ray==2.44.0

# flash-attn
# Download from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3
# ./flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ./flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# verl
rm -rf verl  # Before removing the verl directory, save the a copy of __init__.py and sum_reward.py under verl/verl/utils/reward_score
git clone https://github.com/volcengine/verl.git verl
cd verl
git checkout v0.5.0
pip install -e .
# Move __init__.py and sum_reward.py back to verl/verl/utils/reward_score
```

## 📂 Project Structure

```
Abstain-R1/
├── data/
│   ├── Abstain-CoT/           # Training data with CoT format
│   ├── Abstain-Test-SUM/      # Test data with SUM subset
│   └── Abstain-Test-wo-SUM/   # Test data without SUM subset
├── run/
│   ├── train_abstain.sh                  # SFT training script
│   ├── run_verl.sh                       # GRPO/RLVR training script
│   ├── run_abstain_vllm_inference.sh     # Inference script (wo-SUM)
│   ├── run_vllm_inference_sum.sh         # Inference script (SUM)
│   ├── run_abstain_model_batch_vllm.py   # Inference Python (wo-SUM)
│   ├── run_single_model_batch_vllm_sum.py # Inference Python (SUM)
│   ├── evaluate_abstain-wosum.py         # Evaluation (wo-SUM, Permissive)
│   ├── evaluate_with_reference_sum.py    # Evaluation (SUM, Permissive)
│   ├── evaluate_unanswerable.py          # Evaluation (SUM, Strict)
│   └── evaluate_with_math_verify.py      # Evaluation (SUM, Strict)
├── models/                    # Model download scripts
├── config/                    # Training configurations
└── verl/                      # verl framework
```

## 🚀 Training

### SFT Training

```bash
bash run/train_abstain.sh
```

This script will:
- Automatically detect available GPUs
- Use multi-GPU training with `torchrun` if 4+ GPUs are available
- Use single-GPU training otherwise

### GRPO/RLVR Training

```bash
bash run/run_verl.sh
```

Key parameters (can be modified in the script):
- `MODEL_PATH`: Path to the base model (default: Qwen2.5-3B-Instruct)
- `data.train_batch_size`: Training batch size (default: 256)
- `actor_rollout_ref.rollout.n`: Number of rollout samples (default: 5)
- `trainer.total_training_steps`: Total training steps (default: 200)

## 🔮 Inference

### Inference on Abstain-Test-wo-SUM

```bash
bash run/run_abstain_vllm_inference.sh
```

Or use Python directly:

```bash
python run/run_abstain_model_batch_vllm.py \
    --model_path /path/to/model \
    --input_file data/Abstain-Test-wo-SUM/abstain-Test-withoutsum.jsonl \
    --output_file results/output.jsonl \
    --batch_size 128 \
    --max_new_tokens 4096
```

### Inference on Abstain-Test-SUM

```bash
bash run/run_vllm_inference_sum.sh
```

Or use Python directly:

```bash
python run/run_single_model_batch_vllm_sum.py \
    --model_path /path/to/model \
    --input_file data/Abstain-Test-SUM/sum_test_with_clarification_retry.jsonl \
    --output_file results/output_sum.jsonl \
    --batch_size 128 \
    --max_new_tokens 4096
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | - | Path to model or HuggingFace model ID |
| `--input_file` | - | Input JSONL file |
| `--output_file` | - | Output JSONL file |
| `--batch_size` | 128 | Batch size for inference |
| `--max_new_tokens` | 4096 | Maximum tokens to generate |
| `--temperature` | 0 | Sampling temperature (0 for greedy) |

## 📊 Evaluation

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correctness rate on answerable questions |
| **False Refusal Rate** | Rate of incorrectly refusing answerable questions |
| **Refusal Rate** | Rate of correctly refusing unanswerable questions |
| **Clarification Accuracy** | Quality of clarification for unanswerable questions |

### Permissive Protocol

Evaluate on **Abstain-Test-wo-SUM**:

```bash
python run/evaluate_abstain-wosum.py \
    --reference data/Abstain-Test-wo-SUM/abstain-Test-withoutsum.jsonl \
    --files results/your_model_result.jsonl
```

Evaluate on **Abstain-Test-SUM**:

```bash
python run/evaluate_with_reference_sum.py \
    --reference data/Abstain-Test-SUM/sum_test_with_clarification_retry.jsonl \
    --files results/your_model_result.jsonl
```

### Strict Protocol

A stricter evaluation protocol specifically for **Abstain-Test-SUM**:

```bash
# Evaluate unanswerable questions
python run/evaluate_unanswerable.py \
    --files results/your_model_result.jsonl

# Evaluate with math verification
python run/evaluate_with_math_verify.py \
    --files results/your_model_result.jsonl
```

## 📝 Data Format

### Training Data (Abstain-CoT)

```json
{
  "question": "...",
  "answer": "<thinking>\n...\n</thinking>\n<answer>\n...\n</answer>"
}
```

### Test Data (Abstain-Test)

The test dataset **Abstain-Test** consists of two subsets with different formats:

#### Abstain-Test-wo-SUM

Contains mixed questions from multiple datasets (AlCuna, BBQ, FalseQA, etc.):

```json
{
  "question": "...",
  "answer": "ground truth answer or null",
  "should_abstain": true/false,
  "clarification": "why the question is unanswerable (if applicable)",
  "dataset": "source dataset name"
}
```

#### Abstain-Test-SUM

Contains math questions with paired answerable/unanswerable versions:

```json
{
  "answerable_question": "original answerable question",
  "unanswerable_question": "modified unanswerable version",
  "ground_truth": "answer to the answerable question",
  "clarification": "why the modified question is unanswerable"
}
```

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgements

- [verl](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - Base model
