from huggingface_hub import snapshot_download

models = {
    "Qwen25-3B-Abstain-R1-Clarification": "leoleung04/Qwen25-3B-Abstain-R1-Clarification",
        "Qwen25-3B-Abstain-R1-Clarification": "leoleung04/Qwen25-3B-Abstain-R1-Clarification",
    "Qwen25-3B-Abstain-R1-NoClarification": "leoleung04/Qwen25-3B-Abstain-R1-NoClarification",
    "Qwen25-3B-Abstain-SFT": "leoleung04/Qwen25-3B-Abstain-SFT",
}

base_dir = "/workspace/models"

for name, repo in models.items():

    snapshot_download(
        repo_id=repo,
        local_dir=f"{base_dir}/{name}",
        local_dir_use_symlinks=False,
        revision="main"
    )