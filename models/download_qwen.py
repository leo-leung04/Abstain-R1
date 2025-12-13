from huggingface_hub import snapshot_download
import os

models = {
    "qwen25_3b_instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
}

base_dir = "/workspace/models"   # 你可改成自己的路径
os.makedirs(base_dir, exist_ok=True)

for name, repo in models.items():
    print(f"\n===== Downloading {repo} =====")
    snapshot_download(
        repo_id=repo,
        local_dir=f"{base_dir}/{name}",
        local_dir_use_symlinks=
        False,
        revision="main"
    )
    print(f"✔ Done: {repo} saved to {base_dir}/{name}")
