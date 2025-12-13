from huggingface_hub import snapshot_download

models = {
    "general_verifier": "HKUST-Audio/xVerify-3B-Ia",
}

base_dir = "/workspace/models"

for name, repo in models.items():

    snapshot_download(
        repo_id=repo,
        local_dir=f"{base_dir}/{name}",
        local_dir_use_symlinks=False,
        revision="main"
    )