import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
    local_dir="/data/opt_data_backup/data/jiang/cross/CRoSS/coco_val/sd-inpainting",
    ignore_patterns=["*.bin"],
)
print("done")
