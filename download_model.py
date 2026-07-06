#!/usr/bin/env python3
"""One-time setup: download the Kimi-K2.6 NVFP4 checkpoint (~554 GiB) into the
'kimi-k26' Modal volume at /models/Kimi-K2.6-NVFP4, where the session runners
expect it. Modal bills volume storage while it exists; delete the volume when done.

Usage:
    modal run download_model.py
"""
import modal

app = modal.App("brl11-download-model")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))

vol = modal.Volume.from_name("kimi-k26", create_if_missing=True)

REPO = "nvidia/Kimi-K2.6-NVFP4"
DEST = "/models/Kimi-K2.6-NVFP4"


@app.function(image=image, volumes={"/models": vol}, timeout=6 * 3600, cpu=8)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, local_dir=DEST)
    vol.commit()
    print(f"{REPO} -> {DEST} done")


@app.local_entrypoint()
def main():
    download.remote()
