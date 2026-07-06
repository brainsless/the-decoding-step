#!/usr/bin/env python3
"""Download tiktoken.model and tokenization_kimi.py from moonshotai/Kimi-K2.6
into assets/ and verify against pinned SHA-256 hashes. The runners need these to
rebuild the benchmark prompts; a wrong tokenizer fails here instead of as a
prompt-sha mismatch later.
"""
import hashlib
import os
import sys
import urllib.request

REPO = "moonshotai/Kimi-K2.6"
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FILES = {
    # local name           remote name             sha256
    "kimi_tiktoken.model": ("tiktoken.model",
        "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"),
    "tokenization_kimi.py": ("tokenization_kimi.py",
        "2ab1ffb6f5c4380758bd8d9752ff1041c09024182676a4311528fbdf92fb9599"),
}


def fetch(local, remote, sha):
    dest = os.path.join(ASSETS, local)
    if os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha:
        print(f"  {local}: present, hash ok")
        return
    url = f"https://huggingface.co/{REPO}/resolve/main/{remote}"
    print(f"  {local}: downloading {url}")
    data = urllib.request.urlopen(url, timeout=120).read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        sys.exit(f"hash mismatch for {remote}: got {got}, pinned {sha}. "
                 f"The upstream file changed; do not proceed.")
    open(dest, "wb").write(data)
    print(f"  {local}: written, hash ok")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    for local, (remote, sha) in FILES.items():
        fetch(local, remote, sha)
    print("assets ready")
