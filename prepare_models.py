#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-download HF models into local cache.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model repo id")
    p.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / ".hf_cache"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Install it to pre-download models.")
        return 1
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        cache_dir=str(cache_dir),
        local_dir=None,
        local_dir_use_symlinks=False,
    )
    print("[done] model cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
