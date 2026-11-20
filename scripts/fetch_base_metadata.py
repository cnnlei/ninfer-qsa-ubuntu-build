#!/usr/bin/env python3
"""Fetch only Qwen3.8-27B configuration/tokenizer/vision metadata, never weights."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    required = FILES if args.vision else FILES[:5]
    missing = [name for name in required if not (args.directory / name).is_file()]
    if missing:
        snapshot_download(
            repo_id="Qwen/Qwen3.8-27B",
            local_dir=str(args.directory),
            allow_patterns=list(required),
        )
    still_missing = [name for name in required if not (args.directory / name).is_file()]
    if still_missing:
        raise SystemExit("Missing base metadata: " + ", ".join(still_missing))
    print(f"Base metadata ready: {args.directory}")


if __name__ == "__main__":
    main()
