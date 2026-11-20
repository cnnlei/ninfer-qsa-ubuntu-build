#!/usr/bin/env python3
"""Fetch only Qwen3.8-27B configuration/tokenizer/vision metadata, never weights."""

import argparse
import json
from pathlib import Path
import shutil

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
    parser.add_argument("--prepared-dir", type=Path, required=True,
                        help="Separate metadata copy for the NInfer QSA converter")
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
    if args.directory.resolve() == args.prepared_dir.resolve():
        raise SystemExit("Prepared metadata must be a separate directory")
    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        if name != "config.json":
            shutil.copy2(args.directory / name, args.prepared_dir / name)
    cfg = json.loads((args.directory / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    if text_cfg.get("hidden_size") != 5120 or text_cfg.get("num_hidden_layers") != 64:
        raise SystemExit("Expected the official dense Qwen3.8-27B model config")
    # In the pinned fork this specifically preserves the three QSA indexer norms.
    # Other direct norms stay BF16, as required by the existing engine kernels.
    text_cfg["fp32_direct"] = True
    (args.prepared_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Base metadata ready: {args.prepared_dir} (QSA norms FP32; original metadata unchanged)")


if __name__ == "__main__":
    main()
