#!/usr/bin/env python3
"""Create a relocatable Windows ZIP from a prepared staging directory.

Use an explicit allowlist: never include user models, download caches, temporary
files or credentials from other locations.  Include the shallow upstream Git
checkout so the same pinned-commit and clean-tree checks still work offline.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


PREFIX = "NInfer-QSA-Windows-Portable"
FILES = (
    "convert-windows-portable.bat",
    "README.md",
    "PORTABLE-INFO.txt",
    "scripts/convert_windows.py",
    "scripts/hf_v2.py",
    "scripts/fetch_base_metadata.py",
    "assets/base-model/config.json",
    "assets/base-model/tokenizer.json",
    "assets/base-model/tokenizer_config.json",
    "assets/base-model/chat_template.jinja",
    "assets/base-model/generation_config.json",
    "assets/base-model/preprocessor_config.json",
    "assets/base-model/video_preprocessor_config.json",
)
TREES = (".portable/python", ".portable/git", "upstream")
REQUIRED = (
    ".portable/python/python.exe",
    ".portable/git/cmd/git.exe",
    "upstream/.git/HEAD",
    "upstream/tools/convert/convert_qsa27b.py",
)


def package(bundle: Path, archive: Path) -> Path:
    bundle, archive = bundle.resolve(), archive.resolve()
    if archive.exists() or archive.is_relative_to(bundle):
        raise ValueError("Use a new ZIP outside the staging directory")
    for relative in FILES + REQUIRED:
        if not (bundle / relative).is_file():
            raise FileNotFoundError(bundle / relative)
    paths = [bundle / name for name in FILES]
    for relative in TREES:
        for path in (bundle / relative).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in portable bundle: {path}")
            if path.is_file():
                paths.append(path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as output:
        for path in sorted(paths):
            if not path.resolve().is_relative_to(bundle):
                raise ValueError(f"Portable bundle input outside staging directory: {path}")
            output.write(path, f"{PREFIX}/{path.relative_to(bundle).as_posix()}")
    # GitHub release files must be smaller than 2 GiB.
    if archive.stat().st_size >= 2 * 1024**3:
        raise ValueError("Portable ZIP exceeds the GitHub release file limit")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    sums = archive.with_name("SHA256SUMS-Windows-Portable.txt")
    with sums.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{digest}  {archive.name}\n")
    print(f"Portable archive: {archive} ({archive.stat().st_size:,} bytes; {len(paths):,} files)")
    print(f"SHA-256: {digest}")
    return sums


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    package(args.bundle, args.archive)
