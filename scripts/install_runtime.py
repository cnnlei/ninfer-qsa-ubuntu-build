#!/usr/bin/env python3
"""Fetch/check/install the pinned, split upstream runtime published by our CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(value: str) -> str:
    if not value or any(x in value for x in ("/", "\\", "\n", "\r")) or value in (".", ".."):
        raise ValueError(f"Unsafe asset name: {value!r}")
    return value


def manifest(path: Path, flat: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+[ *]?(.+)", line)
        if not match:
            raise ValueError(f"Malformed checksum line in {path.name}")
        sha, name = match.groups()
        if flat:
            safe_name(name)
        else:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError(f"Unsafe path in checksum list: {name!r}")
        if name in result:
            raise ValueError(f"Duplicate checksum entry: {name}")
        result[name] = sha.lower()
    if not result:
        raise ValueError(f"Empty checksum file: {path}")
    return result


def check_hashes(directory: Path, checks: dict[str, str]) -> None:
    for name, expected in checks.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or unexpected file: {path}")
        if digest(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch: {path}. Remove that exact file and retry.")
        print(f"  OK {name}", flush=True)


def release_assets(repo: str, tag: str | None, prefix: str) -> dict[str, dict]:
    url = f"https://api.github.com/repos/{repo}/releases/"
    url += "latest" if not tag else "tags/" + quote(tag, safe="")
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "ninfer-qsa-ubuntu-build"})
    try:
        with urlopen(req, timeout=30) as response:
            release = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(f"No matching release yet. Check https://github.com/{repo}/actions") from exc
        raise RuntimeError(f"GitHub API HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot query GitHub releases: {exc.reason}") from exc
    checksum = f"SHA256SUMS-{prefix}.txt"
    archive_re = re.compile(re.escape(prefix) + r"\.tar\.gz\.\d{3}")
    assets = {
        a["name"]: a for a in release.get("assets", [])
        if a.get("name") == checksum or archive_re.fullmatch(a.get("name", ""))
    }
    if checksum not in assets:
        raise RuntimeError(f"Release {release.get('tag_name')} does not contain the pinned SM120a build")
    print(f"Release: {release.get('tag_name')} ({release.get('html_url')})", flush=True)
    return assets


def download_asset(asset: dict, directory: Path, repo: str) -> None:
    name = safe_name(asset["name"])
    url = asset["browser_download_url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith(f"/{repo}/releases/download/"):
        raise RuntimeError(f"Unexpected GitHub asset URL for {name}")
    expected_size = int(asset["size"])
    path = directory / name
    if path.exists():
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Existing asset has the wrong size: {path}; remove it and retry")
        return
    partial = path.with_name(name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"Oversized partial download: {partial}")
    if not partial.exists() or partial.stat().st_size != expected_size:
        cmd = ["curl", "--fail", "--location", "--retry", "4", "--connect-timeout", "30"]
        if partial.exists() and partial.stat().st_size:
            cmd += ["--continue-at", "-"]
        cmd += ["--output", str(partial), url]
        subprocess.run(cmd, check=True)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded size differs: {partial}")
    partial.rename(path)


def verify_build(path: Path, commit: str, check_contents: bool = False) -> None:
    info = json.loads((path / "BUILD-INFO.json").read_text())
    if (
        info.get("source_revision") != commit
        or info.get("architectures") != ["120a"]
        or info.get("cuda") != "13.4"
        or info.get("modified_sources")
        or info.get("layout") != "shared-runtime-v1"
    ):
        raise RuntimeError(f"Wrong source/architecture/toolchain in {path / 'BUILD-INFO.json'}")
    server = path / "ninfer-serve"
    if not server.is_file() or not os.access(server, os.X_OK):
        raise RuntimeError(f"Missing executable: {server}")
    if check_contents:
        check_hashes(path, manifest(path / "SHA256SUMS.txt", flat=False))
        subprocess.run([str(server), "--version"], check=True, timeout=60)
        help_text = (path / "help/ninfer-serve.txt").read_text()
        for flag in ("--qwen38-27b-qsa", "--qsa-top-blocks", "--kv-resident"):
            if flag not in help_text:
                raise RuntimeError(f"QSA flag missing: {flag}")


def extract_archives(parts: list[Path], destination: Path, prefix: str) -> None:
    concat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    try:
        assert concat.stdout is not None
        with tarfile.open(fileobj=concat.stdout, mode="r|gz") as archive:
            for member in archive:
                p = PurePosixPath(member.name)
                if (not p.parts or p.parts[0] != prefix or p.is_absolute()
                        or ".." in p.parts or not (member.isfile() or member.isdir())):
                    raise RuntimeError(f"Unexpected archive entry: {member.name!r}")
                archive.extract(member, path=destination, filter="data")
        concat.stdout.close()
        if concat.wait() != 0:
            raise RuntimeError("Cannot concatenate archive volumes")
    finally:
        if concat.stdout is not None:
            concat.stdout.close()
        if concat.poll() is None:
            concat.terminate()
            concat.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--offline", action="store_true", help="use manifest and volumes already in --downloads")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("invalid GitHub repo")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("invalid source commit")
    if platform.machine() != "x86_64":
        raise RuntimeError("This build requires Linux x86_64")
    glibc = os.confstr("CS_GNU_LIBC_VERSION") or ""
    version = re.search(r"(\d+)\.(\d+)", glibc)
    if not version or tuple(map(int, version.groups())) < (2, 39):
        raise RuntimeError(f"Ubuntu 24.04+ / glibc >= 2.39 required; detected {glibc!r}")

    prefix = f"ninfer-linux-x64-cuda13.4-sm120a-{args.commit[:12]}"
    final = args.runtime_root.resolve() / prefix
    downloads = args.downloads.resolve()
    if final.exists():
        verify_build(final, args.commit)
        print(f"Already installed: {final}")
        return

    downloads.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    checksum = f"SHA256SUMS-{prefix}.txt"
    if not args.offline:
        assets = release_assets(args.repo, args.tag, prefix)
        download_asset(assets[checksum], downloads, args.repo)
    checks = manifest(downloads / checksum, flat=True)
    pattern = re.compile(re.escape(prefix) + r"\.tar\.gz\.(\d{3})")
    names = sorted(checks)
    volume_numbers = []
    for name in names:
        match = pattern.fullmatch(name)
        if not match:
            raise RuntimeError(f"Unexpected volume: {name}")
        volume_numbers.append(int(match[1]))
    if volume_numbers != list(range(1, len(names) + 1)):
        raise RuntimeError("Missing or nonconsecutive archive volumes")
    if not args.offline:
        for name in names:
            if name not in assets:
                raise RuntimeError(f"Release missing {name}")
            download_asset(assets[name], downloads, args.repo)
    check_hashes(downloads, checks)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=final.parent) as temporary:
        stage = Path(temporary)
        extract_archives([downloads / n for n in names], stage, prefix)
        candidate = stage / prefix
        verify_build(candidate, args.commit, check_contents=True)
        if final.exists():
            raise RuntimeError(f"Destination appeared during install: {final}")
        candidate.rename(final)
    print(f"Installed: {final}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
