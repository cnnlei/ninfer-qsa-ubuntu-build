#!/usr/bin/env python3
"""Interactive Windows conversion of ISTA IQ3_S MTP + the pinned HF v2 indexers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import hf_v2

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/13thblade/ninfer-all-QSA.git"
TAG = "win-cuda13.4-29c79545429f"
OUTPUT_NAME = "Qwen3.8-27B-GSQ-RCO-IQ3_S-QSA-v2-mtp.ninfer"
MODEL_ID = "qwen3.8-27b-qsa-v2-iq3s"


def run(argv, *, cwd=ROOT, env=None, check=True):
    return subprocess.run([str(part) for part in argv], cwd=cwd, env=env, check=check)


def git_output(path: Path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def ask_path(prompt: str, default: Path) -> Path:
    answer = input(f"{prompt}\n  [默认 {default}]\n> ").strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'":
        answer = answer[1:-1]
    return Path(answer).expanduser().resolve() if answer else default.resolve()


def sibling_or_default(explicit: Path | None, gguf: Path, filename: str, default: Path) -> Path:
    """Reuse files next to the chosen GGUF, without searching other directories."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    sibling = gguf.parent / filename
    return sibling.resolve() if sibling.is_file() else default.resolve()


def setup_environment(source: Path, venv: Path) -> Path:
    if not shutil.which("git"):
        raise RuntimeError("未找到 Git，请安装 Git for Windows 后重新运行：https://git-scm.com/download/win")
    if not source.exists():
        run(["git", "-c", "core.autocrlf=false", "clone", "--depth", "1", "--branch", TAG, UPSTREAM, source])
    if git_output(source, "rev-parse", "HEAD") != hf_v2.UPSTREAM_COMMIT:
        raise RuntimeError(f"源码目录版本不匹配，拒绝覆盖：{source}")
    if git_output(source, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"转换器源码存在本地修改，拒绝使用：{source}")
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        run([sys.executable, "-m", "venv", venv])
    probe = "import sys,torch,numpy,gguf,safetensors,huggingface_hub,jinja2; assert sys.version_info >= (3,11)"
    if subprocess.run([str(python), "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        print("安装独立虚拟环境的 CPU 版 PyTorch、转换器依赖…", flush=True)
        run([python, "-m", "pip", "install", "--disable-pip-version-check", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([python, "-m", "pip", "install", "--disable-pip-version-check", "numpy", "gguf",
             "safetensors", "huggingface_hub", "jinja2"])
    run([python, "-c", probe])
    return python


def conversion_flags(mtp: bool, vision: bool):
    result = []
    if mtp:
        result.append("--mtp")
    if vision:
        result.append("--vision")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, help="existing exact ISTA IQ3_S MTP file (empty = auto-download)")
    parser.add_argument("--mmproj", type=Path, help="existing official ISTA BF16 mmproj")
    parser.add_argument("--indexer", type=Path, help="existing exact HF v2 qsa-indexers.safetensors")
    parser.add_argument("--output", type=Path, help="output .ninfer path, including filename")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--base-model-dir", type=Path, help="optional existing official Qwen config/tokenizer files")
    parser.add_argument("--no-vision", action="store_true", help="omit vision weights and mmproj download")
    parser.add_argument("--no-mtp", action="store_true", help="omit MTP component (normally keep it)")
    parser.add_argument("--proposal", action="store_true", help="also pack optional proposal head (more weight memory)")
    parser.add_argument("--auto", action="store_true", help="no path prompts; use defaults/CLI paths")
    parser.add_argument("--check-only", action="store_true", help="upstream preflight only, no .ninfer output")
    parser.add_argument("--plan", action="store_true", help="show exact model and indexer sources, no downloads or writes")
    args = parser.parse_args()
    assets = args.asset_dir.expanduser().resolve()
    raw_default = assets / "hf-v2" / hf_v2.INDEXER_FILE
    indexer = assets / "hf-v2" / "indexer_tensors-v2-ninfer.safetensors"
    base = (args.base_model_dir or assets / "base-model").resolve()
    prepared = assets / "base-model-qsa-v2"
    gguf = (args.gguf or assets / "inputs" / hf_v2.GGUF_FILE).resolve()
    mmproj_default = assets / "inputs" / hf_v2.MMPROJ_FILE
    mmproj = sibling_or_default(args.mmproj, gguf, hf_v2.MMPROJ_FILE, mmproj_default)
    raw = sibling_or_default(args.indexer, gguf, hf_v2.INDEXER_FILE, raw_default)
    out_name = OUTPUT_NAME if not args.no_mtp else OUTPUT_NAME.replace("-mtp", "")
    if args.no_vision:
        out_name = out_name.replace(".ninfer", "-text.ninfer")
    output = (args.output or ROOT / "models" / out_name).resolve()
    vision, mtp = not args.no_vision, not args.no_mtp
    components = ",".join(["text", *(["vision"] if vision else []), *(["mtp"] if mtp else [])])

    print("\n=== Qwen3.8-27B GSQ-RCO IQ3_S / HF QSA indexer v2 ===", flush=True)
    print(f"模型: {hf_v2.GGUF_REPO}/{hf_v2.GGUF_FILE}")
    print(f"索引器: {hf_v2.INDEXER_REPO}/{hf_v2.INDEXER_FILE} (v2, SHA-256 {hf_v2.INDEXER_SHA})")
    print(f"转换器: 13thblade/ninfer-all-QSA @ {hf_v2.UPSTREAM_COMMIT}")
    print("不下载 GitHub Release 的 QSA/GDN 张量；保留基模型自己的 GDN controls。")
    if not args.auto and not args.plan and not sys.argv[1:]:
        gguf = ask_path("请输入已有 IQ3_S-mtp GGUF 路径；留空自动下载", gguf)
        if vision:
            mmproj = sibling_or_default(args.mmproj, gguf, hf_v2.MMPROJ_FILE, mmproj_default)
            mmproj = ask_path("请输入已有 ISTA BF16 mmproj 路径；回车使用默认路径", mmproj)
        raw = sibling_or_default(args.indexer, gguf, hf_v2.INDEXER_FILE, raw_default)
        raw = ask_path("请输入已下载的 HF v2 qsa-indexers.safetensors 路径；回车使用默认路径", raw)
        output = ask_path("请输入输出 .ninfer 的完整路径（可选共享模型盘）", output)
    print(f"文本输入: {gguf}\n视觉输入: {mmproj if vision else '不启用'}\nQSA v2 输入: {raw}\n组件: {components}\n输出: {output}\n", flush=True)
    if args.plan:
        return
    if sys.version_info < (3, 11) or struct.calcsize("P") != 8:
        raise RuntimeError("需要 64 位 Python 3.11 或 3.12（或兼容的更新版本）")
    if output.suffix.lower() != ".ninfer":
        raise RuntimeError("输出路径必须以 .ninfer 结尾")
    if not args.check_only and output.exists():
        if not hf_v2.provenance_path(output).is_file():
            raise RuntimeError(f"输出已存在且无 v2 来源证明，不会覆盖：{output}")
        hf_v2.verify_marker(output, mtp=mtp, vision=vision)
        print("此 v2 模型已完成，校验通过，无需重复转换。")
        return
    report = Path(str(output) + ".conversion.json")
    if not args.check_only and (report.exists() or hf_v2.provenance_path(output).exists()):
        raise RuntimeError(f"输出路径有旧转换报告，请选择新路径：{output}")

    source = ROOT / "upstream"
    venv = ROOT / ".venv-win"
    python = setup_environment(source, venv)
    run([python, ROOT / "scripts/hf_v2.py", "indexer", "--source", raw, "--output", indexer, "--upstream", source])
    meta = [python, ROOT / "scripts/fetch_base_metadata.py", base, "--prepared-dir", prepared]
    if vision:
        meta += ["--vision"]
    run(meta)
    inputs = [python, ROOT / "scripts/hf_v2.py", "model-inputs", "--gguf", gguf]
    if vision:
        inputs += ["--mmproj", mmproj]
    run(inputs)

    if args.check_only:
        target = output
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        target = output.with_name(f"{output.stem}.partial.{os.getpid()}.ninfer")
        if target.exists():
            raise RuntimeError(f"临时输出已存在：{target}")
    command = [python, source / "tools/convert/convert_qsa27b.py", "--base-model", prepared,
               "--gguf", gguf, "--out", target, "--components", components, "--device", "cpu",
               "--name", MODEL_ID, "--proposal" if args.proposal else "--no-proposal"]
    if vision:
        command += ["--vision", mmproj]
    if args.check_only:
        command += ["--check-only"]
    env = dict(os.environ, NINFER_QSA_INDEXER_ASSET=str(indexer), NINFER_GDN_CONTROLS_BF16="off")
    env.pop("QSA_INDEXER_GGUF", None)
    run(command, cwd=source, env=env)
    if args.check_only:
        print("检查完成，未生成模型。")
        return

    temporary_report = Path(str(target) + ".conversion.json")
    if not target.is_file() or not temporary_report.is_file():
        raise RuntimeError(f"转换器没有生成完整文件；请检查临时输出：{target}")
    run([python, ROOT / "scripts/hf_v2.py", "record-model", "--model", target, "--upstream", source,
         *conversion_flags(mtp, vision)])
    if any(path.exists() for path in (output, report, hf_v2.provenance_path(output))):
        raise RuntimeError(f"目标路径在转换中已出现，拒绝覆盖：{output}")
    temporary_report.rename(report)
    hf_v2.provenance_path(target).rename(hf_v2.provenance_path(output))
    target.rename(output)
    print(f"\n转换成功：{output}\n请连同 {report.name} 和 {hf_v2.provenance_path(output).name} 一起保留。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        print(f"转换失败：{exc}\n旧模型不会被覆盖。", file=sys.stderr)
        sys.exit(1)
