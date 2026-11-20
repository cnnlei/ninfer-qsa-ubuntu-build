#!/usr/bin/env python3
"""Download the exact HF inputs and adapt the author's v2 indexer for the upstream converter.

The HF indexers are safe F32 tensors with zero-centred norms. The pinned NInfer
converter expects the corresponding GGUF names and norms represented as 1+w;
it then unfolds those norms and casts only the two projections to BF16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

UPSTREAM_COMMIT = "29c79545429f03c2ec9d74370b3b2c59b62cca0d"
GGUF_REPO = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
GGUF_FILE = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
GGUF_SHA = "58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f"
MMPROJ_FILE = "mmproj-Qwen3.8-27B-BF16.gguf"
MMPROJ_SHA = "13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16"
INDEXER_REPO = "higa2515/Qwen3.8-27B-QSA-BF16"
INDEXER_FILE = "qsa-indexers.safetensors"
INDEXER_SHA = "985345467f60a2c0ae6cf9d8c875ce36d6a6da4aef4aa7428e04e457e4aad6d4"
HF_SUFFIX = {
    "query": "q_proj.weight",
    "key": "k_proj.weight",
    "query_norm": "q_norm",
    "key_norm": "k_block_norm",
    "key_pre_norm": "k_norm",
}
NORMS = {"query_norm", "key_norm", "key_pre_norm"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact_file(repo: str, filename: str, path: Path, expected: str) -> Path:
    path = path.resolve()
    if not path.exists():
        # A missing customized filename is probably a typo; never silently write into a
        # user's unrelated directory or rename a different downloaded model.
        if path.name != filename or str(path).startswith("/path/to/"):
            raise RuntimeError(f"Missing {path}; use the exact filename {filename} or an existing file")
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo}/{filename} -> {path}", flush=True)
        got = Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=path.parent))
        if got.resolve() != path:
            raise RuntimeError(f"Unexpected Hugging Face download path: {got}")
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"SHA-256 mismatch (expected exact pinned HF file): {path}")
    print(f"SHA-256 OK: {path.name}", flush=True)
    return path


def hf_name(layer: int, role: str) -> str:
    return f"model.language_model.layers.{layer}.self_attn._qsa_indexer.{HF_SUFFIX[role]}"


def check_upstream(upstream: Path):
    sys.path.insert(0, str(upstream.resolve()))
    from tools.convert import qsa_gguf

    if tuple(qsa_gguf.qsa_layers()) != tuple(range(3, 64, 4)) or set(qsa_gguf.ROLES) != set(HF_SUFFIX):
        raise RuntimeError("Unexpected QSA geometry in converter")
    return qsa_gguf


def adapt_indexer(source: Path, output: Path, upstream: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    qsa = check_upstream(upstream)
    source = exact_file(INDEXER_REPO, INDEXER_FILE, source, INDEXER_SHA)
    expected = qsa.expected_tensors()
    names = {hf_name(layer, role) for layer in qsa.qsa_layers() for role in qsa.ROLES}
    metadata = {"source_repo": INDEXER_REPO, "source_sha256": INDEXER_SHA, "version": "v2",
                "converter_commit": UPSTREAM_COMMIT, "norm_encoding": "gguf_1_plus_w_fp32"}
    tensors = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != names or len(names) != 80:
            raise RuntimeError(f"Expected exact 80 v2 F32 indexer tensors; missing={sorted(names-set(handle.keys()))[:3]}")
        for layer in qsa.qsa_layers():
            for role in qsa.ROLES:
                name = qsa.gguf_tensor(layer, role)
                value = handle.get_tensor(hf_name(layer, role))
                shape, gguf_type = expected[name]
                if value.dtype != torch.float32 or tuple(value.shape) != tuple(shape) or gguf_type != "F32":
                    raise RuntimeError(f"Unexpected dtype/shape for {name}: {value.dtype} {tuple(value.shape)}")
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"Non-finite v2 tensor: {name}")
                # Exactly the public author GGUF conversion: HF norms are zero-centred,
                # while qsa_gguf.sources_from_asset subtracts 1.0 on reading GGUF-style norms.
                mapped = (value + 1.0 if role in NORMS else value).contiguous()
                tensors[name] = mapped
    if output.exists():
        with safe_open(str(output), framework="pt", device="cpu") as existing:
            if existing.metadata() != metadata or set(existing.keys()) != set(tensors):
                raise RuntimeError(f"Existing indexer adapter is not the exact HF v2 asset: {output}")
            for name, data in tensors.items():
                stored = existing.get_tensor(name)
                if stored.dtype != torch.float32 or not torch.equal(stored, data):
                    raise RuntimeError(f"Existing adapted v2 tensor differs: {name}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".partial")
        if temp.exists():
            raise RuntimeError(f"Partial indexer adapter already exists: {temp}")
        save_file(tensors, str(temp), metadata=metadata)
        if output.exists():
            raise RuntimeError(f"Indexer adapter already exists: {output}")
        temp.rename(output)

    # Exercise the actual upstream asset reader. Verify all logical values/dtypes, not
    # just that 80 names are accepted. The engine requires BF16 Q/K projections;
    # the selected fp32_direct config keeps its three indexer norms in FP32.
    logical = qsa.sources_from_asset(path=output, dtype="fp32")
    if len(logical) != 80:
        raise RuntimeError("Upstream converter did not recognize all 80 indexer parameters")
    for layer in qsa.qsa_layers():
        for role in qsa.ROLES:
            logical_name = qsa.parameter_name(layer, role)
            mapped = tensors[qsa.gguf_tensor(layer, role)]
            wanted = (mapped - 1.0) if role in NORMS else mapped.to(torch.bfloat16)
            found = logical[logical_name].values()
            if found.dtype != wanted.dtype or not torch.equal(found, wanted.reshape(-1)):
                raise RuntimeError(f"Upstream indexer reader value mismatch: {logical_name}")
    print(f"HF v2: verified 80 tensors + 16 layers + upstream reader; adapted asset: {output}", flush=True)


def provenance_path(model: Path) -> Path:
    return Path(str(model) + ".qsa-v2.json")


def verify_marker(model: Path, *, mtp: bool, vision: bool, quick: bool = False) -> None:
    mark = json.loads(provenance_path(model).read_text())
    expected_components = {"text"} | ({"mtp"} if mtp else set()) | ({"vision"} if vision else set())
    expected = {"model_repo": GGUF_REPO, "model_file": GGUF_FILE, "model_sha256": GGUF_SHA,
                "indexer_repo": INDEXER_REPO, "indexer_file": INDEXER_FILE,
                "indexer_sha256": INDEXER_SHA, "indexer_version": "v2",
                "converter_commit": UPSTREAM_COMMIT,
                "gdn_controls": "base GGUF; no external GitHub tensors",
                "components": sorted(expected_components), "vision_sha256": MMPROJ_SHA if vision else None}
    if any(mark.get(k) != v for k, v in expected.items()) or mark.get("size_bytes") != model.stat().st_size:
        raise RuntimeError(f"Wrong/stale model provenance or components: {model}")
    if not quick and sha256(model) != mark.get("artifact_sha256"):
        raise RuntimeError(f"Converted model SHA-256 differs: {model}")
    print(f"Verified HF v2 model provenance: {model.name}")


def verify_or_record_model(model: Path, upstream: Path, *, mtp: bool, vision: bool) -> None:
    qsa = check_upstream(upstream)
    from tools.artifact.reader import Artifact

    report_path = Path(str(model) + ".conversion.json")
    report = json.loads(report_path.read_text())
    config = report["components"]["text"]["config"]
    if not config.get("fp32_direct") or any(config.get(k) != v for k, v in qsa.artifact_config().items()):
        raise RuntimeError("Converted model lacks the expected QSA geometry/FP32 indexer norm config")
    expected_components = {"text"} | ({"mtp"} if mtp else set()) | ({"vision"} if vision else set())
    if set(report["components"]) != expected_components:
        raise RuntimeError(f"Wrong converted components: {list(report['components'])}")
    methods = {param: job["format"] for job in report["methods"] for param in job["parameters"]
               if "/attention/indexer/" in param}
    want_methods = {qsa.parameter_name(layer, role): "fp32" if role in NORMS else "bf16"
                    for layer in qsa.qsa_layers() for role in qsa.ROLES}
    if methods != want_methods:
        raise RuntimeError("Converted indexer formats/parameters do not match the 80 intended tensors")
    with Artifact(model) as artifact:
        if len(artifact.directory.files) != 1 or set(artifact.directory.components) != expected_components:
            raise RuntimeError("Expected a single-file NInfer v3 artifact with the selected components")
        if not set(want_methods).issubset(artifact.directory.bindings):
            raise RuntimeError("Missing indexer bindings in actual NInfer artifact")
        artifact_id = artifact.artifact_id.hex()
        if artifact_id != report.get("artifact_id"):
            raise RuntimeError("Artifact id differs from conversion report")
    record = {"model_repo": GGUF_REPO, "model_file": GGUF_FILE, "model_sha256": GGUF_SHA,
              "indexer_repo": INDEXER_REPO, "indexer_file": INDEXER_FILE,
              "indexer_sha256": INDEXER_SHA, "indexer_version": "v2",
              "converter_commit": UPSTREAM_COMMIT, "gdn_controls": "base GGUF; no external GitHub tensors",
              "components": sorted(expected_components), "vision_sha256": MMPROJ_SHA if vision else None,
              "artifact_id": artifact_id, "size_bytes": model.stat().st_size,
              "artifact_sha256": sha256(model)}
    mark = provenance_path(model)
    if mark.exists():
        if json.loads(mark.read_text()) != record:
            raise RuntimeError(f"Existing v2 model provenance differs: {mark}")
    else:
        # Creation is only allowed immediately after the caller's successful conversion.
        with mark.open("x", encoding="utf-8") as out:
            json.dump(record, out, indent=2, ensure_ascii=False)
            out.write("\n")
    print(f"Verified V3 artifact: {artifact_id}, {record['size_bytes']} bytes, {record['artifact_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    indexer = sub.add_parser("indexer")
    indexer.add_argument("--source", type=Path, required=True)
    indexer.add_argument("--output", type=Path, required=True)
    indexer.add_argument("--upstream", type=Path, required=True)
    model = sub.add_parser("model-inputs")
    model.add_argument("--gguf", type=Path, required=True)
    model.add_argument("--mmproj", type=Path)
    verify = sub.add_parser("record-model")
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--upstream", type=Path, required=True)
    verify.add_argument("--mtp", action="store_true")
    verify.add_argument("--vision", action="store_true")
    marker = sub.add_parser("verify-marker")
    marker.add_argument("--model", type=Path, required=True)
    marker.add_argument("--mtp", action="store_true")
    marker.add_argument("--vision", action="store_true")
    marker.add_argument("--quick", action="store_true", help="check provenance and size, not the full model hash")
    args = parser.parse_args()
    if args.command == "indexer":
        adapt_indexer(args.source, args.output, args.upstream)
    elif args.command == "model-inputs":
        exact_file(GGUF_REPO, GGUF_FILE, args.gguf, GGUF_SHA)
        if args.mmproj:
            exact_file(GGUF_REPO, MMPROJ_FILE, args.mmproj, MMPROJ_SHA)
    elif args.command == "record-model":
        verify_or_record_model(args.model, args.upstream, mtp=args.mtp, vision=args.vision)
    else:
        verify_marker(args.model, mtp=args.mtp, vision=args.vision, quick=args.quick)


if __name__ == "__main__":
    main()
