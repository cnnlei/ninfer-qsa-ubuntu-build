"""Local, model-free tests. No GPU, network access, or root required."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "29c79545429f03c2ec9d74370b3b2c59b62cca0d"
PREFIX = f"ninfer-linux-x64-cuda13.4-sm120a-{COMMIT[:12]}"
spec = importlib.util.spec_from_file_location("install_runtime", ROOT / "scripts/install_runtime.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ScriptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.config = self.path / "qsa.env"
        self.state = self.path / "run"
        self.config.write_text(f"QSA_STATE_DIR={shlex.quote(str(self.state))}\nQSA_HOST_CACHE_MIB=0\n")
        self.env = dict(os.environ, QSA_CONFIG=str(self.config))

    def append(self, text):
        with self.config.open("a") as stream:
            stream.write(text + "\n")

    def run_cli(self, *args, check=True):
        return subprocess.run(["bash", str(ROOT / "qsa.sh"), *args], env=self.env,
                              capture_output=True, text=True, check=check, timeout=20)

    def test_plan_streaming(self):
        self.append('QSA_MODEL="model with spaces.ninfer"')
        args = shlex.split(self.run_cli("plan").stdout)
        self.assertIn("--qwen38-27b-qsa", args)
        self.assertEqual(args[args.index("--kv-capacity") + 1], "262144")
        self.assertEqual(args[args.index("--kv-resident") + 1], "8192")
        self.assertNotIn("--kv-headroom-mib", args)
        self.assertNotIn("--rope-yarn", args)
        self.assertTrue(args[1].endswith("model with spaces.ninfer"))

    def test_plan_yarn_and_resident(self):
        self.append("QSA_CONTEXT=524288\nQSA_KV_RESIDENT=all\nQSA_MTP=0\nQSA_VISION=0")
        args = shlex.split(self.run_cli("plan").stdout)
        self.assertIn("--rope-yarn", args)
        self.assertIn("--kv-headroom-mib", args)
        self.assertEqual(args[args.index("--kv-capacity") + 1], "auto")
        self.assertEqual(args[args.index("--spec") + 1], "none")
        self.assertNotIn("--vision", args)

    def test_invalid_residency(self):
        self.append("QSA_TOP_BLOCKS=4096\nQSA_KV_RESIDENT=4096")
        self.assertNotEqual(self.run_cli("plan", check=False).returncode, 0)

    def test_key_required_and_redacted(self):
        self.append("QSA_BIND=0.0.0.0")
        self.assertNotEqual(self.run_cli("plan", check=False).returncode, 0)
        key = self.path / "key"
        key.write_text("test-secret-not-real\n")
        key.chmod(0o600)
        self.append(f"QSA_API_KEY_FILE={shlex.quote(str(key))}")
        result = self.run_cli("plan").stdout
        self.assertNotIn("test-secret-not-real", result)
        self.assertIn("hidden", result)

    def test_init_preserves_existing_config(self):
        previous = self.config.read_bytes()
        self.run_cli("init")
        self.assertEqual(previous, self.config.read_bytes())

    def test_stop_refuses_pid_reuse_even_with_invalid_config(self):
        self.state.mkdir()
        (self.state / "server.pid").write_text(f"{os.getpid()} 0 {Path('/proc/sys/kernel/random/boot_id').read_text().strip()}\n")
        self.append("QSA_PORT=invalid")
        self.run_cli("stop")
        self.assertFalse((self.state / "server.pid").exists())
        os.kill(os.getpid(), 0)

    def test_mocked_start_status_stop(self):
        fake = self.path / "ninfer-serve"
        fake.write_text('''#!/usr/bin/env python3
import sys, http.server
if "--version" in sys.argv:
    print("mock 120a"); sys.exit(0)
if "--help" in sys.argv:
    print("--qwen38-27b-qsa --kv-resident"); sys.exit(0)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
    def do_POST(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"mock":true}')
http.server.HTTPServer(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])),Handler).serve_forever()
''')
        fake.chmod(0o755)
        gpu = self.path / "nvidia-smi"
        gpu.write_text('#!/bin/bash\necho "NVIDIA GeForce RTX 5070 Ti, 16384, 610.62, 12.0"\n')
        gpu.chmod(0o755)
        self.env["PATH"] = str(self.path) + ":" + self.env["PATH"]
        model = self.path / "mock.ninfer"
        model.write_bytes(b"mock")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.append(f"QSA_SERVE_BIN={shlex.quote(str(fake))}\nQSA_MODEL={shlex.quote(str(model))}\nQSA_PORT={port}\nQSA_START_WAIT=5\nQSA_STOP_WAIT=5")
        try:
            self.run_cli("start")
            self.assertIn("Ready", self.run_cli("status").stdout)
            self.assertIn("已经运行", self.run_cli("start").stdout)
            self.assertIn('"mock":true', self.run_cli("test").stdout)
            self.run_cli("stop")
            self.assertIn("未运行", self.run_cli("status").stdout)
        finally:
            self.run_cli("stop", "--force", check=False)


class ArchiveTest(unittest.TestCase):
    def test_manifest_and_split_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / PREFIX
            package.mkdir()
            (package / "hello").write_text("payload")
            arc = root / "bundle.tar.gz"
            with tarfile.open(arc, "w:gz") as out:
                out.add(package, arcname=PREFIX)
            raw = arc.read_bytes()
            volumes = [root / ("a.%03d" % i) for i in (1, 2)]
            volumes[0].write_bytes(raw[:len(raw)//2])
            volumes[1].write_bytes(raw[len(raw)//2:])
            sums = root / "SHA256SUMS.txt"
            sums.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in volumes))
            installer.check_hashes(root, installer.manifest(sums, flat=True))
            stage = root / "stage"
            stage.mkdir()
            installer.extract_archives(volumes, stage, PREFIX)
            self.assertEqual((stage / PREFIX / "hello").read_text(), "payload")

    def test_path_traversal_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            sums = Path(temp) / "SHA256SUMS.txt"
            sums.write_text("0" * 64 + "  ../outside\n")
            with self.assertRaises(ValueError):
                installer.manifest(sums, flat=False)


if __name__ == "__main__":
    unittest.main()
