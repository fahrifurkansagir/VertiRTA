#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host-Side Zero-Waste Single-Shot Colab G4 Batch Thunder Run Orchestrator (Plan 6C Task 6).

Enforces ALL Approach C-G4 operational safety locks (#329 hardened for google-colab-cli v0.8.0):
  1. Absolute Zero-Waste Lock: `--live-g4` ALSO requires explicit `--confirm-single-shot-g4`
     AND passes local `0 CU` pre-flight (`pytest tests/colab/ -v`) strictly BEFORE provisioning.
  2. Verified `colab_cli v0.8.0` API & CLI Syntax:
       - Uses `from colab_cli.common import state; state.client.list_assignments()` &
         `state.client.unassign(endpoint)` via dynamically resolved `colab` interpreter (zero
         hardcoded usernames/home paths).
       - Passes ALL remote Python code to `colab exec -s <session> --timeout <sec>` strictly
         via `stdin` (`input=code`), never as an invalid positional CLI arg.
  3. Remote VM Bootstrap Before Wave 1:
       - Creates `remote_work_dir` BEFORE `colab upload`.
       - Runs `pip uninstall -y torchao` (#295 guard), installs missing packages with
         `--upgrade-strategy only-if-needed`, and installs `hj_reachability@v0.7.0` strictly
         with `--no-deps` so Blackwell `sm_120` `torch`/`jax` CUDA wheels are never overwritten.
  4. Wall-Clock Watchdog (`--max-wall-minutes`, default 150) + Dead-PID Check:
       - Checks `os.kill(driver_pid, 0)` alongside `heartbeat.json` on every poll so an early
         crash NEVER hangs `while True:` burning `10.05 CU/hr`.
       - Best-effort downloads `driver.log`, `heartbeat.json`, and `artifacts/logs/*.log` even
         on failure before `finally:` tears down ONLY `owned_endpoint` (preserving pre-existing
         external Colab sessions!).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs import WAVE_BUDGETS  # noqa: E402
from scripts.colab.g4_jobs.common import sha256_file, write_canonical_json  # noqa: E402
from scripts.colab.remote_g4_driver import build_wave_commands  # noqa: E402

G4_ESTIMATED_CU_PER_HOUR = 10.05
DEFAULT_SESSION_NAME = "vertirta-m3-g4-thunder"
DEFAULT_MAX_WALL_MINUTES = 150.0
TARBALL_EXCLUDED_TOPLEVEL: frozenset[str] = frozenset(
    {
        ".git",
        "build",
        "artifacts",
        ".pytest_cache",
        ".venv",
        "node_modules",
        "__pycache__",
        "docs",
    }
)

REMOTE_BOOTSTRAP_PACKAGES: tuple[str, ...] = (
    "trl",
    "peft",
    "accelerate",
    "z3-solver",
    "remotezip",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "soundfile",
    "datasets",
    "librosa",
    "jiwer",
    "lxml",
    "pyyaml",
    "timm",
    "tifffile",
    "opencv-python-headless",
)


class ColabBackend(Protocol):
    """Abstract Colab backend enabling deterministic unit testing of try/finally teardown."""

    def list_assignment_endpoints(self) -> set[str]: ...
    def open_g4_session(self, session_name: str) -> None: ...
    def upload_and_run_remote(
        self,
        session_name: str,
        tarball_path: Path,
        remote_work_dir: str,
        remote_cmd: list[str],
        *,
        max_wall_minutes: float = DEFAULT_MAX_WALL_MINUTES,
        poll_interval_s: float = 60.0,
    ) -> dict[str, Any]: ...
    def download_artifacts(
        self,
        session_name: str,
        remote_work_dir: str,
        out_artifacts: Path,
        blob_dir: Path,
    ) -> dict[str, Any]: ...
    def download_logs_best_effort(
        self,
        session_name: str,
        remote_work_dir: str,
        out_artifacts: Path,
    ) -> list[str]: ...
    def stop_session(self, session_name: str) -> None: ...
    def unassign_endpoint(self, endpoint: str) -> None: ...


def resolve_colab_cli_python() -> Path | None:
    """Dynamically resolve the colab_cli virtualenv Python interpreter (zero hardcoded usernames)."""
    colab_bin = shutil.which("colab")
    if colab_bin:
        try:
            first_line = Path(colab_bin).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            if first_line.startswith("#!"):
                interp = Path(first_line[2:].strip().split()[0])
                if interp.is_file():
                    return interp
        except Exception:
            pass
    fallback = Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "bin" / "python"
    return fallback if fallback.is_file() else None


def build_remote_bootstrap_script(remote_work_dir: str) -> str:
    """Return remote Python script installing required M3 deps while protecting Blackwell CUDA wheels."""
    pkgs_json = json.dumps(list(REMOTE_BOOTSTRAP_PACKAGES))
    return "\n".join(
        [
            "import os, subprocess, sys",
            f"os.makedirs({remote_work_dir!r} + '/src', exist_ok=True)",
            f"os.makedirs({remote_work_dir!r} + '/artifacts/logs', exist_ok=True)",
            f"os.makedirs({remote_work_dir!r} + '/blobs', exist_ok=True)",
            f"subprocess.run(['tar', '-xzf', {remote_work_dir!r} + '/src.tar.gz', '-C', {remote_work_dir!r} + '/src'], check=True)",
            "# Remove incompatible torchao preinstalled on Colab (#295 guard)",
            "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)",
            f"pkgs = {pkgs_json}",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade-strategy', 'only-if-needed', *pkgs], check=True)",
            "# Install StanfordASL/hj_reachability v0.7.0 with --no-deps to protect Blackwell sm_120 JAX wheels",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'git+https://github.com/StanfordASL/hj_reachability.git@v0.7.0'], check=True)",
            "import torch, jax, hj_reachability, transformers, trl, peft, z3, onnx, remotezip, datasets, tifffile, cv2",
            "print('BOOTSTRAP_OK', torch.__version__, jax.__version__, 'cuda=', torch.cuda.is_available())",
        ]
    )


def build_remote_poll_script(remote_work_dir: str) -> str:
    """Return remote Python snippet checking BOTH heartbeat.json AND live driver PID via os.kill(pid, 0)."""
    return "\n".join(
        [
            "import json, os, pathlib",
            f"root = pathlib.Path({remote_work_dir!r})",
            "hb_p = root / 'artifacts' / 'heartbeat.json'",
            "pid_p = root / 'driver.pid'",
            "hb = json.loads(hb_p.read_text(encoding='utf-8')) if hb_p.is_file() else {}",
            "pid = int(pid_p.read_text().strip()) if pid_p.is_file() else hb.get('pid')",
            "alive = False",
            "if pid is not None:",
            "    try:",
            "        os.kill(int(pid), 0)",
            "        alive = True",
            "    except OSError:",
            "        alive = False",
            "print(json.dumps({'heartbeat': hb, 'driver_pid': pid, 'driver_alive': alive}))",
        ]
    )


def build_minimal_source_tarball(repo_root: Path, out_tar_gz: Path) -> Path:
    """Pack minimal clean source bundle excluding .git, build/, artifacts/, node_modules/, >2 MB blobs."""
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)

    def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        if not parts:
            return ti
        if parts[0] in TARBALL_EXCLUDED_TOPLEVEL:
            return None
        if "__pycache__" in parts or ".pytest_cache" in parts or ".git" in parts:
            return None
        if ti.isfile() and ti.size > 2 * 1024 * 1024:
            return None
        return ti

    with tarfile.open(out_tar_gz, "w:gz") as tf:
        for child in sorted(repo_root.iterdir()):
            if child.name in TARBALL_EXCLUDED_TOPLEVEL:
                continue
            tf.add(child, arcname=child.name, filter=_filter)
    return out_tar_gz


class LiveColabCliBackend:
    """Real google-colab-cli v0.8.0 + colab_cli.common.state.client backend (Phase 3 ONLY)."""

    def __init__(self, *, runner: Any = subprocess.run) -> None:
        self.colab_bin = shutil.which("colab")
        self.colab_py = resolve_colab_cli_python()
        self._runner = runner

    def _run_colab_py(self, code: str) -> str:
        if self.colab_py is None:
            raise RuntimeError("Could not dynamically resolve colab_cli Python interpreter.")
        res = self._runner(
            [str(self.colab_py), "-c", code],
            text=True,
            capture_output=True,
            check=True,
        )
        return str(res.stdout).strip()

    def _exec_stdin(self, session_name: str, code: str, *, timeout_s: float = 180.0, check: bool = True) -> Any:
        if not self.colab_bin:
            raise RuntimeError("`colab` CLI not found on PATH.")
        return self._runner(
            [self.colab_bin, "exec", "-s", session_name, "--timeout", str(timeout_s)],
            input=code,
            text=True,
            capture_output=True,
            check=check,
        )

    def list_assignment_endpoints(self) -> set[str]:
        out = self._run_colab_py(
            "import json; from colab_cli.common import state; "
            "print(json.dumps([a.endpoint for a in state.client.list_assignments()]))"
        )
        return set(json.loads(out))

    def open_g4_session(self, session_name: str) -> None:
        if not self.colab_bin:
            raise RuntimeError("`colab` CLI not found on PATH.")
        self._runner([self.colab_bin, "new", "--gpu", "G4", "-s", session_name], check=True)

    def upload_and_run_remote(
        self,
        session_name: str,
        tarball_path: Path,
        remote_work_dir: str,
        remote_cmd: list[str],
        *,
        max_wall_minutes: float = DEFAULT_MAX_WALL_MINUTES,
        poll_interval_s: float = 60.0,
    ) -> dict[str, Any]:
        assert self.colab_bin is not None
        # 1) Create remote_work_dir BEFORE colab upload (#329 fix)
        self._exec_stdin(
            session_name,
            f"import os; os.makedirs({remote_work_dir!r}, exist_ok=True); print('MKDIR_OK')",
            timeout_s=120.0,
            check=True,
        )
        # 2) Upload source bundle
        self._runner(
            [self.colab_bin, "upload", "-s", session_name, str(tarball_path), f"{remote_work_dir}/src.tar.gz"],
            check=True,
        )
        # 3) Bootstrap dependencies (uninstall torchao, install deps, hj_reachability --no-deps)
        self._exec_stdin(
            session_name,
            build_remote_bootstrap_script(remote_work_dir),
            timeout_s=600.0,
            check=True,
        )
        # 4) Launch detached remote_g4_driver recording driver.pid
        launch_code = "\n".join(
            [
                "import os, pathlib, subprocess",
                f"log_f = open({remote_work_dir!r} + '/driver.log', 'wb')",
                f"p = subprocess.Popen({remote_cmd!r}, cwd={remote_work_dir!r} + '/src', stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)",
                f"pathlib.Path({remote_work_dir!r} + '/driver.pid').write_text(str(p.pid), encoding='utf-8')",
                "print('LAUNCHED_PID', p.pid)",
            ]
        )
        self._exec_stdin(session_name, launch_code, timeout_s=120.0, check=True)

        # 5) Resilient poll with wall-clock ceiling & dead-PID watchdog (#329 fix)
        poll_code = build_remote_poll_script(remote_work_dir)
        t_start = time.monotonic()
        max_wall_s = max(1.0, float(max_wall_minutes) * 60.0)
        dead_grace_polls = 0
        while True:
            if (time.monotonic() - t_start) > max_wall_s:
                raise TimeoutError(
                    f"Remote G4 thunder run exceeded --max-wall-minutes={max_wall_minutes}; triggering teardown."
                )
            time.sleep(poll_interval_s)
            poll = self._exec_stdin(session_name, poll_code, timeout_s=120.0, check=False)
            if poll.returncode == 0 and poll.stdout.strip():
                lines = [ln.strip() for ln in poll.stdout.strip().splitlines() if ln.strip().startswith("{")]
                if lines:
                    try:
                        status = json.loads(lines[-1])
                    except json.JSONDecodeError:
                        continue
                    hb = status.get("heartbeat") or {}
                    hb_state = hb.get("state")
                    if hb_state in ("DONE", "FAILED"):
                        if hb_state == "FAILED":
                            raise RuntimeError(f"Remote G4 driver reported FAILED: {hb}")
                        return hb
                    if not status.get("driver_alive", False):
                        dead_grace_polls += 1
                        if dead_grace_polls >= 2:
                            raise RuntimeError(
                                f"Remote G4 driver PID {status.get('driver_pid')} exited prematurely without DONE state (heartbeat={hb})."
                            )
                    else:
                        dead_grace_polls = 0

    def download_logs_best_effort(
        self,
        session_name: str,
        remote_work_dir: str,
        out_artifacts: Path,
    ) -> list[str]:
        if not self.colab_bin:
            return []
        logs_local = out_artifacts / "remote_logs"
        logs_local.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        candidates = [
            ("driver.log", f"{remote_work_dir}/driver.log"),
            ("heartbeat.json", f"{remote_work_dir}/artifacts/heartbeat.json"),
            ("g4_thunder_telemetry.json", f"{remote_work_dir}/artifacts/g4_thunder_telemetry.json"),
            ("remote_bundle_manifest.json", f"{remote_work_dir}/artifacts/remote_bundle_manifest.json"),
        ]
        for job_key in ("qwen3_v3", "whisper_bvi", "forward_wire", "jax_4d_hires_lc", "pidnet_v21_xfeat"):
            candidates.append((f"{job_key}.log", f"{remote_work_dir}/artifacts/logs/{job_key}.log"))
        for fname, rpath in candidates:
            dst = logs_local / fname
            res = self._runner(
                [self.colab_bin, "download", "-s", session_name, rpath, str(dst)],
                check=False,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and dst.is_file():
                downloaded.append(fname)
        return downloaded

    def download_artifacts(
        self,
        session_name: str,
        remote_work_dir: str,
        out_artifacts: Path,
        blob_dir: Path,
    ) -> dict[str, Any]:
        assert self.colab_bin is not None
        out_artifacts.mkdir(parents=True, exist_ok=True)
        blob_dir.mkdir(parents=True, exist_ok=True)
        manifest_local = out_artifacts / "remote_bundle_manifest.json"
        self._runner(
            [self.colab_bin, "download", "-s", session_name, f"{remote_work_dir}/artifacts/remote_bundle_manifest.json", str(manifest_local)],
            check=True,
        )
        manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
        for jf in manifest.get("json_artifacts", []):
            dst = out_artifacts / jf["filename"]
            self._runner(
                [self.colab_bin, "download", "-s", session_name, f"{remote_work_dir}/artifacts/{jf['filename']}", str(dst)],
                check=True,
            )
            if sha256_file(dst) != jf["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch after downloading {jf['filename']}")
        for bf in manifest.get("blob_artifacts", []):
            dst = blob_dir / bf["relpath"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            self._runner(
                [self.colab_bin, "download", "-s", session_name, f"{remote_work_dir}/blobs/{bf['relpath']}", str(dst)],
                check=True,
            )
            if sha256_file(dst) != bf["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch after downloading blob {bf['relpath']}")
        return manifest

    def stop_session(self, session_name: str) -> None:
        if self.colab_bin:
            self._runner([self.colab_bin, "stop", "-s", session_name], check=False)

    def unassign_endpoint(self, endpoint: str) -> None:
        self._run_colab_py(
            f"from colab_cli.common import state; state.client.unassign({endpoint!r})"
        )


def execute_single_shot_lifecycle(
    backend: ColabBackend,
    *,
    session_name: str,
    out_artifacts: Path,
    blob_dir: Path,
    tarball_path: Path,
    remote_work_dir: str = "/content/vertirta_v400_g4",
    max_wall_minutes: float = DEFAULT_MAX_WALL_MINUTES,
    jobs: str | None = None,
    vertibench_v2_dir: str | None = None,
) -> dict[str, Any]:
    """Single-open / single-close G4 lifecycle preserving pre-existing external sessions."""
    t_open = time.perf_counter()
    blob_dir.mkdir(parents=True, exist_ok=True)
    out_artifacts.mkdir(parents=True, exist_ok=True)
    lockfile = blob_dir / ".active_g4_endpoint.json"

    pre_existing = backend.list_assignment_endpoints()
    owned_endpoint: str | None = None
    run_error: str | None = None
    remote_manifest: dict[str, Any] = {}
    heartbeat_final: dict[str, Any] = {}
    downloaded_logs: list[str] = []

    try:
        backend.open_g4_session(session_name)
        after_open = backend.list_assignment_endpoints()
        new_endpoints = sorted(after_open - pre_existing)
        if not new_endpoints:
            raise RuntimeError("No new Colab G4 endpoint appeared after open_g4_session().")
        owned_endpoint = new_endpoints[0]
        lockfile.write_text(
            json.dumps(
                {
                    "session_name": session_name,
                    "owned_endpoint": owned_endpoint,
                    "pre_existing_endpoints": sorted(pre_existing),
                    "opened_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        remote_cmd = [
            "python3",
            "scripts/colab/remote_g4_driver.py",
            "--artifacts-dir",
            f"{remote_work_dir}/artifacts",
            "--blob-dir",
            f"{remote_work_dir}/blobs",
        ]
        if jobs:
            remote_cmd.extend(["--jobs", jobs])
        if vertibench_v2_dir:
            remote_cmd.extend(["--vertibench-v2-dir", vertibench_v2_dir])

        heartbeat_final = backend.upload_and_run_remote(
            session_name,
            tarball_path,
            remote_work_dir,
            remote_cmd,
            max_wall_minutes=max_wall_minutes,
        )
        remote_manifest = backend.download_artifacts(
            session_name, remote_work_dir, out_artifacts, blob_dir
        )
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Best-effort forensic log capture BEFORE terminating G4 (#329 fix)
        if owned_endpoint is not None:
            try:
                downloaded_logs = backend.download_logs_best_effort(
                    session_name, remote_work_dir, out_artifacts
                )
            except Exception:
                pass
        # Guaranteed single-close teardown of ONLY owned_endpoint (never pre_existing!)
        try:
            backend.stop_session(session_name)
        except Exception:
            pass
        if owned_endpoint is not None:
            remaining_mid = backend.list_assignment_endpoints()
            if owned_endpoint in remaining_mid:
                backend.unassign_endpoint(owned_endpoint)
        remaining_final = backend.list_assignment_endpoints()
        owned_released = (owned_endpoint is None) or (owned_endpoint not in remaining_final)
        elapsed_s = time.perf_counter() - t_open
        if lockfile.is_file() and owned_released:
            lockfile.unlink()

        summary = {
            "schema_version": "vertirta.g4_thunder_run_manifest/v4.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_name": session_name,
            "owned_endpoint": owned_endpoint,
            "pre_existing_endpoints_preserved": sorted(pre_existing & remaining_final),
            "remaining_active_assignments": len(remaining_final),
            "owned_endpoint_released": owned_released,
            "teardown_verified": bool(owned_released),
            "downloaded_remote_logs": downloaded_logs,
            "elapsed_wall_s": round(elapsed_s, 3),
            "estimated_cu_consumed": round((elapsed_s / 3600.0) * G4_ESTIMATED_CU_PER_HOUR, 4),
            "heartbeat_final": heartbeat_final,
            "remote_manifest_state": remote_manifest.get("state"),
            "run_error": run_error,
        }
        write_canonical_json(out_artifacts / "g4_thunder_run_manifest.json", summary)

    return summary


def run_local_preflight(repo_root: Path) -> None:
    """Mandatory 0 CU local pre-flight test gate before opening any Colab G4."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/colab/", "-v"],
        cwd=str(repo_root),
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Pre-G4 local pre-flight failed (rc={res.returncode}); aborting at 0.00 CU.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VertiRTA M3 Single-Shot 80-88 GB Colab G4 Batch Thunder Run Orchestrator"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true", help="0 CU config/plan audit.")
    mode_group.add_argument("--smoke-cpu", action="store_true", help="0 CU full local Wave 1 -> Barrier -> Wave 2 CPU smoke.")
    mode_group.add_argument("--live-g4", action="store_true", help="Phase 3 live single-shot G4 run (requires --confirm-single-shot-g4).")
    parser.add_argument(
        "--confirm-single-shot-g4",
        action="store_true",
        help="Mandatory explicit safety confirmation required alongside --live-g4.",
    )
    parser.add_argument("--session-name", type=str, default=DEFAULT_SESSION_NAME)
    parser.add_argument("--out-artifacts", type=Path, default=Path("artifacts/v4.0.0"))
    parser.add_argument("--blob-dir", type=Path, default=Path("/var/tmp/vertirta_v400_blobs"))
    parser.add_argument("--vertibench-v2-dir", type=Path, default=None)
    parser.add_argument("--jobs", type=str, default=None, help="Optional comma-separated job filter.")
    parser.add_argument(
        "--max-wall-minutes",
        type=float,
        default=DEFAULT_MAX_WALL_MINUTES,
        help=f"Hard wall-clock watchdog ceiling in minutes (default: {DEFAULT_MAX_WALL_MINUTES}).",
    )
    args = parser.parse_args()

    if args.dry_run:
        waves = build_wave_commands(
            repo_root=REPO_ROOT,
            artifacts_dir=args.out_artifacts,
            blob_dir=args.blob_dir,
            mode="dry_run",
            jobs_filter=set(args.jobs.split(",")) if args.jobs else None,
            vertibench_v2_dir=args.vertibench_v2_dir,
        )
        colab_py = resolve_colab_cli_python()
        report = {
            "schema_version": "vertirta.g4_thunder_run_manifest/v4.0.0",
            "mode": "DRY_RUN",
            "cu_consumed": 0.0,
            "max_wall_minutes": float(args.max_wall_minutes),
            "wave_budgets": WAVE_BUDGETS,
            "wave1_vram_sum": round(sum(WAVE_BUDGETS["wave1"].values()), 4),
            "wave2_vram_sum": round(sum(WAVE_BUDGETS["wave2"].values()), 4),
            "colab_cli_python_detected": colab_py is not None,
            "waves": waves,
        }
        print(json.dumps(report, indent=2))
        return 0

    if args.smoke_cpu:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/colab/remote_g4_driver.py"),
            "--smoke-cpu",
            "--artifacts-dir",
            str(args.out_artifacts),
            "--blob-dir",
            str(args.blob_dir),
        ]
        if args.jobs:
            cmd.extend(["--jobs", args.jobs])
        if args.vertibench_v2_dir:
            cmd.extend(["--vertibench-v2-dir", str(args.vertibench_v2_dir)])
        subprocess.run(cmd, check=True)
        return 0

    if not args.confirm_single_shot_g4:
        raise SystemExit("Refusing --live-g4 without explicit --confirm-single-shot-g4 lock (0.00 CU spent).")

    run_local_preflight(REPO_ROOT)
    tarball = args.blob_dir / "vertirta_m3_src_bundle.tar.gz"
    build_minimal_source_tarball(REPO_ROOT, tarball)
    backend = LiveColabCliBackend()
    execute_single_shot_lifecycle(
        backend,
        session_name=args.session_name,
        out_artifacts=args.out_artifacts,
        blob_dir=args.blob_dir,
        tarball_path=tarball,
        max_wall_minutes=args.max_wall_minutes,
        jobs=args.jobs,
        vertibench_v2_dir=str(args.vertibench_v2_dir) if args.vertibench_v2_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
