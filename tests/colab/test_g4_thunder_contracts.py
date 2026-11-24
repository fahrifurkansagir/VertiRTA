# SPDX-License-Identifier: Apache-2.0
"""Plan 6C G4 Thunder Run contract, anti-padding, teardown safety, and 0-CU smoke suite."""

from __future__ import annotations

from argparse import Namespace
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts.bench_gtx1650s_m3_timesliced import select_platform_cuda_provider_options
from scripts.colab.g4_jobs.common import (
    CANONICAL_SNR_STRATA,
    WAVE_BUDGETS,
    validate_job_license_manifest,
)
from scripts.colab.g4_jobs.job_forward_wire import (
    build_forward_license_manifest,
    ingest_real_ttpla_apache2,
)
from scripts.colab.g4_jobs.job_jax_4d_hires_and_lc import (
    LC_CD_R,
    build_lift_cruise_regimes,
)
from scripts.colab.g4_jobs.job_pidnet_v21_xfeat import (
    JPEG_MAGIC,
    PNG_MAGIC,
    apply_oblique_perspective_aug,
    is_valid_skyscenes_image_member,
    select_pareto_checkpoint,
)
from scripts.colab.remote_g4_driver import run_driver, wait_for_gpu_drain_barrier
from scripts.colab.run_g4_m3_thunder_run import (
    LiveColabCliBackend,
    build_remote_bootstrap_script,
    build_remote_poll_script,
    execute_single_shot_lifecycle,
    resolve_colab_cli_python,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_wave_vram_fractions_and_canonical_snr_strata() -> None:
    assert sum(WAVE_BUDGETS["wave1"].values()) == pytest.approx(0.88, abs=1e-6)
    assert sum(WAVE_BUDGETS["wave2"].values()) == pytest.approx(0.88, abs=1e-6)
    assert sum(WAVE_BUDGETS["wave1"].values()) <= 0.90
    assert sum(WAVE_BUDGETS["wave2"].values()) <= 0.90
    assert CANONICAL_SNR_STRATA == ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")


@pytest.mark.parametrize(
    "banned_name,banned_lic",
    [
        ("UWB-ATCC", "CC BY-NC-SA 4.0"),
        ("ATCO2", "CC BY-NC-ND 4.0"),
        ("UAVid", "CC BY-NC-SA 4.0"),
        ("InsPLAD", "CC BY-NC 3.0"),
        ("DINOv2-SALAD", "GPL-3.0"),
        ("tr_TR-fahrettin-medium", "CC0"),
        ("piper1-gpl", "GPL-3.0"),
        ("espeak-ng", "GPL-3.0"),
        ("BlenderProc", "GPL-3.0"),
        ("OpenVINS", "GPL-3.0"),
    ],
)
def test_license_firewall_blocks_global_hard_bans(banned_name: str, banned_lic: str) -> None:
    with pytest.raises(ValueError):
        validate_job_license_manifest(
            job_name="job_test",
            train_sources=[{"name": banned_name, "license": banned_lic}],
            eval_only_sources=[],
            models=[],
        )
    # Global bans must ALSO be rejected even in eval_only_sources (e.g. UAVid)
    with pytest.raises(ValueError):
        validate_job_license_manifest(
            job_name="job_test",
            train_sources=[],
            eval_only_sources=[
                {"name": banned_name, "license": banned_lic, "quarantined_zero_shot_eval_only": True}
            ],
            models=[],
        )


def test_license_firewall_separates_train_vs_eval_uavscenes_and_forward_only_ttpla() -> None:
    # AeroScapes / ICG / UAVScenes MUST be rejected in train_sources...
    for ds in ("AeroScapes", "ICG", "UAVScenes"):
        with pytest.raises(ValueError):
            validate_job_license_manifest(
                job_name="job_pidnet_v21_xfeat",
                train_sources=[{"name": ds, "license": "CC BY 4.0"}],
                eval_only_sources=[],
                models=[],
            )
    # ...and permitted strictly in eval_only_sources when explicitly quarantined!
    rep = validate_job_license_manifest(
        job_name="job_pidnet_v21_xfeat",
        train_sources=[{"name": "FLAIR-1", "license": "etalab-2.0"}],
        eval_only_sources=[
            {"name": "UAVScenes", "license": "CC BY-NC-SA 4.0", "quarantined_zero_shot_eval_only": True},
            {"name": "AeroScapes", "license": "CC BY-SA 4.0", "quarantined_zero_shot_eval_only": True},
        ],
        models=[{"name": "PIDNet-S", "license": "MIT"}],
    )
    assert rep["zero_sharealike_verified"] is True

    # TTPLA allowed ONLY in job_forward_wire, forbidden inside nadir SLZ
    with pytest.raises(ValueError, match="TTPLA"):
        validate_job_license_manifest(
            job_name="job_pidnet_v21_xfeat",
            train_sources=[{"name": "TTPLA", "license": "Apache-2.0"}],
            eval_only_sources=[],
            models=[],
        )
    rep_fwd = validate_job_license_manifest(
        job_name="job_forward_wire",
        train_sources=[{"name": "TTPLA", "license": "Apache-2.0"}],
        eval_only_sources=[],
        models=[{"name": "M-LSD-WireSeg", "license": "Apache-2.0"}],
    )
    assert rep_fwd["zero_nc_sa_gpl_verified"] is True


def test_ast_audit_forbids_dummy_vram_padding_and_personal_paths() -> None:
    targets = [
        REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py",
        REPO_ROOT / "scripts/colab/remote_g4_driver.py",
        REPO_ROOT / "scripts/colab/g4_jobs/common.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_forward_wire.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_jax_4d_hires_and_lc.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py",
        REPO_ROOT / "scripts/bench_gtx1650s_m3_timesliced.py",
    ]
    forbidden_tokens = ("dummy_vram", "pad_vram", "reserve_vram", "/home/dias", "C:\\Users\\")
    for p in targets:
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM found in {p}"
        assert b"\r" not in raw, f"CR byte found in {p}"
        text = raw.decode("utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"Forbidden token {tok!r} found in {p}"
        tree = ast.parse(text, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "empty", f"Forbidden torch.empty/np.empty filler call in {p}"


def test_platform_aware_auto_cudnn_tuning_lock() -> None:
    win_opts = select_platform_cuda_provider_options(system_name="Windows", enable_cuda_graph=True)
    assert win_opts["cudnn_conv_algo_search"] == "EXHAUSTIVE"
    assert win_opts["cudnn_conv_use_max_workspace"] == "1"
    assert win_opts["enable_cuda_graph"] == "1"

    lin_opts = select_platform_cuda_provider_options(system_name="Linux", enable_cuda_graph=True)
    assert lin_opts["cudnn_conv_algo_search"] == "HEURISTIC"
    assert lin_opts["cudnn_conv_use_max_workspace"] == "0"
    assert lin_opts["enable_cuda_graph"] == "1"


def test_skyscenes_nested_tar_filter_and_pareto_selector() -> None:
    assert is_valid_skyscenes_image_member(512_000, PNG_MAGIC + b"rest") is True
    assert is_valid_skyscenes_image_member(512_000, JPEG_MAGIC + b"rest") is True
    # Rejects 1.83 GB nested tar bug (#285) and non-image magic
    assert is_valid_skyscenes_image_member(1_830_000_000, PNG_MAGIC + b"rest") is False
    assert is_valid_skyscenes_image_member(100_000, b"././@LongLink") is False

    res = select_pareto_checkpoint(
        [
            {"epoch": 1, "val_miou": 0.56, "val_fsr": 0.048},
            {"epoch": 2, "val_miou": 0.52, "val_fsr": 0.025},
            {"epoch": 3, "val_miou": 0.49, "val_fsr": 0.018},
        ],
        max_fsr=0.030,
    )
    assert res["fsr_gate_met"] is True
    assert res["selected"]["epoch"] == 2


def test_gpu_drain_barrier_polls_until_zero_pids_and_low_mem() -> None:
    seq = iter(
        [
            {"available": True, "memory_used_mib": 42000.0, "compute_apps": [{"pid": 123}]},
            {"available": True, "memory_used_mib": 1800.0, "compute_apps": []},
            {"available": True, "memory_used_mib": 420.0, "compute_apps": []},
        ]
    )
    info = wait_for_gpu_drain_barrier(
        timeout_s=5.0, poll_s=0.01, snapshot_fn=lambda: next(seq), smoke_mode=False
    )
    assert info["drained"] is True
    assert info["remaining_pids"] == 0
    assert info["memory_used_mib"] < 1500.0


def test_single_shot_teardown_preserves_preexisting_endpoint_on_crash(tmp_path: Path) -> None:
    class _FakeColabBackend:
        def __init__(self) -> None:
            self.active: set[str] = {"gpu-g4-external-preexisting-001"}
            self.unassigned: list[str] = []
            self.stopped: list[str] = []
            self.events: list[str] = []
            self.last_remote_cmd: list[str] = []

        def list_assignment_endpoints(self) -> set[str]:
            return set(self.active)

        def open_g4_session(self, session_name: str) -> None:
            self.events.append("open")
            self.active.add("gpu-g4-ours-002")

        def upload_and_run_remote(
            self,
            session_name: str,
            tarball_path: Path,
            remote_work_dir: str,
            remote_cmd: list[str],
            *,
            max_wall_minutes: float = 150.0,
            poll_interval_s: float = 60.0,
        ) -> dict[str, Any]:
            self.last_remote_cmd = list(remote_cmd)
            self.events.append("run")
            raise RuntimeError("Simulated mid-wave remote crash")

        def download_artifacts(
            self, session_name: str, remote_work_dir: str, out_artifacts: Path, blob_dir: Path
        ) -> dict[str, Any]:
            return {}

        def download_logs_best_effort(
            self, session_name: str, remote_work_dir: str, out_artifacts: Path
        ) -> list[str]:
            self.events.append("download_logs")
            return ["driver.log", "heartbeat.json"]

        def stop_session(self, session_name: str) -> None:
            self.events.append("stop")
            self.stopped.append(session_name)
            # Simulate `colab stop` failing to release backend allocation so explicit unassign() MUST run!

        def unassign_endpoint(self, endpoint: str) -> None:
            self.events.append("unassign")
            self.unassigned.append(endpoint)
            self.active.discard(endpoint)

    backend = _FakeColabBackend()
    out_art = tmp_path / "artifacts"
    blob_dir = tmp_path / "blobs"
    fake_tar = tmp_path / "src.tar.gz"
    fake_tar.write_bytes(b"tar")

    with pytest.raises(RuntimeError, match="Simulated mid-wave remote crash"):
        execute_single_shot_lifecycle(
            backend,
            session_name="vertirta-test-g4",
            out_artifacts=out_art,
            blob_dir=blob_dir,
            tarball_path=fake_tar,
            jobs="forward_wire,jax_4d_hires_lc",
            vertibench_v2_dir="/content/vb2",
        )

    # MUST capture logs BEFORE teardown, unassign ONLY our endpoint, NEVER touch pre-existing!
    assert backend.events == ["open", "run", "download_logs", "stop", "unassign"]
    assert "--jobs" in backend.last_remote_cmd
    assert "--vertibench-v2-dir" in backend.last_remote_cmd
    assert backend.stopped == ["vertirta-test-g4"]
    assert backend.unassigned == ["gpu-g4-ours-002"]
    assert backend.active == {"gpu-g4-external-preexisting-001"}

    manifest = json.loads((out_art / "g4_thunder_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["owned_endpoint"] == "gpu-g4-ours-002"
    assert manifest["owned_endpoint_released"] is True
    assert manifest["teardown_verified"] is True
    assert manifest["downloaded_remote_logs"] == ["driver.log", "heartbeat.json"]
    assert manifest["pre_existing_endpoints_preserved"] == ["gpu-g4-external-preexisting-001"]
    assert manifest["remaining_active_assignments"] == 1
    assert not (blob_dir / ".active_g4_endpoint.json").exists()


def test_live_colab_cli_backend_v080_api_stdin_bootstrap_and_dead_pid_watchdog(tmp_path: Path) -> None:
    """Verify Showstopper #1 fixes against google-colab-cli v0.8.0 contract at 0.00 CU."""
    calls: list[tuple[list[str], str | None]] = []
    poll_counter = {"n": 0}

    def _mock_runner(cmd: list[str], **kwargs: Any) -> Any:
        code_in = kwargs.get("input")
        calls.append((list(cmd), code_in))
        if len(cmd) >= 3 and cmd[1] == "-c":
            code = cmd[2]
            assert "from colab_cli.common import state" in code
            assert "get_credentials" not in code
            if "list_assignments" in code:
                return subprocess.CompletedProcess(cmd, 0, stdout='["ep-1"]\n', stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if len(cmd) >= 2 and cmd[1] == "exec":
            # google-colab-cli v0.8.0 takes NO positional code arg; code MUST arrive via stdin!
            assert len(cmd) == 6 and cmd[2] == "-s" and cmd[4] == "--timeout"
            assert isinstance(code_in, str) and len(code_in) > 0
            if "driver_alive" in code_in:
                poll_counter["n"] += 1
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"heartbeat": {"state": "RUNNING_WAVE1"}, "driver_pid": 9999, "driver_alive": false}\n',
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="OK\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    backend = LiveColabCliBackend(runner=_mock_runner)
    backend.colab_bin = "colab"
    backend.colab_py = Path("/usr/bin/python3")

    assert backend.list_assignment_endpoints() == {"ep-1"}
    backend.unassign_endpoint("ep-1")

    tar_p = tmp_path / "src.tar.gz"
    tar_p.write_bytes(b"tar")
    with pytest.raises(RuntimeError, match="exited prematurely"):
        backend.upload_and_run_remote(
            "sess-1",
            tar_p,
            "/content/work",
            ["python3", "scripts/colab/remote_g4_driver.py"],
            max_wall_minutes=5.0,
            poll_interval_s=0.001,
        )

    # Confirm exact order: mkdir BEFORE colab upload -> bootstrap -> launch -> 2 dead-PID polls
    exec_upload_seq = [c[0][1] for c in calls if c[0][0] == "colab"]
    assert exec_upload_seq[:4] == ["exec", "upload", "exec", "exec"]
    mkdir_stdin = [c[1] for c in calls if c[0][0] == "colab" and c[0][1] == "exec"][0]
    assert mkdir_stdin is not None and "os.makedirs" in mkdir_stdin
    bootstrap_stdin = [c[1] for c in calls if c[0][0] == "colab" and c[0][1] == "exec"][1]
    assert bootstrap_stdin is not None
    assert "pip', 'uninstall', '-y', 'torchao'" in bootstrap_stdin
    assert "'--no-deps', 'git+https://github.com/StanfordASL/hj_reachability.git@v0.7.0'" in bootstrap_stdin
    assert "tifffile" in bootstrap_stdin
    assert "opencv-python-headless" in bootstrap_stdin
    assert poll_counter["n"] == 2

    # Confirm bootstrap & poll script builders directly as well
    assert "torchao" in build_remote_bootstrap_script("/content/w")
    assert "os.kill(int(pid), 0)" in build_remote_poll_script("/content/w")

    # Confirm wall-clock ceiling triggers TimeoutError watchdog (prevents runaway CU burn!)
    import scripts.colab.run_g4_m3_thunder_run as orch_mod

    tick = iter([0.0, 99999.0])
    orig_monotonic = orch_mod.time.monotonic
    orch_mod.time.monotonic = lambda: next(tick)
    try:
        with pytest.raises(TimeoutError, match="max-wall-minutes"):
            backend.upload_and_run_remote(
                "sess-timeout",
                tar_p,
                "/content/work",
                ["python3", "scripts/colab/remote_g4_driver.py"],
                max_wall_minutes=1.0,
                poll_interval_s=0.001,
            )
    finally:
        orch_mod.time.monotonic = orig_monotonic


def test_remote_driver_flushes_failed_heartbeat_and_manifest_on_drain_timeout(tmp_path: Path) -> None:
    """Verify remote_g4_driver try/finally flushes FAILED heartbeat + manifest on drain timeout."""
    art_dir = tmp_path / "artifacts"
    blob_dir = tmp_path / "blobs"
    args = Namespace(
        dry_run=False,
        smoke_cpu=False,
        artifacts_dir=art_dir,
        blob_dir=blob_dir,
        vertibench_v2_dir=None,
        jobs="__nonexistent_filter__",
    )
    busy_gpu = lambda: {"available": True, "memory_used_mib": 40000.0, "compute_apps": [{"pid": 777}]}
    with pytest.raises(RuntimeError, match="GPU drain barrier timed out"):
        run_driver(args, drain_snapshot_fn=busy_gpu, drain_timeout_s=0.02)

    hb = json.loads((art_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert hb["state"] == "FAILED"
    assert hb["pid"] == os.getpid()
    assert "timed out" in str(hb["error"])
    assert (art_dir / "g4_thunder_telemetry.json").is_file()
    manifest = json.loads((art_dir / "remote_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "FAILED"
    assert "timed out" in str(manifest["error"])


def test_lift_cruise_streamlined_drag_and_forward_wire_honest_ttpla_ingestion(tmp_path: Path) -> None:
    """Verify Showstoppers #3 & #4: streamlined LC cd_r, TTPLA LabelMe rasterizer, honest license audit."""
    # Showstopper #4: Lift+Cruise streamlined fixed-wing drag (NOT X8 flat-plate 0.00408!)
    lc_regimes = build_lift_cruise_regimes()
    assert len(lc_regimes) == 4
    assert LC_CD_R == pytest.approx(0.0004244, rel=1e-2)
    for reg in lc_regimes:
        assert reg.cd_r == pytest.approx(LC_CD_R, rel=1e-6)
        assert reg.cd_r < 0.001

    # Showstopper #3: Honest license manifest scopes out uningested corpora
    ts_none, _, _ = build_forward_license_manifest(ingested_ttpla=False)
    assert [s["name"] for s in ts_none] == ["VertiRTA_Procedural_Catenary_Horizon"]
    ts_ttpla, _, _ = build_forward_license_manifest(ingested_ttpla=True)
    assert [s["name"] for s in ts_ttpla] == ["TTPLA", "VertiRTA_Procedural_Catenary_Horizon"]

    # Showstopper #3: Real LabelMe TTPLA pair rasterization
    from PIL import Image

    data_dir = tmp_path / "ttpla_local"
    data_dir.mkdir()
    Image.fromarray(np.full((128, 128, 3), 180, dtype=np.uint8)).save(data_dir / "sample_01.png")
    (data_dir / "sample_01.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {"label": "cable", "points": [[10.0, 40.0], [118.0, 48.0]]},
                    {"label": "tower_lattice", "points": [[50.0, 20.0], [70.0, 20.0], [70.0, 110.0], [50.0, 110.0]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    recs, info = ingest_real_ttpla_apache2(tmp_path / "work", data_dir=data_dir, max_samples=5)
    assert len(recs) == 1
    assert info["frames_ingested"] == 1
    img_chw, wire_mask, junc_map, boxes, cls_ids = recs[0]
    assert img_chw.shape == (3, 640, 640)
    assert wire_mask.shape == (1, 640, 640) and float(wire_mask.max()) > 0.5
    assert float(junc_map[0].max()) == 1.0
    assert int(cls_ids[0]) == 2 and float(boxes[0, 2]) > float(boxes[0, 0])

    # Showstopper #2: Real PIL oblique perspective augmentation preserves 512x512 shape
    rgb_p = Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8))
    msk_p = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    nrm_p = Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8))
    rgb_w, msk_w, nrm_w = apply_oblique_perspective_aug(rgb_p, msk_p, nrm_p)
    assert rgb_w.size == (512, 512) and msk_w.size == (512, 512) and nrm_w is not None and nrm_w.size == (512, 512)


def test_orchestrator_refuses_live_g4_without_double_confirm_and_dry_run_passes() -> None:
    _ = resolve_colab_cli_python()  # Must never crash even if colab absent
    r_refuse = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py"), "--live-g4"],
        capture_output=True,
        text=True,
    )
    assert r_refuse.returncode != 0
    assert "confirm-single-shot-g4" in (r_refuse.stderr + r_refuse.stdout)

    r_dry = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py"), "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    dry = json.loads(r_dry.stdout)
    assert dry["mode"] == "DRY_RUN"
    assert dry["cu_consumed"] == 0.0
    assert dry["wave1_vram_sum"] == pytest.approx(0.88, abs=1e-4)
    assert dry["wave2_vram_sum"] == pytest.approx(0.88, abs=1e-4)


def test_end_to_end_smoke_cpu_pipeline_and_tu116_timesliced_sha256_gate(tmp_path: Path) -> None:
    """0-CU local end-to-end smoke of Wave-1/Wave-2 driver, 3 owned jobs, ONNX exports, LUTs & bench."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    art_dir = tmp_path / "artifacts"
    blob_dir = tmp_path / "blobs"

    # 1) Run Wave 1 -> Drain Barrier -> Wave 2 locally via orchestrator --smoke-cpu (0 CU!)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py"),
            "--smoke-cpu",
            "--out-artifacts",
            str(art_dir),
            "--blob-dir",
            str(blob_dir),
            "--jobs",
            "forward_wire,jax_4d_hires_lc,pidnet_v21_xfeat",
        ],
        check=True,
    )

    bundle = json.loads((art_dir / "remote_bundle_manifest.json").read_text(encoding="utf-8"))
    assert bundle["state"] == "DONE"
    assert bundle["drain_barrier"]["drained"] is True
    assert len(bundle["job_results"]) == 3

    # Verify JAX 2.5:1 projection soundness, streamlined LC drag, sequential mode & dual VRTALUT1/VRTALUT2 exports
    jax_m = json.loads((art_dir / "g4_jax_4d_hires_and_lc_manifest.json").read_text(encoding="utf-8"))
    assert jax_m["execution_mode"] == "sequential_per_regime_oom_safe"
    assert jax_m["lift_cruise_cd_r"] < 0.001
    assert jax_m["smoke_ratio_2p5_audit"]["passed"] is True
    assert jax_m["artifacts"]["x8_240_vrtalut1"]["size_bytes"] == 11313864
    assert jax_m["artifacts"]["x8_240_vrtalut1"]["magic"] == "VRTALUT1"
    assert jax_m["artifacts"]["x8_240_vrtalut2_hires"]["magic"] == "VRTALUT2"
    assert jax_m["artifacts"]["lift_cruise_8p1_vrtalut1"]["size_bytes"] == 11313864
    assert jax_m["x8_soundness_passed"] is True
    assert jax_m["lc_soundness_passed"] is True
    assert jax_m["soundness_passed"] is True

    # Verify Forward Wire honest license scoping + held-out validation metrics
    fwd_m = json.loads((art_dir / "g4_forward_wire_metrics.json").read_text(encoding="utf-8"))
    assert fwd_m["dataset_ingestion"]["real_ttpla_frames_ingested"] == 0
    assert fwd_m["license_audit"]["train_source_names"] == ["VertiRTA_Procedural_Catenary_Horizon"]
    assert fwd_m["heldout_validation"]["val_samples"] >= 1

    # Verify all 4 Opset-17 ONNX graphs exist and match Plan 6C specs
    pid_m = json.loads((art_dir / "g4_pidnet_v21_xfeat_metrics.json").read_text(encoding="utf-8"))
    assert pid_m["onnx_exports"]["pidnet_slz_v21"]["deploy_parameters"] == 8003817
    assert pid_m["kartal_reloc_eval"]["cep50_m"] >= 0.0
    assert pid_m["kartal_reloc_eval"]["matcher_backend"]

    # 2) Run time-sliced edge benchmark in --smoke-cpu mode (all 4 ONNX models verified by SHA-256!)
    bench_json = art_dir / "local_gtx1650s_m3_timesliced_proof.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/bench_gtx1650s_m3_timesliced.py"),
            "--smoke-cpu",
            "--blob-dir",
            str(blob_dir),
            "--artifacts-dir",
            str(art_dir),
            "--out-json",
            str(bench_json),
        ],
        check=True,
    )
    bench = json.loads(bench_json.read_text(encoding="utf-8"))
    assert bench["gates"]["onnx_sha256_all_verified"] is True
    assert bench["gates"]["all_trained_outputs_finite_and_nondegenerate"] is True

    # 3) Tamper 1 byte of xfeat_512_opset17.onnx -> SHA-256 parity check MUST fail closed!
    xfeat_path = blob_dir / "xfeat_512_opset17.onnx"
    raw = bytearray(xfeat_path.read_bytes())
    raw[-1] ^= 0x55
    xfeat_path.write_bytes(bytes(raw))
    tamper_res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/bench_gtx1650s_m3_timesliced.py"),
            "--smoke-cpu",
            "--blob-dir",
            str(blob_dir),
            "--artifacts-dir",
            str(art_dir),
            "--out-json",
            str(art_dir / "tamper_fail.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert tamper_res.returncode != 0
    assert "SHA-256 mismatch" in (tamper_res.stderr + tamper_res.stdout)
