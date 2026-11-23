# SPDX-License-Identifier: Apache-2.0
"""Contract and scientific-integrity tests for v3.1.0 Track 2 (Colab L4 4D JAX PDE + Paired 30k RA-L Closure)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pytest

from scripts.colab.run_l4_jax_cbvf import build_colab_cli_plan

REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / "artifacts" / "v3.1.0" / "jax_4d_cbvf_l4_manifest.json"
PAIRED_STATS_PATH = REPO / "artifacts" / "v3.1.0" / "paired_30k_ral_stats.json"
RAL_SUMMARY_PATH = REPO / "docs" / "paper_ral" / "generated" / "ral_stats_summary.json"
RAL_TABLES_PATH = REPO / "docs" / "paper_ral" / "generated" / "ral_tables.tex"
RAL_TEX_PATH = REPO / "docs" / "paper_ral" / "vertirta_ral_manuscript.tex"


def test_colab_l4_plan_enforces_gpu_cost_guard_and_session_stop() -> None:
    plan = build_colab_cli_plan(session="vertirta-jax-l4", gpu="L4")
    flat = [" ".join(cmd) for cmd in plan]
    assert any("--gpu L4" in s for s in flat)
    assert any("stop -s vertirta-jax-l4" in s for s in flat)
    assert any("/tmp/vertirta_v31_blobs/cbvf_4d_x8_240.vrta_lut" in s for s in flat)
    for forbidden_gpu in ("G4", "H100", "A100"):
        with pytest.raises(ValueError, match="cost lock"):
            build_colab_cli_plan(gpu=forbidden_gpu)


def test_l4_jax_4d_cbvf_manifest_is_authentic_sound_and_hash_verified() -> None:
    assert MANIFEST_PATH.is_file(), f"Missing L4 JAX manifest: {MANIFEST_PATH}"
    raw = MANIFEST_PATH.read_bytes()
    assert b"\r" not in raw, "CRLF detected in jax_4d_cbvf_l4_manifest.json"
    m = json.loads(raw.decode("utf-8"))

    # Verify self-consistent manifest SHA-256
    recorded_sha = m["manifest_sha256"]
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    computed_sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    assert recorded_sha == computed_sha

    assert m["is_quick_cpu_fixture"] is False
    assert m["fine_grid_shape"] == [81, 57, 81, 57]
    assert m["fine_grid_nodes_per_regime"] == 21316689
    assert m["export_grid_shape"] == [41, 29, 41, 29]
    assert "L4" in m["hardware"]["gpu_name"] or "T4" in m["hardware"]["gpu_name"]
    assert m["hardware"]["vram_total_mib"] < 60000
    assert m["hardware"]["jax_peak_vram_mib"] > 0.0

    lut_meta = m["artifacts"]["cbvf_4d_lut"]
    cel_meta = m["artifacts"]["cel_2d_lut"]
    assert lut_meta["size_bytes"] == 11313864
    assert cel_meta["size_bytes"] == 11230
    assert (lut_meta["flags"] & 1) == 0  # bit 0 = quick_cpu_fixture MUST be 0
    assert len(lut_meta["payload_sha256"]) == 64
    assert len(cel_meta["payload_sha256"]) == 64

    err = m["errata_locks"]
    assert err["soundness_passed"] is True
    assert err["erratum1_max_cfl_ratio"] <= 1.0001
    assert err["erratum2_smax_zero_m"] <= 1e-6
    assert 0.0 < err["erratum3_v_cap_2m_cross_mps"] < 16.5

    assert len(m["regimes"]) == 4
    for r in m["regimes"]:
        assert r["solve_wall_s"] > 0.0
        assert math.isfinite(r["eps_grid_m"]) and r["eps_grid_m"] > 0.0
        assert r["pde_vs_proxy_abs_diff_m"]["mean"] > 0.0


def test_paired_30k_ral_closure_retires_pending_iqm_and_keeps_double_anonymous() -> None:
    assert PAIRED_STATS_PATH.is_file()
    assert RAL_SUMMARY_PATH.is_file()
    assert RAL_TABLES_PATH.is_file()

    stats = json.loads(PAIRED_STATS_PATH.read_text(encoding="utf-8"))
    ral_sum = json.loads(RAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert stats["sha256"] == ral_sum["sha256"]

    assert stats["method"] == "paired_per_seed_exact"
    assert stats["iqm_status"] == "COMPUTED_FROM_PER_RUN_VECTORS"
    assert stats["runs_per_config"] == 10000
    assert stats["ab_total_simulations"] == 30000
    assert stats["csfl_n"] == 6500 and stats["cel_n"] == 1500

    by_pair = {p["name"]: p for p in stats["holm_bonferroni_pairs"]}
    assert by_pair["csfl:hj_cbvf__indi_wls_vs_hocbf__indi_wls"]["significant_holm"] is True
    assert by_pair["cel:hj_cbvf__indi_wls_vs_hj_cbvf__stock_ca"]["significant_holm"] is True

    for cfg in ("hj_cbvf__indi_wls", "hocbf__indi_wls", "hj_cbvf__stock_ca"):
        for field in ("min_cbvf_margin_m", "touchdown_vz_mps", "max_tilt_deg", "min_v_term_v"):
            rec = stats["iqm_metrics"][cfg][field]
            assert rec["n"] == 10000 and rec["n_resamples"] == 1000
            assert rec["ci_low"] <= rec["iqm"] <= rec["ci_high"]
        p_rec = stats["pool_iqm_metrics"][cfg]
        assert p_rec["csfl_min_cbvf_margin_m"]["n"] == 6500
        assert p_rec["csfl_max_tilt_deg"]["n"] == 6500
        assert p_rec["cel_touchdown_vz_mps"]["n"] == 1500

    assert stats["pool_iqm_metrics"]["hj_cbvf__indi_wls"]["cel_touchdown_vz_mps"]["iqm"] < 2.0
    assert stats["jax_4d_cbvf_l4"]["manifest_path"] == "artifacts/v3.1.0/jax_4d_cbvf_l4_manifest.json"
    assert len(stats["gust_failure_cliff_sweep"]) == 8
    assert len(stats["speed_gust_authority_cliff_sweep"]) == 8
    assert stats["speed_gust_authority_cliff_sweep"][0]["hj_cbvf__indi_wls_pass_pct"] == 100.0
    assert stats["speed_gust_authority_cliff_sweep"][-1]["hj_cbvf__indi_wls_pass_pct"] == 0.0
    assert stats["stratum_breakdown"]["hj_cbvf__indi_wls"]["S3"]["k"] == 261
    assert stats["stratum_breakdown"]["hj_cbvf__indi_wls"]["S6"]["k"] == 48
    assert stats["live_ci_verification"]["renode_scenarios_passed"] == 3
    assert stats["live_ci_verification"]["esbmc_properties_verified"] == 1905
    assert "physical_stm32h7_hitl" in stats["pending_evidence"]
    assert "per_run_rliable_iqm" not in stats["pending_evidence"]

    tex_tables = RAL_TABLES_PATH.read_text(encoding="utf-8")
    assert "Pending (Per-Run Export)" not in tex_tables
    assert "Pending (Physical STM32H7 HITL)" in tex_tables
    assert "12.50" in tex_tables and "1{,}905" in tex_tables and "28{,}886" in tex_tables

    manuscript = RAL_TEX_PATH.read_text(encoding="utf-8")
    assert "Anonymous Authors" in manuscript
    for leak in ("github.com/", "github.io", "mailto:", "TODO", "[TBD]", "Pending (Per-Run Export)", "Qwen3", "PIDNet", "VertiBench"):
        assert leak not in manuscript, f"Leak/stale pending token in manuscript: {leak}"

    # Ensure zero multi-MB binary blobs were committed under artifacts/v3.1.0/.
    for p in (REPO / "artifacts" / "v3.1.0").rglob("*"):
        if p.is_file():
            assert p.suffix not in (".vrta_lut", ".vrta_cel2d", ".onnx", ".gguf", ".safetensors", ".pt")
            assert p.stat().st_size < 256_000, f"Oversized file under artifacts/v3.1.0/: {p}"
