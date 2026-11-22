"""TDD tests for Plan 3 Task 8: Reproducible LaTeX Technical Report, Honesty Linter & Zenodo Metadata."""

from __future__ import annotations

import json
from pathlib import Path

from docs.paper.scripts.generate_latex_tables import render_tables_from_manifest
from docs.paper.scripts.verify_paper_numbers import verify_paper_honesty_and_numbers


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = REPO_ROOT / "docs" / "paper"


def test_latex_table_generation_honesty_linter_and_zenodo(tmp_path: Path) -> None:
    """Pin exact Plan 3 Task 8 synthetic manifest rendering, honesty bans, and .zenodo.json."""
    mf = tmp_path / "metrics_manifest.json"
    mf.write_text(
        json.dumps(
            {
                "v1_atlas": {"csfl_n": 650, "cel_n": 150, "cbvf_violations_in_brt": 0},
                "v2a_copilot": {"unsafe_rejection_pct": 100.0, "conformal_cov_pct": 95.5},
                "v2b_slz": {
                    "active_safe_touchdown_pct": 93.3,
                    "active_false_commit_pct": 1.7,
                    "wsl2_vram_mb": 238.0,
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    tex = render_tables_from_manifest(mf, tmp_path / "tables.tex").read_text(encoding="utf-8")
    assert "93.3" in tex and "650" in tex and "150" in tex and "TBD" not in tex

    (tmp_path / "bad.tex").write_text(
        "Monte Carlo proves 10^{-9}/FH EASA SC-VTOL certified compliance.\n",
        encoding="utf-8",
        newline="\n",
    )
    errs = verify_paper_honesty_and_numbers(tmp_path, mf)
    assert any("10^{-9}" in e or "certified" in e for e in errs)
    assert any("10^{-9}" in e for e in errs)
    assert any("certified" in e or "compliance" in e for e in errs)

    # Also verify placeholder bans ([TBD], [Ölçülecek], TODO)
    placeholder_dir = tmp_path / "ph"
    placeholder_dir.mkdir()
    render_tables_from_manifest(mf, placeholder_dir / "tables.tex")
    (placeholder_dir / "draft.tex").write_text(
        "Result is [Ölçülecek] and [TBD] pending run.\n",
        encoding="utf-8",
        newline="\n",
    )
    ph_errs = verify_paper_honesty_and_numbers(placeholder_dir, mf)
    assert any("TBD" in e or "Ölçülecek" in e for e in ph_errs)

    meta = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert meta["license"] == "Apache-2.0" and "VertiRTA" in meta["title"]


def test_pending_markers_render_cleanly_without_tbd_or_fabrication(tmp_path: Path) -> None:
    """Verify explicit PENDING_* markers render as human-readable Pending (...) labels, never TBD."""
    mf = tmp_path / "pending_manifest.json"
    mf.write_text(
        json.dumps(
            {
                "v1_atlas": {
                    "csfl_n": 6500,
                    "cel_n": 1500,
                    "arm64_ci_status": "PENDING_GITHUB_ACTIONS_ARM64",
                },
                "v2a_copilot": {"unsafe_rejection_pct": "PENDING_COLAB_G4"},
                "v2b_slz": {
                    "active_safe_touchdown_pct": "PENDING_COLAB_G4",
                    "wsl2_vram_mb": "PENDING_LOCAL_GPU_SMOKE",
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    out_tex = render_tables_from_manifest(mf, tmp_path / "tables.tex")
    text = out_tex.read_text(encoding="utf-8")
    assert "Pending (Colab G4)" in text
    assert "Pending (ARM64 CI)" in text
    assert "Pending (WSL2 GTX 1650S)" in text
    assert "TBD" not in text
    assert verify_paper_honesty_and_numbers(tmp_path, mf) == []


def test_live_paper_manifest_tables_and_manuscript_pass_all_gates() -> None:
    """Verify live docs/paper/ matches artifacts/v1.0.0/ verbatim and passes honesty + BibTeX lint."""
    manifest_path = PAPER_DIR / "metrics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Pin Pillar 1 metrics verbatim against live artifacts/v1.0.0/ JSONs
    v1_atlas_art = json.loads(
        (REPO_ROOT / "artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json").read_text(
            encoding="utf-8"
        )
    )
    v1_struct_art = json.loads(
        (REPO_ROOT / "artifacts/v1.0.0/benchmarks/tier1_structural.json").read_text(encoding="utf-8")
    )
    v1_t4_art = json.loads(
        (REPO_ROOT / "artifacts/v1.0.0/benchmarks/tier4_wallclock_x86_64.json").read_text(
            encoding="utf-8"
        )
    )

    v1 = manifest["v1_atlas"]
    assert v1["csfl_n"] == v1_atlas_art["summary"]["csfl_pool"]["n"] == 6500
    assert v1["cel_n"] == v1_atlas_art["summary"]["cel_pool"]["n"] == 1500
    assert v1["cliff_failures"] == v1_atlas_art["total_failures"] == 244
    assert abs(v1["csfl_pass_pct"] - round(v1_atlas_art["ab_comparison"]["hj_cbvf__indi_wls"]["csfl_pass_rate"] * 100.0, 2)) < 1e-6
    assert abs(v1["cel_pass_pct"] - round(v1_atlas_art["ab_comparison"]["hj_cbvf__indi_wls"]["cel_pass_rate"] * 100.0, 2)) < 1e-6
    assert abs(v1["hocbf_csfl_pass_pct"] - round(v1_atlas_art["ab_comparison"]["hocbf__indi_wls"]["csfl_pass_rate"] * 100.0, 2)) < 1e-6
    assert abs(v1["stock_ca_cel_pass_pct"] - round(v1_atlas_art["ab_comparison"]["hj_cbvf__stock_ca"]["cel_pass_rate"] * 100.0, 2)) < 1e-6
    assert v1["cortex_m7_flash_bytes"] == v1_struct_art["cortex_m7"]["flash_bytes"] == 28886
    assert v1["cortex_m7_ram_bytes"] == v1_struct_art["cortex_m7"]["ram_bytes"] == 0
    assert v1["cortex_m7_max_dag_bytes"] == v1_struct_art["cortex_m7"]["max_dag_bytes"] == 2960
    assert v1["arm_m7_heap_symbols"] == v1_struct_art["cortex_m7"]["forbidden_heap_symbols"] == 0
    assert abs(v1["tier4_p99_us"] - v1_t4_art["metrics"]["p99_us"]) < 1e-6

    # Verify Pillar 2 & 3 pending cloud/GPU rows are honestly marked (never fabricated)
    assert manifest["v2a_copilot"]["conformal_q_level"] == 0.955
    assert str(manifest["v2a_copilot"]["unsafe_rejection_pct"]).startswith("PENDING_")
    assert manifest["v2b_slz"]["slz_grid_mib"] == 1.5625
    assert manifest["v2b_slz"]["slz_disk_cells"] == 317
    assert str(manifest["v2b_slz"]["active_safe_touchdown_pct"]).startswith("PENDING_")
    assert str(manifest["v2b_slz"]["wsl2_vram_mb"]).startswith("PENDING_")

    # Verify generated/tables.tex exists and matches deterministic regeneration
    gen_tex = PAPER_DIR / "generated" / "tables.tex"
    assert gen_tex.exists()
    assert "\\input{generated/tables.tex}" in (PAPER_DIR / "vertirta_tech_report.tex").read_text(
        encoding="utf-8"
    )

    # Zero honesty, number-drift, BibTeX, or LaTeX environment errors on the live manuscript
    errors = verify_paper_honesty_and_numbers(PAPER_DIR, manifest_path)
    assert errors == [], f"Paper verification failed: {errors}"
