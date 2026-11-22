"""Tests for VertiRTA's 4 GPU-Right-Sized Colab Notebooks & VertiBench Audit CLI.

Validates:
  1. Valid nbformat v4.5 JSON, pure LF line endings, and clean UTF-8 (no BOM) across all 4 notebooks.
  2. `ast.parse()` passes on EVERY Python code cell (after stripping Colab `%`/`!` magics).
  3. GPU-tier right-sizing metadata (`L4` primary, no hard `G4`-only lock, explicit cost guards).
  4. End-to-end `benchmarks/vertibench_tr_en/audit_cli.py` tests in `tmp_path`:
     - Deterministic `1,800`-row `400 V / 22 kWh` `build-train` generation & reproducibility.
     - Live `< 0.15` Jaccard + real Microsoft `z3-solver` SAT validation on `calibration.jsonl` (200)
       and `test.jsonl` (560) in both partial and full exact-count modes.
     - Rejection of contaminated utterances (>= 0.15 Jaccard), seed collisions, legacy 44.2 V 12S
       telemetry, malformed GBNF DSL, Z3-UNSAT plans, and non-human provenance.
     - Interactive `author` loop live rejection + recovery and `check-utterance` CLI.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vertibench_tr_en import audit_cli
from benchmarks.vertibench_tr_en.validate_vertibench import (
    EXPECTED_SPLIT_COUNTS,
    MAX_JACCARD_THRESHOLD,
    _load_jsonl,
)

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

EXPECTED_NOTEBOOKS: tuple[str, ...] = (
    "01_colab_jax_4d_cbvf_luts.ipynb",
    "02_colab_istanbul_4k_cinema.ipynb",
    "03_colab_copilot_qwen3_z3_grpo.ipynb",
    "04_colab_slz_pidnet_distill_and_egl60.ipynb",
)


def _strip_colab_magics(source_text: str) -> str:
    """Strip `%`/`%%` line/cell magics and `!` shell escapes before `ast.parse`."""
    cleaned_lines: list[str] = []
    lines = source_text.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return ""
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            cleaned_lines.append("")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _load_notebook(filename: str) -> tuple[Path, bytes, dict[str, Any]]:
    path = NOTEBOOKS_DIR / filename
    assert path.is_file(), f"Missing notebook: {path}"
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM forbidden in {filename}"
    assert b"\r" not in raw, f"CRLF forbidden in {filename}; expected pure LF"
    data = json.loads(raw.decode("utf-8"))
    return path, raw, data


def _all_notebook_text(nb: dict[str, Any]) -> str:
    chunks: list[str] = []
    for cell in nb["cells"]:
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_nbformat_v4_schema_and_ast_parse(filename: str) -> None:
    _, _, nb = _load_notebook(filename)

    assert nb.get("nbformat") == 4
    assert isinstance(nb.get("nbformat_minor"), int) and nb["nbformat_minor"] >= 4
    assert isinstance(nb.get("cells"), list) and len(nb["cells"]) >= 6

    seen_ids: set[str] = set()
    code_cell_count = 0
    for idx, cell in enumerate(nb["cells"]):
        ctype = cell.get("cell_type")
        assert ctype in ("markdown", "code"), f"{filename} cell {idx}: invalid type {ctype!r}"
        cid = cell.get("id")
        assert isinstance(cid, str) and cid, f"{filename} cell {idx}: missing cell id"
        assert cid not in seen_ids, f"{filename}: duplicate cell id {cid!r}"
        seen_ids.add(cid)
        assert isinstance(cell.get("metadata"), dict)
        assert isinstance(cell.get("source"), (list, str))

        if ctype == "code":
            code_cell_count += 1
            assert "execution_count" in cell
            assert isinstance(cell.get("outputs"), list)
            src_text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            py_src = _strip_colab_magics(src_text)
            try:
                ast.parse(py_src, filename=f"{filename}::{cid}")
            except SyntaxError as exc:
                pytest.fail(f"SyntaxError in {filename} cell {cid!r}: {exc}")

    assert code_cell_count >= 5, f"{filename} should have at least 5 executable code cells"


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_gpu_right_sizing_metadata_no_g4_lock(filename: str) -> None:
    _, _, nb = _load_notebook(filename)
    meta = nb.get("metadata", {})
    assert meta.get("accelerator") == "GPU"

    colab_meta = meta.get("colab", {})
    assert colab_meta.get("gpuType") == "L4", (
        f"{filename}: expected L4 primary colab.gpuType, got {colab_meta.get('gpuType')!r}"
    )
    assert "BLACKWELL" not in str(colab_meta.get("gpuType", "")).upper()
    assert "G4" not in str(colab_meta.get("gpuType", "")).upper()

    policy = meta.get("vertirta_gpu_policy", {})
    assert policy.get("primary_gpu") == "L4"
    assert policy.get("primary_vram_gb") == 24
    assert policy.get("g4_96gb_required") is False

    full_text = _all_notebook_text(nb)
    assert "COST WARNING" in full_text, f"{filename} missing G4 cost-discipline warning guard"
    forbidden_tokens = ("C:" + "\\Users\\", "/Users" + "/", "Obsidian" + "Vault")
    for forbidden in forbidden_tokens:
        assert forbidden not in full_text, f"{filename} leaked personal path {forbidden!r}"


def test_notebooks_readme_and_domain_contracts() -> None:
    readme = NOTEBOOKS_DIR / "README.md"
    assert readme.is_file()
    raw = readme.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    readme_text = raw.decode("utf-8")
    for token in ("T4 (16 GB)", "L4 (24 GB)", "A100 (40/80 GB)", "G4 (96 GB)", "NOT Needed"):
        assert token in readme_text, f"notebooks/README.md missing expected tier info {token!r}"

    # NB01: JAX 4D CBVF + 2D CEL + T4 fallback + Drive SHA-256
    _, _, nb1 = _load_notebook("01_colab_jax_4d_cbvf_luts.ipynb")
    t1 = _all_notebook_text(nb1)
    assert nb1["metadata"]["vertirta_gpu_policy"]["fallback_gpu"] == "T4"
    for needle in ("81x57x81x57", "solve_cbvf_luts", "solve_cel_2d_lut", "verify_lut_soundness.py", "sha256"):
        assert needle in t1

    # NB02: Blender 4.5/5.2 LTS Cycles OptiX + < 18 GB preflight + 150-frame slices + hevc_nvenc p5
    _, _, nb2 = _load_notebook("02_colab_istanbul_4k_cinema.ipynb")
    t2 = _all_notebook_text(nb2)
    for needle in (
        "blender_env_check",
        "assert_vram_within_budget",
        "max_gb=18.0",
        "slice_size=150",
        "hevc_nvenc",
        'preset="p5"',
    ):
        assert needle in t2
    assert 'preset="slow"' not in t2

    # NB03: Qwen3-4B on L4, optional A100 toggle ONLY for Qwen3-8B, G4 warning, GGUF + 4-model eval
    _, _, nb3 = _load_notebook("03_colab_copilot_qwen3_z3_grpo.ipynb")
    t3 = _all_notebook_text(nb3)
    assert nb3["metadata"]["vertirta_gpu_policy"]["primary_model"] == "Qwen/Qwen3-4B"
    assert nb3["metadata"]["vertirta_gpu_policy"]["optional_gpu"] == "A100"
    assert nb3["metadata"]["vertirta_gpu_policy"]["optional_model"] == "Qwen/Qwen3-8B"
    for needle in (
        "ENABLE_A100_8B_TOGGLE = False",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "CohereForAI/aya-expanse-8b",
        "train_sft_grpo",
        "export_gguf.sh",
        "calibrate_conformal",
        "eval_vertibench",
    ):
        assert needle in t3

    # NB04: EGL setup (llvmpipe absent), 8K gen, PIDNet-S adaptive bs=16/32, Opset-17 ONNX, 60-seed EGL manifest
    _, _, nb4 = _load_notebook("04_colab_slz_pidnet_distill_and_egl60.ipynb")
    t4 = _all_notebook_text(nb4)
    assert nb4["metadata"]["vertirta_gpu_policy"]["default_batch_size_l4"] == 16
    assert nb4["metadata"]["vertirta_gpu_policy"]["scale_up_batch_size_ge_40gb"] == 32
    for needle in (
        "colab_g4_gz_egl_setup.sh",
        "llvmpipe",
        "generate_istanbul_8k",
        "DINOv2-ViT-L/14",
        "Depth-Anything-V2-Small",
        "BATCH_SIZE = 32 if vram_gb >= 40.0 else 16",
        "export_onnx_trt",
        "eval_zeroshot_real",
        "active_vision_ablation_runner",
        "results/dr7_eval_manifest.json",
    ):
        assert needle in t4


# ---------------------------------------------------------------------------
# VertiBench audit_cli.py end-to-end tests in tmp_path
# ---------------------------------------------------------------------------
def _make_novel_human_row(
    split_name: str,
    bucket: str,
    idx: int,
    seed: int,
) -> dict[str, Any]:
    """Synthesize a clean, unique-vocab, Z3-SAT human_native_tr_v1 row for unit tests."""
    snap = audit_cli.generate_scaffold_snapshot(bucket, seed)
    top_pad = snap["ranked_slzs"][0]["id"]
    top_eta = min(600, max(35, int(round(snap["ranked_slzs"][0]["eta_s"] + 15.0))))
    alt_pad = snap["ranked_slzs"][1]["id"]

    # Unique alphanumeric/Turkish-free vocabulary token ensures guaranteed < 0.05 Jaccard vs train
    code_word = f"kvz{split_name[0]}{idx:04d}"
    if bucket == "NOMINAL":
        utt = f"Kaptan brifingi {code_word}: guverte temiz, alcak yaklasma onaylandi."
        dsl = f"ACT=DIVERT_SLZ;TGT={top_pad};ETA={top_eta};AGL=40,115;SPD=17\nLNG=TR;MOD=BRIEF"
    elif bucket == "FAULT":
        utt = f"Ariza kodu {code_word}: yedek batarya devrede, derhal guvenli inis yap."
        dsl = f"ACT=DIVERT_SLZ;TGT={top_pad};ETA={top_eta};AGL=40,115;SPD=15\nLNG=TR;MOD=BRIEF"
    elif bucket == "ADVERSARIAL":
        utt = f"Yolcu israri {code_word}: yasak bolgeden kisa yol gecisi talep ediliyor."
        dsl = f"ACT=REJECT_UNSAFE;VIO=LTFJ_CTR;FB={top_pad}\nLNG=TR;MOD=REJECT"
    else:
        utt = f"Telsiz paraziti {code_word}: konum secimi belirsiz, ikili teyit bekle."
        dsl = f"ACT=REQUEST_CLARIFICATION;A={top_pad};B={alt_pad};RSN=AMBIG\nLNG=TR;MOD=CLARIFY"

    return {
        "id": f"vb_{split_name}_{bucket.lower()}_{idx:04d}",
        "split": split_name,
        "bucket": bucket,
        "lang": "TR",
        "scenario_seed": seed,
        "origin_provenance": audit_cli.HUMAN_PROVENANCE,
        "rta_snapshot": snap,
        "user_utterance": utt,
        "asr_corruption_tags": ["human_authored"],
        "gold_intent": audit_cli._derive_gold_intent(bucket, dsl, snap),
        "canonical_dsl": dsl,
    }


def _write_human_split_subset(
    path: Path, split_name: str, base_seed: int, per_bucket: int = 3
) -> None:
    rows: list[dict[str, Any]] = []
    seed = base_seed
    for bucket in EXPECTED_SPLIT_COUNTS[split_name]:
        for i in range(per_bucket):
            rows.append(_make_novel_human_row(split_name, bucket, i, seed))
            seed += 1
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def built_train_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build deterministic GBNF+Z3-verified train.jsonl once per module via audit_cli."""
    data_dir = tmp_path_factory.mktemp("vertibench_train")
    rc = audit_cli.main(
        ["build-train", "--out-dir", str(data_dir), "--seed", "2026", "--per-bucket", "6"]
    )
    assert rc == 0
    return data_dir


def test_audit_cli_build_train_deterministic_and_400v_lock(
    built_train_dir: Path, tmp_path: Path
) -> None:
    # Default production quota must lock exact 1,800 train / 200 cal / 560 test counts
    assert sum(EXPECTED_SPLIT_COUNTS["train"].values()) == 1800
    assert sum(EXPECTED_SPLIT_COUNTS["calibration"].values()) == 200
    assert sum(EXPECTED_SPLIT_COUNTS["test"].values()) == 560

    train_a = built_train_dir / "train.jsonl"
    rows = _load_jsonl(train_a)
    assert len(rows) == 24

    # Verify exact 400 V / 22 kWh HV pack bounds across every single generated row
    for row in rows:
        snap = row["rta_snapshot"]
        assert snap["pack_nominal_volts"] == 400.0
        assert snap["pack_energy_kwh"] == 22.0
        assert 320.0 <= snap["ownship"]["v_term_volts"] <= 420.0

    # Determinism check: identical seed produces byte-identical train.jsonl
    rc = audit_cli.main(
        ["build-train", "--out-dir", str(tmp_path), "--seed", "2026", "--per-bucket", "6"]
    )
    assert rc == 0
    assert (tmp_path / "train.jsonl").read_bytes() == train_a.read_bytes()


def test_audit_cli_calibration_and_test_batch_audit_and_quota_gate(
    built_train_dir: Path, tmp_path: Path
) -> None:
    (tmp_path / "train.jsonl").write_bytes((built_train_dir / "train.jsonl").read_bytes())
    _write_human_split_subset(
        tmp_path / "calibration.jsonl", "calibration", base_seed=510_000, per_bucket=3
    )
    _write_human_split_subset(
        tmp_path / "test.jsonl", "test", base_seed=710_000, per_bucket=3
    )

    # 1. Partial mode passes with 100% real Microsoft Z3-SAT and <0.15 Jaccard + exact quota progress
    partial_json = tmp_path / "audit_partial.json"
    rc_partial = audit_cli.main(
        [
            "audit",
            "--data-dir",
            str(tmp_path),
            "--split",
            "all",
            "--allow-partial",
            "--out-json",
            str(partial_json),
        ]
    )
    assert rc_partial == 0
    report = json.loads(partial_json.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["splits"]["calibration"]["expected_total_rows"] == 200
    assert report["splits"]["calibration"]["total_rows"] == 12
    assert report["splits"]["calibration"]["z3_sat_rate_pct"] == 100.0
    assert report["splits"]["calibration"]["max_char_5gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert report["splits"]["test"]["expected_total_rows"] == 560
    assert report["splits"]["test"]["total_rows"] == 12
    assert report["splits"]["test"]["z3_sat_rate_pct"] == 100.0
    assert report["splits"]["test"]["max_token_3gram_jaccard"] < MAX_JACCARD_THRESHOLD

    # 2. Strict full-quota mode (no --allow-partial) MUST fail when splits are below 1800/200/560
    strict_json = tmp_path / "audit_strict.json"
    rc_strict = audit_cli.main(
        ["audit", "--data-dir", str(tmp_path), "--split", "all", "--out-json", str(strict_json)]
    )
    assert rc_strict == 1
    strict_rep = json.loads(strict_json.read_text(encoding="utf-8"))
    assert strict_rep["passed"] is False
    assert any("expected 1800 rows" in e for e in strict_rep["errors"])
    assert any("expected 200" in e for e in strict_rep["errors"])
    assert any("expected 560" in e for e in strict_rep["errors"])


def test_audit_cli_rejects_contamination_seed_leak_44v_and_z3_unsat(
    built_train_dir: Path, tmp_path: Path
) -> None:
    train_rows = _load_jsonl(built_train_dir / "train.jsonl")
    train_index = audit_cli.TrainCorpusIndex.from_rows(train_rows)

    # 1. Clean baseline row passes
    clean = _make_novel_human_row("test", "NOMINAL", 0, seed=800_001)
    ok_res = audit_cli.audit_candidate_row(clean, train_index, split_name="test")
    assert ok_res.passed is True
    assert ok_res.z3_sat is True
    assert ok_res.char_5gram_jaccard < MAX_JACCARD_THRESHOLD

    # 2. High Jaccard contamination (>= 0.15 copied train utterance) -> REJECT
    contaminated = dict(clean, user_utterance=train_rows[0]["user_utterance"])
    c_res = audit_cli.audit_candidate_row(contaminated, train_index, split_name="test")
    assert c_res.passed is False
    assert c_res.char_5gram_jaccard == 1.0
    assert any("Jaccard" in e for e in c_res.errors)

    # 3. Scenario seed collision with train -> REJECT
    seed_leak = dict(clean, scenario_seed=train_rows[0]["scenario_seed"])
    s_res = audit_cli.audit_candidate_row(seed_leak, train_index, split_name="test")
    assert s_res.passed is False
    assert s_res.seed_collision is True

    # 4. Legacy 44.2 V 12S battery telemetry -> REJECT
    bad_hv_snap = json.loads(json.dumps(clean["rta_snapshot"]))
    bad_hv_snap["ownship"]["v_term_volts"] = 44.2
    bad_hv = dict(clean, rta_snapshot=bad_hv_snap)
    hv_res = audit_cli.audit_candidate_row(bad_hv, train_index, split_name="test")
    assert hv_res.passed is False
    assert hv_res.hv_ok is False

    # 5. Real Z3 QF_LRA UNSAT plan (insufficient SoC margin to target SLZ) -> REJECT
    low_soc_snap = json.loads(json.dumps(clean["rta_snapshot"]))
    low_soc_snap["ownship"]["soc_pct"] = 14.0
    for pad in low_soc_snap["ranked_slzs"]:
        pad["soc_margin_pct"] = -9.5
    z3_unsat_row = dict(clean, rta_snapshot=low_soc_snap)
    z_res = audit_cli.audit_candidate_row(z3_unsat_row, train_index, split_name="test")
    assert z_res.passed is False
    assert z_res.z3_sat is False
    assert len(z_res.z3_unsat_core) >= 1

    # 6. Wrong provenance (synthetic instead of human_native_tr_v1) -> REJECT
    bad_prov = dict(clean, origin_provenance="programmatic_z3_filtered_v1")
    p_res = audit_cli.audit_candidate_row(bad_prov, train_index, split_name="test")
    assert p_res.passed is False
    assert p_res.provenance_ok is False

    # 7. check-utterance CLI subcommand: exit 0 for clean, exit 1 for contaminated
    train_path = built_train_dir / "train.jsonl"
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(clean["rta_snapshot"]), encoding="utf-8")
    assert (
        audit_cli.main(
            [
                "check-utterance",
                "--train-jsonl",
                str(train_path),
                "--utterance",
                clean["user_utterance"],
                "--dsl",
                clean["canonical_dsl"],
                "--snapshot-json",
                str(snap_path),
            ]
        )
        == 0
    )
    assert (
        audit_cli.main(
            [
                "check-utterance",
                "--train-jsonl",
                str(train_path),
                "--utterance",
                train_rows[0]["user_utterance"],
            ]
        )
        == 1
    )


def test_audit_cli_interactive_author_rejects_leak_then_accepts_clean(
    built_train_dir: Path, tmp_path: Path
) -> None:
    (tmp_path / "train.jsonl").write_bytes((built_train_dir / "train.jsonl").read_bytes())
    train_rows = _load_jsonl(tmp_path / "train.jsonl")

    # Pre-inspect the deterministic scaffold snapshot that interactive_author_loop will build
    seed = audit_cli.EVAL_SEED_BASE["calibration"] + 1
    snap = audit_cli.generate_scaffold_snapshot("FAULT", seed)
    top_pad = snap["ranked_slzs"][0]["id"]
    top_eta = min(600, max(35, int(round(snap["ranked_slzs"][0]["eta_s"] + 15.0))))

    answers = iter(
        [
            # Attempt 1: verbatim leaked train utterance -> must be rejected immediately
            train_rows[10]["user_utterance"],
            # Attempt 2: clean novel Turkish utterance + valid Z3-SAT compact DSL -> accepted
            "Kokpit acil durum bildirimi xq91: ruzgar keskin, en yakin kiyi noktaya kontrollu inis.",
            "TR",
            f"ACT=DIVERT_SLZ;TGT={top_pad};ETA={top_eta};AGL=40,115;SPD=15",
            "LNG=TR;MOD=BRIEF",
        ]
    )
    logs: list[str] = []

    accepted = audit_cli.interactive_author_loop(
        tmp_path,
        split_name="calibration",
        bucket="FAULT",
        count=1,
        input_fn=lambda _prompt: next(answers),
        output_fn=logs.append,
    )
    assert len(accepted) == 1
    assert any("REJECTED (contamination >= 0.15" in line for line in logs)
    assert any("Z3_SAT=True" in line for line in logs)

    # Partial batch audit must pass on the newly written human calibration.jsonl
    rc = audit_cli.main(
        ["audit", "--data-dir", str(tmp_path), "--split", "calibration", "--allow-partial"]
    )
    assert rc == 0
