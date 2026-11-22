"""Interactive & Batch Audit CLI for VertiBench-TR/EN.

Capabilities:
  1. `build-train`: Deterministically synthesizes the 1,800-row programmatic
     `train.jsonl` split (locked 400 V / 22 kWh HV pack, compact <=30-token
     ASCII GBNF DSL, 100% real Microsoft z3-solver SAT-filtered).
  2. `audit`: Batch-validates human-authored `calibration.jsonl` (200 rows) and
     `test.jsonl` (560 rows) against `train.jsonl` with live `< 0.15` char
     5-gram / token 3-gram Jaccard anti-contamination checks, 0 scenario-seed
     overlap, 400 V / 22 kWh HV telemetry bounds, compact GBNF parsing, and
     real Microsoft z3-solver QF_LRA verification.
  3. `check-utterance`: One-shot live Jaccard + optional GBNF/Z3 probe for a
     single candidate utterance while authoring.
  4. `author`: Interactive CLI authoring loop that proposes non-colliding
     400 V / 22 kWh Kartal scenarios and gives immediate `< 0.15` Jaccard +
     Z3-SAT feedback before appending `human_native_tr_v1` rows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.vertibench_tr_en.build_train_split import (
    FAULT_MODES,
    _build_snapshot,
    build_train_split,
)
from benchmarks.vertibench_tr_en.validate_vertibench import (
    EXPECTED_SPLIT_COUNTS,
    MAX_JACCARD_THRESHOLD,
    VALID_LANG_CODES,
    _char_ngrams,
    _jaccard,
    _load_jsonl,
    _token_ngrams,
    parse_compact_dsl_py,
    validate_hv_telemetry,
    validate_vertibench,
)
from ml.copilot.grpo_z3_rewards import verify_z3_static_corridor

HUMAN_PROVENANCE = "human_native_tr_v1"
EVAL_SEED_BASE: dict[str, int] = {
    "calibration": 500_000,
    "test": 700_000,
}


@dataclass(frozen=True)
class TrainCorpusIndex:
    """Precomputed n-gram & seed index over `train.jsonl` for fast live auditing."""

    n_rows: int
    seeds: frozenset[int]
    char5_sets: tuple[frozenset[str], ...]
    tok3_sets: tuple[frozenset[tuple[str, ...]], ...]

    @classmethod
    def from_rows(cls, rows: Sequence[dict[str, Any]]) -> "TrainCorpusIndex":
        seeds = frozenset(
            int(r["scenario_seed"]) for r in rows if "scenario_seed" in r and r["scenario_seed"] is not None
        )
        char5 = tuple(frozenset(_char_ngrams(str(r.get("user_utterance", "")), 5)) for r in rows)
        tok3 = tuple(frozenset(_token_ngrams(str(r.get("user_utterance", "")), 3)) for r in rows)
        return cls(n_rows=len(rows), seeds=seeds, char5_sets=char5, tok3_sets=tok3)

    @classmethod
    def from_jsonl(cls, train_path: Path) -> "TrainCorpusIndex":
        return cls.from_rows(_load_jsonl(train_path))

    def max_jaccards(self, utterance: str) -> tuple[float, float]:
        """Return (max_char_5gram_jaccard, max_token_3gram_jaccard) against train."""
        c5_target = _char_ngrams(utterance, 5)
        t3_target = _token_ngrams(utterance, 3)
        max_c5 = 0.0
        for ref_c5 in self.char5_sets:
            score = _jaccard(c5_target, ref_c5)
            if score > max_c5:
                max_c5 = score
        max_t3 = 0.0
        for ref_t3 in self.tok3_sets:
            score = _jaccard(t3_target, ref_t3)
            if score > max_t3:
                max_t3 = score
        return round(max_c5, 4), round(max_t3, 4)


@dataclass
class RowAuditResult:
    """Per-row deterministic audit feedback (HV telemetry + GBNF + Z3 + Jaccard)."""

    passed: bool
    row_id: str
    split: str
    bucket: str
    hv_ok: bool
    dsl_ok: bool
    z3_sat: bool
    z3_fallback_sat: bool
    z3_unsat_core: list[str]
    provenance_ok: bool
    seed_collision: bool
    char_5gram_jaccard: float
    token_3gram_jaccard: float
    errors: list[str] = field(default_factory=list)


@dataclass
class SplitAuditSummary:
    """Aggregate audit metrics for one human-authored split (`calibration` or `test`)."""

    passed: bool
    split: str
    path: str
    total_rows: int
    expected_total_rows: int
    passed_rows: int
    failed_rows: int
    bucket_counts: dict[str, int]
    expected_bucket_counts: dict[str, int]
    bucket_remaining: dict[str, int]
    z3_sat_rate_pct: float
    max_char_5gram_jaccard: float
    max_token_3gram_jaccard: float
    jaccard_threshold: float
    allow_partial: bool
    errors: list[str] = field(default_factory=list)
    failed_row_details: list[dict[str, Any]] = field(default_factory=list)


def audit_candidate_row(
    row: dict[str, Any],
    train_index: TrainCorpusIndex,
    *,
    split_name: str = "test",
    seen_seeds: set[int] | None = None,
) -> RowAuditResult:
    """Audit one candidate VertiBench row with live <0.15 Jaccard + real Z3-SAT checks."""
    errors: list[str] = []
    row_id = str(row.get("id", "<missing-id>"))
    bucket = str(row.get("bucket", ""))
    expected_buckets = EXPECTED_SPLIT_COUNTS.get(split_name, EXPECTED_SPLIT_COUNTS["test"])

    if bucket not in expected_buckets:
        errors.append(f"invalid bucket {bucket!r}")

    if split_name in ("calibration", "test"):
        provenance_ok = row.get("origin_provenance") == HUMAN_PROVENANCE
        if not provenance_ok:
            errors.append(
                f"expected origin_provenance={HUMAN_PROVENANCE!r}, got {row.get('origin_provenance')!r}"
            )
    else:
        provenance_ok = bool(row.get("origin_provenance"))

    snapshot = row.get("rta_snapshot", {})
    hv_ok = isinstance(snapshot, dict) and validate_hv_telemetry(snapshot)
    if not hv_ok:
        errors.append("invalid 400V/22kWh HV telemetry (v_term_volts must be in [320, 420] V)")

    dsl_text = str(row.get("canonical_dsl", ""))
    ast = parse_compact_dsl_py(dsl_text)
    dsl_ok = ast is not None
    if not dsl_ok:
        errors.append("canonical_dsl failed strict compact GBNF parser")

    z3_sat = False
    z3_fb_sat = False
    z3_core: list[str] = []
    if dsl_ok and hv_ok and ast is not None:
        z3_res = verify_z3_static_corridor(ast, snapshot)
        z3_sat = bool(z3_res.is_sat)
        z3_fb_sat = bool(z3_res.fallback_is_sat)
        z3_core = list(z3_res.unsat_labels)
        if not z3_sat:
            errors.append(f"Z3 QF_LRA UNSAT (core={z3_core})")
        if ast.action == "REJECT_UNSAFE" and not z3_fb_sat:
            errors.append(f"REJECT_UNSAFE fallback {ast.fallback!r} is not Z3-SAT")

    seed = row.get("scenario_seed")
    seed_collision = False
    if seed is None:
        errors.append("missing scenario_seed")
    else:
        seed_int = int(seed)
        if split_name in ("calibration", "test") and seed_int in train_index.seeds:
            seed_collision = True
            errors.append(f"scenario_seed {seed_int} collides with train.jsonl")
        if seen_seeds is not None and seed_int in seen_seeds:
            seed_collision = True
            errors.append(f"duplicate scenario_seed {seed_int} within split")

    utt = str(row.get("user_utterance", "")).strip()
    if not utt:
        errors.append("empty user_utterance")
        c5, t3 = 0.0, 0.0
    elif split_name in ("calibration", "test"):
        c5, t3 = train_index.max_jaccards(utt)
        if c5 >= MAX_JACCARD_THRESHOLD:
            errors.append(f"char 5-gram Jaccard {c5:.4f} >= {MAX_JACCARD_THRESHOLD}")
        if t3 >= MAX_JACCARD_THRESHOLD:
            errors.append(f"token 3-gram Jaccard {t3:.4f} >= {MAX_JACCARD_THRESHOLD}")
    else:
        c5, t3 = 0.0, 0.0

    return RowAuditResult(
        passed=len(errors) == 0,
        row_id=row_id,
        split=split_name,
        bucket=bucket,
        hv_ok=hv_ok,
        dsl_ok=dsl_ok,
        z3_sat=z3_sat,
        z3_fallback_sat=z3_fb_sat,
        z3_unsat_core=z3_core,
        provenance_ok=provenance_ok,
        seed_collision=seed_collision,
        char_5gram_jaccard=c5,
        token_3gram_jaccard=t3,
        errors=errors,
    )


def audit_split_file(
    split_path: Path,
    train_index: TrainCorpusIndex,
    *,
    split_name: str,
    allow_partial: bool = False,
) -> SplitAuditSummary:
    """Batch-audit `calibration.jsonl` (200) or `test.jsonl` (560) against `train.jsonl`."""
    if split_name not in ("calibration", "test"):
        raise ValueError(f"Expected split_name in ('calibration', 'test'), got {split_name!r}")

    expected_buckets = dict(EXPECTED_SPLIT_COUNTS[split_name])
    expected_total = sum(expected_buckets.values())
    bucket_counts = {k: 0 for k in expected_buckets}
    errors: list[str] = []
    failed_details: list[dict[str, Any]] = []

    if not split_path.exists():
        errors.append(f"Missing split file: {split_path}")
        return SplitAuditSummary(
            passed=False,
            split=split_name,
            path=str(split_path),
            total_rows=0,
            expected_total_rows=expected_total,
            passed_rows=0,
            failed_rows=0,
            bucket_counts=bucket_counts,
            expected_bucket_counts=expected_buckets,
            bucket_remaining=dict(expected_buckets),
            z3_sat_rate_pct=0.0,
            max_char_5gram_jaccard=0.0,
            max_token_3gram_jaccard=0.0,
            jaccard_threshold=MAX_JACCARD_THRESHOLD,
            allow_partial=allow_partial,
            errors=errors,
            failed_row_details=failed_details,
        )

    rows = _load_jsonl(split_path)
    seen_seeds: set[int] = set()
    seen_ids: set[str] = set()
    passed_rows = 0
    z3_sat_count = 0
    max_c5 = 0.0
    max_t3 = 0.0

    for row in rows:
        res = audit_candidate_row(row, train_index, split_name=split_name, seen_seeds=seen_seeds)
        if row.get("scenario_seed") is not None:
            seen_seeds.add(int(row["scenario_seed"]))
        if res.row_id in seen_ids:
            res.errors.append(f"duplicate row id {res.row_id!r}")
            res.passed = False
        seen_ids.add(res.row_id)

        if res.bucket in bucket_counts:
            bucket_counts[res.bucket] += 1
        if res.z3_sat:
            z3_sat_count += 1
        max_c5 = max(max_c5, res.char_5gram_jaccard)
        max_t3 = max(max_t3, res.token_3gram_jaccard)

        if res.passed:
            passed_rows += 1
        else:
            failed_details.append(asdict(res))
            for err in res.errors:
                errors.append(f"{split_name}/{res.row_id}: {err}")

    bucket_remaining = {
        b: max(0, expected_buckets[b] - bucket_counts[b]) for b in expected_buckets
    }
    if not allow_partial:
        for b, exp_n in expected_buckets.items():
            if bucket_counts[b] != exp_n:
                errors.append(
                    f"{split_name} bucket {b}: expected {exp_n} rows, got {bucket_counts[b]}"
                )
        if len(rows) != expected_total:
            errors.append(
                f"{split_name} total rows: expected {expected_total}, got {len(rows)}"
            )
    else:
        for b, exp_n in expected_buckets.items():
            if bucket_counts[b] > exp_n:
                errors.append(
                    f"{split_name} bucket {b}: exceeds quota {exp_n} (got {bucket_counts[b]})"
                )

    total = len(rows)
    z3_rate = round(100.0 * z3_sat_count / float(total), 2) if total > 0 else 0.0
    return SplitAuditSummary(
        passed=len(errors) == 0,
        split=split_name,
        path=str(split_path),
        total_rows=total,
        expected_total_rows=expected_total,
        passed_rows=passed_rows,
        failed_rows=total - passed_rows,
        bucket_counts=bucket_counts,
        expected_bucket_counts=expected_buckets,
        bucket_remaining=bucket_remaining,
        z3_sat_rate_pct=z3_rate,
        max_char_5gram_jaccard=round(max_c5, 4),
        max_token_3gram_jaccard=round(max_t3, 4),
        jaccard_threshold=MAX_JACCARD_THRESHOLD,
        allow_partial=allow_partial,
        errors=errors,
        failed_row_details=failed_details,
    )


def audit_dataset_dir(
    data_dir: Path,
    *,
    splits: Sequence[str] = ("calibration", "test"),
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Audit `train.jsonl` + requested human-authored splits in `data_dir`."""
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        return {
            "passed": False,
            "data_dir": str(data_dir),
            "errors": [f"Missing train.jsonl at {train_path}; run `audit_cli.py build-train` first."],
            "splits": {},
        }

    train_rows = _load_jsonl(train_path)
    train_index = TrainCorpusIndex.from_rows(train_rows)
    errors: list[str] = []
    expected_train = sum(EXPECTED_SPLIT_COUNTS["train"].values())
    if not allow_partial and len(train_rows) != expected_train:
        errors.append(f"train.jsonl expected {expected_train} rows, got {len(train_rows)}")

    split_reports: dict[str, dict[str, Any]] = {}
    cross_split_seeds: dict[str, set[int]] = {}
    for split_name in splits:
        p = data_dir / f"{split_name}.jsonl"
        rep = audit_split_file(p, train_index, split_name=split_name, allow_partial=allow_partial)
        split_reports[split_name] = asdict(rep)
        if not rep.passed:
            errors.extend(rep.errors)
        if p.exists():
            cross_split_seeds[split_name] = {
                int(r["scenario_seed"]) for r in _load_jsonl(p) if r.get("scenario_seed") is not None
            }

    if "calibration" in cross_split_seeds and "test" in cross_split_seeds:
        overlap = sorted(cross_split_seeds["calibration"] & cross_split_seeds["test"])
        if overlap:
            errors.append(f"scenario_seed overlap between calibration and test: {overlap[:10]}")

    validator_passed: bool | None = None
    if not allow_partial and set(splits) == {"calibration", "test"} and not errors:
        val_rep = validate_vertibench(data_dir, check_schema_only=True)
        validator_passed = val_rep.passed
        if not val_rep.passed:
            errors.extend(val_rep.errors)

    return {
        "passed": len(errors) == 0,
        "data_dir": str(data_dir),
        "train_rows": len(train_rows),
        "allow_partial": allow_partial,
        "validator_passed": validator_passed,
        "splits": split_reports,
        "errors": errors,
    }


def generate_scaffold_snapshot(bucket: str, seed: int) -> dict[str, Any]:
    """Build a valid locked 400 V / 22 kWh Kartal RTA snapshot for human authoring."""
    rng = random.Random(seed)
    own_e = rng.uniform(-3800.0, -450.0)
    own_n = rng.uniform(150.0, 2400.0)
    own_agl = rng.uniform(45.0, 110.0)
    wind_mps = rng.uniform(2.0, 9.0)

    if bucket == "NOMINAL":
        fault = "NOMINAL"
        soc_pct = rng.uniform(48.0, 82.0)
    elif bucket == "FAULT":
        fault = rng.choice(FAULT_MODES[1:])
        soc_pct = rng.uniform(34.0, 62.0)
    elif bucket == "ADVERSARIAL":
        fault = rng.choice(("SINGLE_MOTOR_LOSS", "OPPOSITE_DUAL_LOSS", "NOMINAL"))
        soc_pct = rng.uniform(36.0, 64.0)
    else:
        fault = rng.choice(("SINGLE_MOTOR_LOSS", "SENSOR_TMR_DEGRADED", "NOMINAL"))
        soc_pct = rng.uniform(42.0, 72.0)

    return _build_snapshot(
        rng,
        fault_mode=fault,
        soc_pct=soc_pct,
        own_e=own_e,
        own_n=own_n,
        own_agl=own_agl,
        wind_mps=wind_mps,
    )


def _derive_gold_intent(bucket: str, dsl_text: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    ast = parse_compact_dsl_py(dsl_text)
    sat_pads = [s["id"] for s in snapshot["ranked_slzs"] if float(s["soc_margin_pct"]) >= 0.0]
    if ast is None:
        return {}
    if ast.action == "REJECT_UNSAFE":
        vio_to_label = {
            "LTFJ_CTR": "c_ltfj_ctr",
            "LTBX_CTR": "c_ltbx_ctr",
            "DYN_NFZ": "c_dyn_nfz",
            "SOC_RESERVE": "c_soc_reserve",
            "SLZ_UNSAFE": "c_slz_unsafe",
            "MIN_AGL": "c_min_agl",
            "SHIELD_BYPASS": "c_shield_bypass",
        }
        return {
            "expected_mode": "REJECT_UNSAFE_OVERRIDE",
            "expected_action": "REJECT_UNSAFE",
            "valid_targets": sat_pads,
            "expected_unsat_labels": [vio_to_label.get(ast.violation or "", "c_slz_unsafe")],
            "requires_clarification": False,
            "allow_loiter": False,
        }
    if ast.action == "REQUEST_CLARIFICATION":
        return {
            "expected_mode": "ACTIVE_ELICITATION",
            "expected_action": "REQUEST_CLARIFICATION",
            "valid_targets": sat_pads,
            "top2_sat_pads": [ast.opt_a, ast.opt_b],
            "expected_unsat_labels": [],
            "requires_clarification": True,
            "requires_abstention": False,
            "allow_loiter": True,
        }
    if ast.action == "HOLD_LOITER":
        return {
            "expected_mode": "BRIEF_CONTINGENCY",
            "expected_action": "HOLD_LOITER",
            "valid_targets": sat_pads,
            "expected_unsat_labels": [],
            "requires_clarification": False,
            "requires_abstention": bucket == "AMBIGUOUS_ASR",
            "allow_loiter": True,
        }
    return {
        "expected_mode": "BRIEF_CONTINGENCY",
        "expected_action": ast.action,
        "valid_targets": [ast.target] if bucket == "NOMINAL" and ast.target else sat_pads,
        "expected_unsat_labels": [],
        "requires_clarification": False,
        "allow_loiter": False,
    }


def interactive_author_loop(
    data_dir: Path,
    *,
    split_name: str,
    bucket: str,
    count: int = 1,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Interactive authoring session with live <0.15 Jaccard + Z3-SAT validation."""
    if split_name not in ("calibration", "test"):
        raise ValueError(f"Interactive authoring only applies to calibration/test, got {split_name!r}")
    if bucket not in EXPECTED_SPLIT_COUNTS[split_name]:
        raise ValueError(f"Unknown bucket {bucket!r}")

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}; run build-train before interactive authoring.")

    train_index = TrainCorpusIndex.from_jsonl(train_path)
    split_path = data_dir / f"{split_name}.jsonl"
    existing_rows = _load_jsonl(split_path) if split_path.exists() else []
    used_seeds = set(train_index.seeds) | {
        int(r["scenario_seed"]) for r in existing_rows if r.get("scenario_seed") is not None
    }
    bucket_existing = sum(1 for r in existing_rows if r.get("bucket") == bucket)
    quota = EXPECTED_SPLIT_COUNTS[split_name][bucket]

    next_seed = EVAL_SEED_BASE[split_name] + len(existing_rows) + 1
    accepted_rows: list[dict[str, Any]] = []

    for _ in range(count):
        while next_seed in used_seeds:
            next_seed += 1
        snap = generate_scaffold_snapshot(bucket, next_seed)
        own = snap["ownship"]
        pads = ", ".join(
            f"{s['id']}(eta={int(round(s['eta_s']))}s,m={s['soc_margin_pct']:+.1f}%)"
            for s in snap["ranked_slzs"][:4]
        )
        output_fn(
            f"\n[VertiBench Author] split={split_name} bucket={bucket} "
            f"progress={bucket_existing + len(accepted_rows)}/{quota} seed={next_seed}"
        )
        output_fn(
            f"  Telemetry (400V/22kWh): FAULT={snap['fault_mode']} AGL={own['agl_m']}m "
            f"SOC={own['soc_pct']}% V={own['v_term_volts']}V WIND={own['wind_mps']}m/s"
        )
        output_fn(f"  Ranked SLZs: {pads}")

        while True:
            utt = input_fn("Utterance (or 'q' to quit): ").strip()
            if utt.lower() == "q":
                return accepted_rows
            c5, t3 = train_index.max_jaccards(utt)
            output_fn(
                f"  Live Jaccard vs train (1,800): char5={c5:.4f} (<{MAX_JACCARD_THRESHOLD}) | "
                f"tok3={t3:.4f} (<{MAX_JACCARD_THRESHOLD})"
            )
            if not utt or c5 >= MAX_JACCARD_THRESHOLD or t3 >= MAX_JACCARD_THRESHOLD:
                output_fn("  REJECTED (contamination >= 0.15 or empty). Rephrase naturally.")
                continue

            lang = input_fn("Language [TR/EN/MIX] (default TR): ").strip().upper() or "TR"
            if lang not in VALID_LANG_CODES:
                output_fn(f"  Invalid lang {lang!r}; expected one of {sorted(VALID_LANG_CODES)}.")
                continue

            line1 = input_fn("Compact DSL line 1 (ACT=...): ").strip()
            line2 = input_fn(f"Compact DSL line 2 (default LNG={lang};MOD=BRIEF): ").strip()
            if not line2:
                default_mod = (
                    "REJECT"
                    if line1.startswith("ACT=REJECT_UNSAFE")
                    else "CLARIFY"
                    if line1.startswith("ACT=REQUEST_CLARIFICATION")
                    else "BRIEF"
                )
                line2 = f"LNG={lang};MOD={default_mod}"
            canonical_dsl = f"{line1}\n{line2}"

            row_idx = bucket_existing + len(accepted_rows)
            candidate: dict[str, Any] = {
                "id": f"vb_{split_name}_{bucket.lower()}_{row_idx:04d}",
                "split": split_name,
                "bucket": bucket,
                "lang": lang,
                "scenario_seed": next_seed,
                "origin_provenance": HUMAN_PROVENANCE,
                "rta_snapshot": snap,
                "user_utterance": utt,
                "asr_corruption_tags": ["human_authored"],
                "gold_intent": _derive_gold_intent(bucket, canonical_dsl, snap),
                "canonical_dsl": canonical_dsl,
            }
            audit_res = audit_candidate_row(candidate, train_index, split_name=split_name)
            output_fn(
                f"  Live Gate: HV={audit_res.hv_ok} DSL={audit_res.dsl_ok} "
                f"Z3_SAT={audit_res.z3_sat} core={audit_res.z3_unsat_core}"
            )
            if not audit_res.passed:
                output_fn(f"  REJECTED: {'; '.join(audit_res.errors)}")
                continue

            split_path.parent.mkdir(parents=True, exist_ok=True)
            with split_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            used_seeds.add(next_seed)
            next_seed += 1
            accepted_rows.append(candidate)
            output_fn(f"  ACCEPTED -> appended {candidate['id']} to {split_path}")
            break

    return accepted_rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VertiBench-TR/EN deterministic train generator & interactive/batch human-split auditor."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "build-train",
        help="Generate deterministic 1,800-row 400V/22kWh GBNF+Z3-filtered train.jsonl.",
    )
    p_train.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmarks/vertibench_tr_en/data"),
        help="Output directory for train.jsonl.",
    )
    p_train.add_argument("--seed", type=int, default=2026)
    p_train.add_argument(
        "--per-bucket",
        type=int,
        default=None,
        help="Optional override count per bucket for fast smoke tests (default: full 1,800 stratified rows).",
    )

    p_audit = sub.add_parser(
        "audit",
        help="Batch-audit calibration.jsonl (200) and/or test.jsonl (560) against train.jsonl.",
    )
    p_audit.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/vertibench_tr_en/data"),
    )
    p_audit.add_argument(
        "--split",
        choices=("calibration", "test", "all"),
        default="all",
    )
    p_audit.add_argument(
        "--allow-partial",
        action="store_true",
        help="Audit partial work-in-progress human splits without failing on incomplete quotas.",
    )
    p_audit.add_argument("--out-json", type=Path, default=None)

    p_check = sub.add_parser(
        "check-utterance",
        help="Live <0.15 Jaccard (and optional GBNF + Z3-SAT) check for one utterance.",
    )
    p_check.add_argument("--train-jsonl", type=Path, required=True)
    p_check.add_argument("--utterance", type=str, required=True)
    p_check.add_argument("--dsl", type=str, default=None)
    p_check.add_argument("--snapshot-json", type=Path, default=None)

    p_author = sub.add_parser(
        "author",
        help="Interactive authoring loop for calibration.jsonl / test.jsonl with live Jaccard+Z3 feedback.",
    )
    p_author.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/vertibench_tr_en/data"),
    )
    p_author.add_argument("--split", choices=("calibration", "test"), required=True)
    p_author.add_argument(
        "--bucket",
        choices=("NOMINAL", "FAULT", "ADVERSARIAL", "AMBIGUOUS_ASR"),
        required=True,
    )
    p_author.add_argument("--count", type=int, default=1)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "build-train":
        if args.per_bucket is not None:
            if args.per_bucket <= 0:
                raise ValueError(f"--per-bucket must be positive, got {args.per_bucket}")
            bucket_counts = {b: args.per_bucket for b in EXPECTED_SPLIT_COUNTS["train"]}
        else:
            bucket_counts = dict(EXPECTED_SPLIT_COUNTS["train"])
        out_path = build_train_split(args.out_dir, seed=args.seed, bucket_counts=bucket_counts)
        rows = _load_jsonl(out_path)
        hv_ok_all = all(validate_hv_telemetry(r.get("rta_snapshot", {})) for r in rows)
        expected_total = sum(bucket_counts.values())
        summary = {
            "command": "build-train",
            "train_jsonl": str(out_path),
            "rows": len(rows),
            "seed": args.seed,
            "pack_spec": "400V_22kWh",
            "hv_telemetry_all_valid": hv_ok_all,
            "buckets": {
                b: sum(1 for r in rows if r.get("bucket") == b)
                for b in EXPECTED_SPLIT_COUNTS["train"]
            },
        }
        print(json.dumps(summary, indent=2))
        return 0 if (len(rows) == expected_total and hv_ok_all) else 1

    if args.command == "audit":
        splits = ("calibration", "test") if args.split == "all" else (args.split,)
        report = audit_dataset_dir(
            args.data_dir,
            splits=splits,
            allow_partial=args.allow_partial,
        )
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(rendered + "\n", encoding="utf-8", newline="\n")
        print(rendered)
        return 0 if report["passed"] else 1

    if args.command == "check-utterance":
        train_index = TrainCorpusIndex.from_jsonl(args.train_jsonl)
        c5, t3 = train_index.max_jaccards(args.utterance)
        jaccard_ok = (c5 < MAX_JACCARD_THRESHOLD) and (t3 < MAX_JACCARD_THRESHOLD)
        payload: dict[str, Any] = {
            "passed": jaccard_ok,
            "char_5gram_jaccard": c5,
            "token_3gram_jaccard": t3,
            "threshold": MAX_JACCARD_THRESHOLD,
        }
        if args.dsl is not None:
            ast = parse_compact_dsl_py(args.dsl)
            payload["dsl_valid"] = ast is not None
            if ast is None:
                payload["passed"] = False
            elif args.snapshot_json is not None:
                snap = json.loads(args.snapshot_json.read_text(encoding="utf-8"))
                hv_ok = validate_hv_telemetry(snap)
                z3_res = verify_z3_static_corridor(ast, snap)
                payload["hv_valid"] = hv_ok
                payload["z3_sat"] = bool(z3_res.is_sat)
                payload["z3_unsat_core"] = list(z3_res.unsat_labels)
                if not (hv_ok and z3_res.is_sat):
                    payload["passed"] = False
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload["passed"] else 1

    if args.command == "author":
        accepted = interactive_author_loop(
            args.data_dir,
            split_name=args.split,
            bucket=args.bucket,
            count=args.count,
        )
        print(json.dumps({"command": "author", "accepted_count": len(accepted)}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
