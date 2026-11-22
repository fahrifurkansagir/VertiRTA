"""VertiBench-TR/EN Schema, License Allowlist, Anti-Contamination (<0.15 Jaccard) & Compact DSL Gate.

Enforces:
- Apache-2.0 model license hygiene (train: Qwen3-8B / Qwen3-4B only; zero-shot eval: +Qwen2.5-7B / Gemma-3-4B;
  hard-ban: CohereForAI/aya-expanse-8b CC-BY-NC-4.0).
- Locked 400 V / 22 kWh HV electrical telemetry (v_term_volts in [320, 420] V; rejects legacy 44.2 V 12S).
- Compact <=30-token ASCII GBNF DSL parser mirroring core/copilot/dsl_parser.cpp.
- Anti-contamination split check: 0 scenario-seed overlap and <0.15 character 5-gram & token 3-gram Jaccard
  between human-audited (calibration U test) splits and programmatic train split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_TRAIN_MODELS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B",
    }
)
ALLOWED_ZERO_SHOT_EVAL_MODELS: frozenset[str] = ALLOWED_TRAIN_MODELS | frozenset(
    {
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-3-4b-it",
    }
)
BANNED_MODELS: frozenset[str] = frozenset(
    {
        "CohereForAI/aya-expanse-8b",
    }
)

VALID_SLZ_IDS: frozenset[str] = frozenset(
    {
        "Kartal_Sahil",
        "Dragos_Plaza",
        "Maltepe_Park",
        "Cevizli_Helipad",
    }
)
VALID_VIOLATION_CODES: frozenset[str] = frozenset(
    {
        "LTFJ_CTR",
        "LTBX_CTR",
        "DYN_NFZ",
        "SOC_RESERVE",
        "SLZ_UNSAFE",
        "MIN_AGL",
        "SHIELD_BYPASS",
    }
)
VALID_CLARIFY_REASONS: frozenset[str] = frozenset({"AMBIG", "NO_SLOT", "LOW_ASR"})
VALID_LANG_CODES: frozenset[str] = frozenset({"TR", "EN", "MIX"})
VALID_MODE_CODES: frozenset[str] = frozenset({"BRIEF", "REJECT", "CLARIFY"})

EXPECTED_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "train": {
        "NOMINAL": 450,
        "FAULT": 520,
        "ADVERSARIAL": 450,
        "AMBIGUOUS_ASR": 380,
    },
    "calibration": {
        "NOMINAL": 50,
        "FAULT": 55,
        "ADVERSARIAL": 50,
        "AMBIGUOUS_ASR": 45,
    },
    "test": {
        "NOMINAL": 140,
        "FAULT": 160,
        "ADVERSARIAL": 140,
        "AMBIGUOUS_ASR": 120,
    },
}

MAX_JACCARD_THRESHOLD: float = 0.15


@dataclass(frozen=True)
class CompactDslAstPy:
    action: str
    target: str | None = None
    opt_a: str | None = None
    opt_b: str | None = None
    clarify_reason: str | None = None
    violation: str | None = None
    fallback: str | None = None
    eta_lim_s: int = 0
    duration_s: int = 0
    agl_min_m: int = 35
    agl_max_m: int = 120
    speed_max_mps: int = 18
    lang: str = "TR"
    mode: str = "BRIEF"


@dataclass
class ValidationReport:
    passed: bool
    schema_only: bool
    split_counts: dict[str, int] = field(default_factory=dict)
    max_char_5gram_jaccard: float = 0.0
    max_token_3gram_jaccard: float = 0.0
    errors: list[str] = field(default_factory=list)


def is_model_allowed(model_id: str, mode: str) -> bool:
    """Enforce Apache-2.0 training lock & zero-shot eval allowlist (bans CC-BY-NC models)."""
    if model_id in BANNED_MODELS:
        return False
    if mode == "train":
        return model_id in ALLOWED_TRAIN_MODELS
    if mode == "zero_shot_eval":
        return model_id in ALLOWED_ZERO_SHOT_EVAL_MODELS
    raise ValueError(f"Unsupported mode {mode!r}; expected 'train' or 'zero_shot_eval'")


def validate_hv_telemetry(snapshot: dict[str, Any]) -> bool:
    """Validate 400 V / 22 kWh HV pack bounds (DR-6 Erratum #1; rejects 44.2 V 12S packs)."""
    own = snapshot.get("ownship", snapshot)
    if not isinstance(own, dict):
        return False
    if "v_term_volts" not in own or "soc_pct" not in own:
        return False
    try:
        v_term = float(own["v_term_volts"])
        soc = float(own["soc_pct"])
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(v_term) and math.isfinite(soc)):
        return False
    if not (320.0 <= v_term <= 420.0):
        return False
    if not (0.0 <= soc <= 100.0):
        return False
    if "pack_nominal_volts" in snapshot:
        p_nom = float(snapshot["pack_nominal_volts"])
        if not (380.0 <= p_nom <= 420.0):
            return False
    if "pack_energy_kwh" in snapshot:
        p_kwh = float(snapshot["pack_energy_kwh"])
        if not (20.0 <= p_kwh <= 24.0):
            return False
    return True


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    norm = _normalize_text(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def _token_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = re.findall(r"[^\W_]+", _normalize_text(text), flags=re.UNICODE)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / float(union)) if union > 0 else 0.0


def max_char_5gram_jaccard(text: str, corpus: Sequence[str]) -> float:
    """Maximum character 5-gram Jaccard similarity between text and any prompt in corpus."""
    target = _char_ngrams(text, 5)
    best = 0.0
    for item in corpus:
        score = _jaccard(target, _char_ngrams(item, 5))
        if score > best:
            best = score
    return best


def max_token_3gram_jaccard(text: str, corpus: Sequence[str]) -> float:
    """Maximum token 3-gram Jaccard similarity between text and any prompt in corpus."""
    target = _token_ngrams(text, 3)
    best = 0.0
    for item in corpus:
        score = _jaccard(target, _token_ngrams(item, 3))
        if score > best:
            best = score
    return best


def _parse_int_bounded(val: str, lo: int, hi: int) -> int | None:
    if not val or len(val) > 4 or not val.isdigit():
        return None
    if len(val) > 1 and val[0] == "0":
        return None
    parsed = int(val)
    return parsed if lo <= parsed <= hi else None


def parse_compact_dsl_py(text: str) -> CompactDslAstPy | None:
    """Strict Python mirror of C++20 parse_compact_dsl (config/vertirta_dsl_v1.gbnf)."""
    if not text:
        return None
    for ch in text:
        code = ord(ch)
        if ch in ("\r", "<", ">") or (code < 0x20 and ch != "\n") or code >= 0x7F:
            return None
    cleaned = text[:-1] if text.endswith("\n") else text
    lines = cleaned.split("\n")
    if len(lines) != 2:
        return None

    def split_pairs(line: str) -> list[tuple[str, str]] | None:
        if not line or line.endswith(";"):
            return None
        pairs: list[tuple[str, str]] = []
        for field_str in line.split(";"):
            if "=" not in field_str:
                return None
            k, v = field_str.split("=", 1)
            if not k or not v:
                return None
            pairs.append((k, v))
        return pairs

    p1 = split_pairs(lines[0])
    p2 = split_pairs(lines[1])
    if p1 is None or p2 is None:
        return None
    if len(p2) != 2 or p2[0][0] != "LNG" or p2[1][0] != "MOD":
        return None
    lang, mode = p2[0][1], p2[1][1]
    if lang not in VALID_LANG_CODES or mode not in VALID_MODE_CODES:
        return None

    if not p1 or p1[0][0] != "ACT":
        return None
    action = p1[0][1]

    if action in ("DIVERT_SLZ", "INSPECT_SLZ", "ABORT_RETURN"):
        if len(p1) != 5:
            return None
        if [k for k, _ in p1] != ["ACT", "TGT", "ETA", "AGL", "SPD"]:
            return None
        tgt = p1[1][1]
        if tgt not in VALID_SLZ_IDS:
            return None
        eta = _parse_int_bounded(p1[2][1], 1, 600)
        agl_parts = p1[3][1].split(",")
        if len(agl_parts) != 2:
            return None
        agl_min = _parse_int_bounded(agl_parts[0], 10, 150)
        agl_max = _parse_int_bounded(agl_parts[1], 30, 150)
        spd = _parse_int_bounded(p1[4][1], 5, 30)
        if eta is None or agl_min is None or agl_max is None or spd is None:
            return None
        return CompactDslAstPy(
            action=action,
            target=tgt,
            eta_lim_s=eta,
            agl_min_m=agl_min,
            agl_max_m=agl_max,
            speed_max_mps=spd,
            lang=lang,
            mode=mode,
        )

    if action == "HOLD_LOITER":
        if len(p1) != 3 or [k for k, _ in p1] != ["ACT", "DUR", "AGL"]:
            return None
        dur = _parse_int_bounded(p1[1][1], 10, 300)
        agl = _parse_int_bounded(p1[2][1], 30, 150)
        if dur is None or agl is None:
            return None
        return CompactDslAstPy(
            action=action,
            target=None,
            duration_s=dur,
            eta_lim_s=dur,
            agl_min_m=agl,
            agl_max_m=agl,
            speed_max_mps=5,
            lang=lang,
            mode=mode,
        )

    if action == "REQUEST_CLARIFICATION":
        if len(p1) != 4 or [k for k, _ in p1] != ["ACT", "A", "B", "RSN"]:
            return None
        opt_a, opt_b, rsn = p1[1][1], p1[2][1], p1[3][1]
        if opt_a not in VALID_SLZ_IDS or opt_b not in VALID_SLZ_IDS:
            return None
        if rsn not in VALID_CLARIFY_REASONS:
            return None
        return CompactDslAstPy(
            action=action,
            target=None,
            opt_a=opt_a,
            opt_b=opt_b,
            clarify_reason=rsn,
            duration_s=30,
            eta_lim_s=30,
            lang=lang,
            mode=mode,
        )

    if action == "REJECT_UNSAFE":
        if len(p1) != 3 or [k for k, _ in p1] != ["ACT", "VIO", "FB"]:
            return None
        vio, fb = p1[1][1], p1[2][1]
        if vio not in VALID_VIOLATION_CODES:
            return None
        if fb not in VALID_SLZ_IDS and fb != "HOLD_LOITER":
            return None
        return CompactDslAstPy(
            action=action,
            target=None,
            violation=vio,
            fallback=fb,
            eta_lim_s=100,
            lang=lang,
            mode=mode,
        )

    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def validate_vertibench(
    data_dir: Path | None = None,
    *,
    check_schema_only: bool = False,
) -> ValidationReport:
    """Validate VertiBench-TR/EN schema, license card, and (when present) split integrity."""
    errors: list[str] = []

    gbnf_path = REPO_ROOT / "config" / "vertirta_dsl_v1.gbnf"
    readme_path = REPO_ROOT / "benchmarks" / "vertibench_tr_en" / "README.md"
    if not gbnf_path.exists():
        errors.append(f"Missing GBNF grammar: {gbnf_path}")
    else:
        gbnf_text = gbnf_path.read_text(encoding="utf-8")
        if '"ACT="' not in gbnf_text or "STL=" in gbnf_text:
            errors.append("config/vertirta_dsl_v1.gbnf does not match compact ACT= grammar")

    if not readme_path.exists():
        errors.append(f"Missing Hugging Face Dataset Card: {readme_path}")
    else:
        card = readme_path.read_text(encoding="utf-8")
        if "license: apache-2.0" not in card:
            errors.append("Dataset Card missing 'license: apache-2.0'")

    if check_schema_only:
        sample = "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF"
        if parse_compact_dsl_py(sample) is None:
            errors.append("Compact DSL self-check failed")
        if not is_model_allowed("Qwen/Qwen3-8B", "train") or is_model_allowed(
            "CohereForAI/aya-expanse-8b", "zero_shot_eval"
        ):
            errors.append("License allowlist self-check failed")
        return ValidationReport(passed=len(errors) == 0, schema_only=True, errors=errors)

    if data_dir is None:
        data_dir = REPO_ROOT / "benchmarks" / "vertibench_tr_en" / "data"

    loaded: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, int] = {}
    for split_name, expected_buckets in EXPECTED_SPLIT_COUNTS.items():
        p = data_dir / f"{split_name}.jsonl"
        if not p.exists():
            errors.append(f"Missing split file: {p}")
            continue
        rows = _load_jsonl(p)
        loaded[split_name] = rows
        split_counts[split_name] = len(rows)

        bucket_counts: dict[str, int] = {k: 0 for k in expected_buckets}
        for row in rows:
            b = row.get("bucket", "")
            if b not in bucket_counts:
                errors.append(f"{split_name}/{row.get('id')}: invalid bucket {b!r}")
            else:
                bucket_counts[b] += 1
            if not validate_hv_telemetry(row.get("rta_snapshot", {})):
                errors.append(f"{split_name}/{row.get('id')}: invalid 400V/22kWh telemetry")
            if parse_compact_dsl_py(row.get("canonical_dsl", "")) is None:
                errors.append(f"{split_name}/{row.get('id')}: invalid canonical_dsl")
            if split_name in ("calibration", "test"):
                if row.get("origin_provenance") != "human_native_tr_v1":
                    errors.append(
                        f"{split_name}/{row.get('id')}: expected origin_provenance='human_native_tr_v1'"
                    )

        for b_name, exp_cnt in expected_buckets.items():
            if bucket_counts[b_name] != exp_cnt:
                errors.append(
                    f"{split_name} bucket {b_name}: expected {exp_cnt}, got {bucket_counts[b_name]}"
                )

    max_c5 = 0.0
    max_t3 = 0.0
    if "train" in loaded:
        train_rows = loaded["train"]
        train_seeds = {r.get("scenario_seed") for r in train_rows if "scenario_seed" in r}
        train_utts = [str(r.get("user_utterance", "")) for r in train_rows]

        for eval_split in ("calibration", "test"):
            for row in loaded.get(eval_split, []):
                seed = row.get("scenario_seed")
                if seed is not None and seed in train_seeds:
                    errors.append(f"{eval_split}/{row.get('id')}: leaked scenario_seed {seed}")
                utt = str(row.get("user_utterance", ""))
                c5 = max_char_5gram_jaccard(utt, train_utts)
                t3 = max_token_3gram_jaccard(utt, train_utts)
                max_c5 = max(max_c5, c5)
                max_t3 = max(max_t3, t3)
                if c5 >= MAX_JACCARD_THRESHOLD:
                    errors.append(
                        f"{eval_split}/{row.get('id')}: char 5-gram Jaccard {c5:.3f} >= {MAX_JACCARD_THRESHOLD}"
                    )
                if t3 >= MAX_JACCARD_THRESHOLD:
                    errors.append(
                        f"{eval_split}/{row.get('id')}: token 3-gram Jaccard {t3:.3f} >= {MAX_JACCARD_THRESHOLD}"
                    )

    return ValidationReport(
        passed=len(errors) == 0,
        schema_only=False,
        split_counts=split_counts,
        max_char_5gram_jaccard=round(max_c5, 4),
        max_token_3gram_jaccard=round(max_t3, 4),
        errors=errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VertiBench-TR/EN dataset & schema.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--check-schema",
        action="store_true",
        help="Validate grammar, license card, and schema contracts without requiring human splits.",
    )
    args = parser.parse_args()
    report = validate_vertibench(args.data_dir, check_schema_only=args.check_schema)
    print(json.dumps(report.__dict__, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
