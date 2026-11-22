"""SCSC-141C GSN v3 Living Safety Case builder and MLEAP ODD.yaml validator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import yaml


GSN_BEGIN_MARKER = "<!-- GSN:BEGIN -->"
GSN_END_MARKER = "<!-- GSN:END -->"
MANDATORY_UNDEVELOPED_GOALS = ("G5", "G6")
J2_MAX_SAFE_LON_DEG = 29.1900


def validate_odd_yaml(odd: dict[str, Any]) -> dict[str, Any]:
    """Validate verification/ODD.yaml against frozen v1.2 spec & DR-3/DR-4 errata."""
    errors: list[str] = []

    if odd.get("compliance_claim") != "NONE_RESEARCH_TESTBED_ONLY":
        errors.append("compliance_claim must equal NONE_RESEARCH_TESTBED_ONLY")

    od = odd.get("operational_domain", {})
    scenery = od.get("scenery_elements", {})
    datum = scenery.get("datum_origin", {})
    if (
        datum.get("lat_deg") != 40.8860
        or datum.get("lon_deg") != 29.1880
        or datum.get("alt_amsl_m") != 12.0
    ):
        errors.append("datum_origin must lock Kartal Coastal Vertiport (40.8860 N, 29.1880 E, 12.0 m AMSL)")

    if scenery.get("egm2008_undulation_m") != 36.7:
        errors.append("egm2008_undulation_m must equal 36.7 m")

    lon_bounds = scenery.get("corridor_bounding_box", {}).get("lon_deg", [0.0, 999.0])
    if len(lon_bounds) != 2 or lon_bounds[1] >= J2_MAX_SAFE_LON_DEG:
        errors.append(
            f"corridor_bounding_box.lon_deg upper bound ({lon_bounds}) must stay strictly < {J2_MAX_SAFE_LON_DEG:.4f} deg E west of LTFJ J2 (29.19212 deg E)"
        )

    slz = od.get("slz_requirements", {})
    if slz.get("r_slz_min_m", 0.0) < 5.0:
        errors.append("slz_requirements.r_slz_min_m must be >= 5.0 m")
    if slz.get("slope_max_deg", 99.0) > 8.0:
        errors.append("slz_requirements.slope_max_deg must be <= 8.0 deg")
    if slz.get("roughness_sigma_z_max_m", 99.0) > 0.15:
        errors.append("slz_requirements.roughness_sigma_z_max_m must be <= 0.15 m")

    ops = od.get("operating_parameters", {})
    if ops.get("map_staleness_inflate_ms") != 250:
        errors.append("operating_parameters.map_staleness_inflate_ms must equal 250 ms")

    stage_a = od.get("regulatory_stage_split", {}).get("stage_a_subscale_unmanned", {})
    if stage_a.get("passengers") != 0:
        errors.append("Stage-A SORA 2.5 subscale CONOPS must enforce passengers == 0")

    alternates = scenery.get("designated_vertiports_and_alternates", {})
    if alternates.get("min_reachable_alternates_past_poc", 0) < 2:
        errors.append("Part-IAM requires min_reachable_alternates_past_poc >= 2")

    ai_odds = odd.get("ai_constituent_odds", [])
    if not ai_odds:
        errors.append("ai_constituent_odds must define DR-6 and DR-7 isolation contracts")
    for item in ai_odds:
        if item.get("actuation_authority") != "ZERO":
            errors.append(f"{item.get('constituent_id')} must enforce actuation_authority == ZERO")

    payload = {
        "schema_version": "vertirta.odd_validation/v1.0.0",
        "od_id": od.get("od_id", "UNKNOWN"),
        "passed": len(errors) == 0,
        "errors": errors,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["sha256"] = digest
    return payload


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def build_gsn_mermaid(
    gsn: dict[str, Any],
    artifact_hashes: dict[str, str],
    git_sha: str,
    *,
    require_all_hashes: bool = False,
) -> str:
    """Compile SCSC-141C GSN v3 YAML into a CI-hash-linked Mermaid flowchart."""
    nodes = gsn.get("nodes", [])
    by_id = {node["id"]: node for node in nodes}

    for goal_id in MANDATORY_UNDEVELOPED_GOALS:
        node = by_id.get(goal_id)
        if node is None or node.get("undeveloped") is not True:
            raise ValueError(
                f"Audit violation: Goal {goal_id} MUST remain marked undeveloped: true in GSN v3"
            )

    short_sha = git_sha[:10] if git_sha else "UNCOMMITTED"
    lines: list[str] = [
        "```mermaid",
        f"%% VertiRTA SCSC-141C GSN v3 Living Safety Case | commit={short_sha}",
        "flowchart TD",
    ]

    for node in nodes:
        nid = node["id"]
        ntype = node["type"]
        if ntype == "Goal":
            prefix = f"{nid} [UNDEVELOPED]" if node.get("undeveloped") else nid
            label = _escape_mermaid(f"{prefix}: {node['text']}")
            lines.append(f'    {nid}["{label}"]')
        elif ntype == "Strategy":
            label = _escape_mermaid(f"{nid}: {node['text']}")
            lines.append(f'    {nid}[/"{label}"/]')
        elif ntype in {"Context", "Assumption", "Justification"}:
            label = _escape_mermaid(f"{nid}: {node['text']}")
            lines.append(f'    {nid}(["{label}"])')
        elif ntype == "Solution":
            digest = artifact_hashes.get(nid)
            if require_all_hashes and not digest:
                raise ValueError(f"Release GSN build missing CI SHA-256 hash for Solution {nid}")
            if digest and digest.startswith("PENDING"):
                tag = digest
            else:
                tag = digest[:12] if digest else f"PENDING@{short_sha}"
            label = _escape_mermaid(f"{nid}: {node['label']}<br/>[{tag}]")
            lines.append(f'    {nid}(("{label}"))')
        else:
            raise ValueError(f"Unsupported GSN node type: {ntype}")

    for node in nodes:
        nid = node["id"]
        for ctx in node.get("in_context_of", []):
            lines.append(f"    {nid} -.-> {ctx}")
        for child in node.get("supported_by", []):
            lines.append(f"    {nid} --> {child}")

    for node in nodes:
        nid = node["id"]
        if node.get("undeveloped"):
            lines.append(f"    style {nid} fill:#FFF3CD,stroke:#D93025,stroke-dasharray: 5 5")
        elif node["type"] == "Solution":
            lines.append(f"    style {nid} fill:#E8F0FE,stroke:#1A73E8")

    lines.append("```")
    return "\n".join(lines)


def inject_gsn_into_safety_case(safety_case_path: Path, mermaid_block: str) -> None:
    """Deterministically replace the GSN Mermaid block inside docs/SAFETY_CASE.md."""
    text = safety_case_path.read_text(encoding="utf-8")
    begin_idx = text.find(GSN_BEGIN_MARKER)
    end_idx = text.find(GSN_END_MARKER)
    if begin_idx == -1 or end_idx == -1 or end_idx <= begin_idx:
        raise ValueError(
            f"Missing valid {GSN_BEGIN_MARKER} ... {GSN_END_MARKER} markers in {safety_case_path}"
        )
    replacement = f"{GSN_BEGIN_MARKER}\n{mermaid_block}\n{GSN_END_MARKER}"
    updated = text[:begin_idx] + replacement + text[end_idx + len(GSN_END_MARKER) :]
    safety_case_path.write_text(updated, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ODD.yaml and build GSN v3 Mermaid diagram.")
    parser.add_argument("--gsn", type=Path, default=Path("verification/gsn/safety_case_gsn.yaml"))
    parser.add_argument("--odd", type=Path, default=Path("verification/ODD.yaml"))
    parser.add_argument("--safety-case", type=Path, default=Path("docs/SAFETY_CASE.md"))
    parser.add_argument("--hashes-json", type=Path, default=None)
    parser.add_argument("--git-sha", default="local-dev")
    parser.add_argument("--require-all-hashes", action="store_true")
    parser.add_argument("--out-odd-report", type=Path, default=None)
    args = parser.parse_args()

    odd = yaml.safe_load(args.odd.read_text(encoding="utf-8"))
    odd_report = validate_odd_yaml(odd)
    if args.out_odd_report is not None:
        args.out_odd_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_odd_report.write_text(
            json.dumps(odd_report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    if not odd_report["passed"]:
        raise SystemExit(f"ODD validation failed: {odd_report['errors']}")

    hashes: dict[str, str] = {}
    if args.hashes_json is not None and args.hashes_json.exists():
        hashes = json.loads(args.hashes_json.read_text(encoding="utf-8"))
    hashes.setdefault("Sn3", odd_report["sha256"])

    gsn = yaml.safe_load(args.gsn.read_text(encoding="utf-8"))
    mermaid = build_gsn_mermaid(
        gsn,
        artifact_hashes=hashes,
        git_sha=args.git_sha,
        require_all_hashes=args.require_all_hashes,
    )
    inject_gsn_into_safety_case(args.safety_case, mermaid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
