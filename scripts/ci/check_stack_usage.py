"""Tier-1 CI Gate: Static stack frame & `-fcallgraph-info=su,da` DAG depth + recursion ban."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


CI_NODE_RE = re.compile(r'node:\s*\{\s*title:\s*"([^"]+)"(?:\s+label:\s*"([^"]*)")?')
CI_EDGE_RE = re.compile(r'edge:\s*\{\s*sourcename:\s*"([^"]+)"\s+targetname:\s*"([^"]+)"')
CI_BYTES_RE = re.compile(r"(\d+)\s+bytes\s+\(([^)]+)\)")


def _validate_qualifier(qualifier: str, context: str) -> bool:
    """Validate GCC -fstack-usage qualifier: allow static & dynamic,bounded; ban unbounded dynamic.

    Returns True if the frame is dynamic,bounded (e.g. x86-64 stack alignment with known static cap).
    """
    norm = qualifier.strip().lower()
    if norm == "static":
        return False
    if norm == "dynamic,bounded":
        return True
    raise ValueError(f"Unbounded dynamic stack frame detected ({qualifier}) in {context}")


def _parse_su_files(build_dir: Path) -> tuple[dict[str, tuple[int, str]], int]:
    su_files = sorted(build_dir.rglob("*.su"))
    if not su_files:
        raise FileNotFoundError(
            f"No .su stack-usage files found under {build_dir} (ensure -fno-lto -fstack-usage)"
        )

    frames: dict[str, tuple[int, str]] = {}
    bounded_realigned_count = 0
    for su_file in su_files:
        for raw_line in su_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            location_and_fn, bytes_str, qualifier = parts[0], parts[1], parts[2].strip()
            if _validate_qualifier(qualifier, f"{su_file}: {location_and_fn}"):
                bounded_realigned_count += 1
            frame_bytes = int(bytes_str)
            fn_name = location_and_fn.split(":")[-1]
            prev = frames.get(fn_name)
            if prev is None or frame_bytes > prev[0]:
                frames[fn_name] = (frame_bytes, qualifier)
    return frames, bounded_realigned_count


def _parse_ci_callgraphs(build_dir: Path) -> tuple[dict[str, int], dict[str, set[str]]]:
    node_bytes: dict[str, int] = {}
    edges: dict[str, set[str]] = {}
    for ci_file in sorted(build_dir.rglob("*.ci")):
        text = ci_file.read_text(encoding="utf-8", errors="replace")
        for m in CI_NODE_RE.finditer(text):
            title, label = m.group(1), m.group(2) or ""
            edges.setdefault(title, set())
            bm = CI_BYTES_RE.search(label)
            if bm:
                bval, qual = int(bm.group(1)), bm.group(2).strip()
                _validate_qualifier(qual, f"{ci_file} node '{title}'")
                node_bytes[title] = max(node_bytes.get(title, 0), bval)
        for m in CI_EDGE_RE.finditer(text):
            src, dst = m.group(1), m.group(2)
            edges.setdefault(src, set()).add(dst)
            edges.setdefault(dst, set())
    return node_bytes, edges


def _compute_max_dag_stack(
    node_bytes: dict[str, int],
    edges: dict[str, set[str]],
    su_frames: dict[str, tuple[int, str]],
) -> tuple[int, list[str]]:
    # Map short names from .su as fallback when a .ci node omits byte count
    def self_stack(node: str) -> int:
        if node in node_bytes:
            return node_bytes[node]
        if node in su_frames:
            return su_frames[node][0]
        return 0

    color: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
    memo: dict[str, tuple[int, list[str]]] = {}

    def dfs(u: str) -> tuple[int, list[str]]:
        state = color.get(u, 0)
        if state == 1:
            raise ValueError(f"Forbidden recursion cycle detected in callgraph involving '{u}'")
        if state == 2:
            return memo[u]
        color[u] = 1
        best_child_bytes = 0
        best_child_path: list[str] = []
        for v in sorted(edges.get(u, ())):
            c_bytes, c_path = dfs(v)
            if c_bytes > best_child_bytes:
                best_child_bytes = c_bytes
                best_child_path = c_path
        color[u] = 2
        total = self_stack(u) + best_child_bytes
        path = [u] + best_child_path
        memo[u] = (total, path)
        return memo[u]

    max_total = 0
    max_path: list[str] = []
    for node in sorted(edges):
        total, path = dfs(node)
        if total > max_total:
            max_total = total
            max_path = path

    max_su_single = max((v[0] for v in su_frames.values()), default=0)
    if max_su_single > max_total:
        max_total = max_su_single
    return max_total, max_path


def parse_su_and_ci_dag(
    build_dir: Path,
    max_frame_bytes: int = 4096,
    max_dag_bytes: int = 16384,
) -> dict[str, Any]:
    """Parse GCC .su and .ci callgraph files, banning unbounded dynamic frames and recursion."""
    su_frames, bounded_realigned_count = _parse_su_files(build_dir)
    node_bytes, edges = _parse_ci_callgraphs(build_dir)
    max_frame = max((v[0] for v in su_frames.values()), default=0)
    max_dag, worst_chain = _compute_max_dag_stack(node_bytes, edges, su_frames)

    passed = (max_frame <= max_frame_bytes) and (max_dag <= max_dag_bytes)
    report: dict[str, Any] = {
        "schema_version": "vertirta.tier1_stack_dag/v1.0.0",
        "build_dir": str(build_dir),
        "functions_checked": len(su_frames),
        "bounded_realigned_frames": bounded_realigned_count,
        "callgraph_nodes": len(edges),
        "max_frame_bytes": max_frame,
        "limit_frame_bytes": max_frame_bytes,
        "max_dag_bytes": max_dag,
        "limit_dag_bytes": max_dag_bytes,
        "worst_call_chain": worst_chain[:16],
        "passed": passed,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify static stack frame and callgraph DAG bounds.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--max-frame-bytes", type=int, default=4096)
    parser.add_argument("--max-dag-bytes", type=int, default=16384)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = parse_su_and_ci_dag(
        args.build_dir,
        max_frame_bytes=args.max_frame_bytes,
        max_dag_bytes=args.max_dag_bytes,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
