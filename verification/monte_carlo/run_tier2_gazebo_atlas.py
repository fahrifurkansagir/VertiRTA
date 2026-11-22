"""Tier-2 Verification Pyramid Runner: 1,000 Hot-Reset Parallel PX4/Gazebo Harmonic Atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from verification.monte_carlo.atlas_stats import summarize_atlas_pools
from verification.monte_carlo.cliff_ledger import write_failure_cliff_ledger
from verification.monte_carlo.lhs_sampler import generate_stratified_manifest


FORBIDDEN_PORTS = frozenset({4173, 4174, 8765, 9222, 9223})
BASE_WORKER_UDP_PORT = 8900


def build_gz_hot_reset_cmd(world_name: str = "kartal_coast", worker_id: int = 0) -> list[str]:
    """Build monotonic hot-reset Gazebo service command (model_only: true preserves sim clock).

    Preserves UXRCE_DDS_SYNCT=1 monotonicity and avoids px4-ros2-interface-lib #201/#191.
    """
    if worker_id < 0:
        raise ValueError(f"worker_id must be >= 0, got {worker_id}")
    return [
        "gz",
        "service",
        "-s",
        f"/world/{world_name}/control",
        "--reqtype",
        "gz.msgs.WorldControl",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "5000",
        "--req",
        "reset: {model_only: true}",
    ]


def build_worker_env(worker_id: int) -> dict[str, str]:
    """Create isolated ROS 2 / Gazebo / PX4 environment variables with zero port conflicts."""
    if worker_id < 0 or worker_id >= 32:
        raise ValueError(f"worker_id must be in [0, 31], got {worker_id}")
    udp_port = BASE_WORKER_UDP_PORT + worker_id
    if udp_port in FORBIDDEN_PORTS:
        raise ValueError(f"Worker UDP port {udp_port} collides with forbidden workspace port")
    return {
        "ROS_DOMAIN_ID": str(10 + worker_id),
        "GZ_PARTITION": f"vertirta_w{worker_id}",
        "VERTIRTA_MODEL_NAME": f"vertirta_x8_240_{worker_id}",
        "VERTIRTA_PX4_UDP_PORT": str(udp_port),
        "UXRCE_DDS_SYNCT": "1",
    }


def partition_worker_manifests(
    rows: list[dict[str, Any]],
    num_workers: int = 4,
) -> list[list[dict[str, Any]]]:
    """Partition scenario rows evenly across N isolated workers in deterministic round-robin order."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for idx, row in enumerate(rows):
        shards[idx % num_workers].append(row)
    return shards


def summarize_tier2_results(
    results: list[dict[str, Any]],
    out_dir: Path,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce Sn5 tier2_1k_stratified_atlas.json and failure_cliff_ledger.json with strict isolation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_atlas_pools(results, alpha=alpha)
    ledger = write_failure_cliff_ledger(results, out_dir / "failure_cliff_ledger.json")
    summary["cliff_ledger_sha256"] = ledger["sha256"]
    summary["sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (out_dir / "tier2_1k_stratified_atlas.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary, ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate Tier-2 1,000-run hot-reset PX4/Gazebo Atlas.")
    parser.add_argument("--odd", type=Path, default=Path("verification/ODD.yaml"))
    parser.add_argument("--world", default="kartal_coast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--out-dir", type=Path, default=Path("build/atlas_tier2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = generate_stratified_manifest(args.odd, tier="tier2_1k", seed=args.seed)
    shards = partition_worker_manifests(manifest, num_workers=args.workers)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for k, shard in enumerate(shards):
        shard_path = args.out_dir / f"worker_{k}_manifest.json"
        shard_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8", newline="\n")
        plan.append(
            {
                "worker_id": k,
                "runs": len(shard),
                "env": build_worker_env(k),
                "hot_reset_cmd": build_gz_hot_reset_cmd(args.world, k),
                "manifest": str(shard_path),
            }
        )

    plan_path = args.out_dir / "tier2_dispatch_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")

    if not args.dry_run and shutil.which("gz") is None:
        raise SystemExit("Gazebo Harmonic ('gz') not found on PATH; pass --dry-run to generate dispatch plan.")

    print(json.dumps({"world": args.world, "workers": args.workers, "total_runs": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
