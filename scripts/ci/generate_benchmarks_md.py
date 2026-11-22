"""Generate BENCHMARKS.md with mandatory Cloud-ARM64 Hypervisor & WCET Scope Caveat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXECUTION_CAVEAT_BLOCK = """## Execution Environment & WCET Scope Caveat

> All CI execution benchmarks in this repository are collected on public GitHub-hosted virtualized runners:
> - **ARM64:** `ubuntu-24.04-arm` (Microsoft Azure **Cobalt 100** / Armv9.0-A **Neoverse N2**, 4 vCPU, 16 GB RAM, Ubuntu 24.04 LTS, Azure Linux kernel).
> - **x86_64:** `ubuntu-24.04` (4 vCPU, 16 GB RAM, Ubuntu 24.04 LTS, Azure Linux kernel).
>
> **Strict Interpretation Rules (NO Bare-Metal WCET Claims):**
> 1. **Hypervisor Steal & Multi-Tenant Jitter:** `CLOCK_MONOTONIC_RAW` wall-clock tail latencies (`p99`, `p99.9`, `max`) measured inside multi-tenant cloud VMs include host hypervisor scheduling preemption (reported alongside `/proc/stat` CPU `steal` % and involuntary context switches). Runs exceeding `1.0%` hypervisor steal are explicitly flagged **`NOISY_NEIGHBOR_TAINTED`**.
> 2. **NOT Certifiable RTOS WCET Evidence:** Linux user-space wall-clock timings on virtualized Cobalt 100 / x86_64 runners are indicative CI performance regression metrics ONLY. They do **NOT** constitute Worst-Case Execution Time (WCET) certification evidence under DO-178C or EASA `SC-VTOL-02` / `MOC VTOL.2510`.
> 3. **Hardware-Independent Structural Guarantees:** Auditable determinism claims rest on **Tier 1** (link-time 0-heap trap + `nm -u` ban, `-fstack-usage` + `-fcallgraph-info=su,da` DAG, Cortex-M7 `< 256 KB` Flash / `< 128 KB` RAM), **Tier 2** (0-ULP SHA-256 Same-ISA & `< 1e-12` / 0 mode divergence Cross-ISA), and **Tier 3** (`CLOCK_THREAD_CPUTIME_ID` + `perf stat` / `valgrind --tool=callgrind` deterministic instruction count).
"""


def render_benchmarks_markdown(
    tier1: dict[str, Any],
    tier2: dict[str, Any],
    tier3_by_arch: dict[str, dict[str, Any]],
    tier4_by_arch: dict[str, dict[str, Any]],
) -> str:
    """Render deterministic 4-Tier BENCHMARKS.md from CI artifact summaries."""
    lines = [
        "# VertiRTA — 4-Tier Execution Determinism & Bare-Metal Footprint Benchmarks",
        "",
        EXECUTION_CAVEAT_BLOCK.strip(),
        "",
        "---",
        "",
        "## Tier 1 — Hardware-Independent Structural & Cortex-M7 Footprint (`libvertirta_flight_core`)",
        "",
        "| Metric | Measured Value | Budget / Gate | Status |",
        "| --- | --- | --- | --- |",
        f"| Cortex-M7 Flash (`.text + .data`) | `{tier1.get('flash_bytes', 'N/A')} B` | `< 262144 B` (`256 KB`) | `{'PASS' if tier1.get('passed') else 'FAIL'}` |",
        f"| Cortex-M7 RAM (`.data + .bss`) | `{tier1.get('ram_bytes', 'N/A')} B` | `< 131072 B` (`128 KB`) | `{'PASS' if tier1.get('passed') else 'FAIL'}` |",
        f"| Max Static Frame (`-fstack-usage`) | `{tier1.get('max_frame_bytes', 'N/A')} B` | `<= 4096 B` | `{'PASS' if tier1.get('passed') else 'FAIL'}` |",
        f"| Max Call-Graph DAG (`-fcallgraph-info`) | `{tier1.get('max_dag_bytes', 'N/A')} B` | `<= 16384 B` (0 recursion) | `{'PASS' if tier1.get('passed') else 'FAIL'}` |",
        "",
        "## Tier 2 — `.vrta` Deterministic Replay (`Same-ISA` & `x86_64 <-> ARM64`)",
        "",
        f"- **Same-ISA 0-ULP SHA-256 Match:** `{tier2.get('same_isa_0ulp', False)}`",
        f"- **Cross-ISA Max Continuous Deviation:** `{tier2.get('max_abs_diff', 'N/A')}` (limit `< 1e-12`)",
        f"- **Cross-ISA Discrete Mode Divergences:** `{tier2.get('mode_divergences', 'N/A')}` (limit `0`)",
        "",
        "## Tier 3 & Tier 4 — Work Units (`perf`/`callgrind`) & Transparent Wall-Clock Tail",
        "",
        "| Runner Arch | PMU / Fallback Mode | `Ir / tick` | `p50` (us) | `p99` (us) | `p99.9` (us) | Steal % | Taint Flag |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arch in sorted(set(tier3_by_arch) | set(tier4_by_arch)):
        t3 = tier3_by_arch.get(arch, {})
        t4 = tier4_by_arch.get(arch, {})
        lines.append(
            f"| `{arch}` | `{t3.get('pmu_mode', 'N/A')}` | `{t3.get('ir_per_tick', 'N/A')}` | "
            f"`{t4.get('p50_us', 'N/A')}` | `{t4.get('p99_us', 'N/A')}` | `{t4.get('p999_us', 'N/A')}` | "
            f"`{t4.get('steal_pct', 'N/A')}%` | `{t4.get('taint', 'N/A')}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render BENCHMARKS.md from Tier 1-4 JSON outputs.")
    parser.add_argument("--tier1", type=Path, required=True)
    parser.add_argument("--tier2", type=Path, required=True)
    parser.add_argument("--tier3-arm64", type=Path, required=True)
    parser.add_argument("--tier4-arm64", type=Path, required=True)
    parser.add_argument("--tier3-x86", type=Path, default=None)
    parser.add_argument("--tier4-x86", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t1 = json.loads(args.tier1.read_text(encoding="utf-8"))
    t2 = json.loads(args.tier2.read_text(encoding="utf-8"))
    t3 = {"arm64": json.loads(args.tier3_arm64.read_text(encoding="utf-8"))}
    t4 = {"arm64": json.loads(args.tier4_arm64.read_text(encoding="utf-8"))}
    if args.tier3_x86 and args.tier3_x86.exists():
        t3["x86_64"] = json.loads(args.tier3_x86.read_text(encoding="utf-8"))
    if args.tier4_x86 and args.tier4_x86.exists():
        t4["x86_64"] = json.loads(args.tier4_x86.read_text(encoding="utf-8"))

    md = render_benchmarks_markdown(t1, t2, t3, t4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
