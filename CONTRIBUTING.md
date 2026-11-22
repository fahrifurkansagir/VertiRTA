# Contributing to VertiRTA

Thank you for contributing to **VertiRTA**! All contributions are licensed under the **Apache License 2.0** ([LICENSE](LICENSE)).

> **Non-Certification Scope:** VertiRTA is an open-source research, simulation, and software-verification testbed only. Never commit documentation or paper edits that claim DO-178C, DO-254, EASA `SC-VTOL-02`, `SORA 2.5`, or `SHGM SHT-İHA` certification or operational compliance.

---

## 1. Governance: DCO 1.1 (No Corporate CLA), Permanent Apache-2.0 Core & License/CVE Firewall

1. **DCO 1.1 Sign-Off (No Corporate CLA):** VertiRTA is governed strictly under **Developer Certificate of Origin (`DCO 1.1`)** (`inbound = outbound` under `Apache-2.0`) with **No Corporate CLA** and no copyright assignment friction. Every commit MUST include a `Signed-off-by:` trailer (`git commit -s`).
2. **Permanent Apache-2.0 Commitment:** `libvertirta_flight_core`, `libvertirta_rta_companion`, the append-only 256-byte `.vrta` (`VRTA0001`) binary flight-recorder specification/parsers, `VertiRTA-SDK` (`pip install vertirta`), and single-operator `VertiGCS` (including the static GitHub Pages WASM sandbox) remain **Permanent Apache-2.0**.
3. **Zero-ShareAlike / Zero-NC License & CVE Firewall (`DR-01..DR-13` Hard Gate):**
   - Training datasets, pretrained teacher/student weights, and runtime libraries MUST be permissive (`Apache-2.0`, `MIT`, `BSD`, `ISC`, `etalab-2.0`, `CDLA-Permissive-1.0`, `CC BY 4.0`, `DL-DE-BY-2.0`, `CC0`, Public Domain).
   - **Hard-Banned in Runtime, Weights, or Bundled Training Sets:**
     - *Licenses:* `GPL` / `LGPL` / `AGPL`, `CC BY-SA`, `CC BY-NC*`, `ODbL` (OSM in training).
     - *VIO / SLAM / Vision:* `OpenVINS`, `VINS-Fusion`, `HybVIO`, `ORB-SLAM3`, `sqrtVINS`, `BlenderProc` (`bpy` GPL policy in runtime/training; offline `viz/blender/` & `tools/cfd_offline_gpl/` air-gapped with `ldd`/`nm` zero-GPL runtime gate), `UAVid`, `RescueNet`, `LoveDA`, `DroneDeploy`, `FlyAwareV2`, `DINOv3`, `DINOv2-SALAD`, `SegFormer`, `Depth-Anything-V2-Base/Large/Giant`, `Sample4Geo`, `EfficientLoFTR`, `Ultralytics`, `SuperPoint`/`SuperGlue`, `Deep Hough Transform`, `InsPLAD`, `DDOS`, `Drone-vs-Bird`.
     - *LLM / Voice ASR & TTS:* `CohereForAI/aya-expanse-8b`, `OHF-Voice/piper1-gpl` (`piper-tts >= 1.7.0`), **all prebuilt `tr_TR` Piper voices** (`dfki`, `fahrettin`, `fettah`), `en_US-lessac`/`ryan`/`hfc`/`l2arctic`, `mms-tts-*`, `XTTS-v2`, `OmniVoice`, `UWB-ATCC`, `ATCO2`, `ESC-50`, `DEMAND`, unfiltered `FSD50K`.
   - **Required Clean Stack & Security CVE Pins:** **`libsodium >= 1.0.21`** (Ed25519 subgroup fix), **`Fast DDS >= 2.14.6` / `>= 3.4.1`** (`CVE-2024-28231`/`2025-62799`/`2026-22590`), **`sherpa-onnx >= 2.0.0`** (PR `#4055` zero-`espeak-ng` GPL), **`Basalt`** (`BSD-3` VIO), **`XFeat + LighterGlue`** (`Apache-2.0`), **`MegaLoc`** (`MIT`), and **`TTPLA`** (`Apache-2.0`, decoupled `45°` forward wire head ONLY).
   - **Eval-Only Quarantine:** `AeroScapes`, `ICG_Semantic_Drone`, and `UAVScenes` MUST remain strictly quarantined to evaluation scripts (`ml/slz_perception/data/license_gate.py`).

---

## 2. Hard Architectural & Determinism Invariants

Pull requests violating any invariant below will be rejected by CI:

1. **Zero-Heap Bare-Metal Flight Core (`core/flight_core/`):**
   - Freestanding C++17 (`-fno-exceptions -fno-rtti -fno-threadsafe-statics`).
   - **Zero dynamic allocation:** Must pass `-Wl,--wrap=malloc/calloc/realloc` + `no_heap_link_trap.cpp` and `scripts/ci/check_no_heap_symbols.py` (`0` forbidden symbols on x86_64 and ARM Cortex-M7).
   - **Bounded static stack:** `-fno-lto -fstack-usage -fcallgraph-info=su,da -Wstack-usage=4096 -Wvla` (`max_frame <= 4,096 B`, `max_dag <= 16,384 B`, zero recursion, zero unbounded dynamic frames).
   - **Cortex-M7 Footprint:** `< 256 KiB` Flash (`.text + .data`), `< 128 KiB` RAM (`.data + .bss`).
2. **Strict IEEE-754 Floating-Point Reproducibility:**
   - All deterministic control/RTA/STL code MUST compile with `-ffp-contract=off -fno-fast-math -fno-associative-math -fno-reciprocal-math -frounding-math`.
3. **`ASTM F3269-21` Safety Isolation (`scripts/ci/check_arch_isolation.py`):**
   - `core/flight_core/` and `core/rta_companion/` MUST have **0** includes or CMake links to ROS 2, Python, PyTorch, ONNX Runtime, `llama.cpp`, Z3, or PROJ.
   - AI/perception/co-pilot modules (`core/copilot/`, `core/slz_bayes/`, `ros2_ws/src/vertirta_{mission,perception}_node/`) remain strictly `DAL E` advisory with **0** closed-loop actuation authority and lock-free SPSC isolation.
4. **Honest Metrics & Zero Fabrication:**
   - Never hand-edit numbers in `BENCHMARKS.md` or `docs/paper/generated/tables.tex` — regenerate them from live JSON artifacts and run `python -m docs.paper.scripts.verify_paper_numbers`.
   - Unexecuted cloud/GPU/SITL benchmarks MUST stay explicitly tagged `PENDING_*` (never `[TBD]` or extrapolated estimates).
5. **Line Endings & Hygiene:**
   - Pure UTF-8 (no BOM) and **LF (`\n`)** line endings across all source, YAML, Markdown, JSON, and `.sh` files.
   - Zero personal paths, secrets, API keys, or `.agents/` files (`scripts/ci/check_public_hygiene.py`).

---

## 3. Mandatory Pre-PR Verification Commands

### C++17 / C++20 & Bare-Metal Cortex-M7 (Linux / WSL2)
```bash
# 1. Host Debug, Release, and ASan/UBSan builds & CTest
cmake --preset dev-linux-x64 && cmake --build --preset dev-linux-x64 && ctest --preset dev-linux-x64 --output-on-failure
cmake --preset release-linux-x64 && cmake --build --preset release-linux-x64 && ctest --preset release-linux-x64 --output-on-failure
cmake --preset asan-ubsan && cmake --build --preset asan-ubsan && ctest --preset asan-ubsan --output-on-failure

# 2. Bare-metal ARM Cortex-M7 footprint & zero-heap check
cmake --preset arm-cortex-m7 && cmake --build --preset arm-cortex-m7
python3 scripts/ci/check_no_heap_symbols.py --archive build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a --nm-bin arm-none-eabi-nm
python3 scripts/ci/check_bare_metal_size.py --target build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a --size-bin arm-none-eabi-size

# 3. Architecture isolation & public-release hygiene
python3 scripts/ci/check_arch_isolation.py --repo-root .
python3 scripts/ci/check_public_hygiene.py --repo-root .
```

### Python Verification, Assurance, Co-Pilot, SLZ & Paper Suites
Use isolated `uv run` environments (avoid global `pip install`):
```bash
PYTHONPATH=. uv run --python 3.11 --with pytest --with pyyaml --with scipy --with matplotlib -- \
  python -m pytest tests/verification/ tests/determinism/ docs/paper/tests/ -v

PYTHONPATH=. uv run --python 3.11 --with pytest --with z3-solver --with pyyaml --with numpy -- \
  python -m pytest tests/python/ tests/copilot/test_z3_smt2_crosscheck.py -v

PYTHONPATH=. uv run --python 3.11 --with pytest --with numpy --with scipy --with pyyaml --with torch --with onnx --with onnxscript -- \
  python -m pytest ml/slz_perception/tests/ -v
```

### Pull Request Checklist
- [ ] Commits signed off (`git commit -s`, Apache-2.0 DCO).
- [ ] All C++ presets (`dev-linux-x64`, `release-linux-x64`, `asan-ubsan`, `arm-cortex-m7`) pass with `-Werror`.
- [ ] `check_arch_isolation.py`, `check_public_hygiene.py`, and `verify_paper_numbers.py` pass with `0` violations.
- [ ] Relevant `pytest` suites pass with zero skipped security/license gates.
- [ ] No large binary blobs (`.vrta` dumps, `.onnx`, `.gguf`, raw DEMs) committed outside approved golden fixtures.
