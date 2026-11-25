# Security Policy

## 1. Scope & Safety-Critical Operational Disclaimer

**VertiRTA** (`libvertirta_flight_core`, `libvertirta_rta_companion`, `vertirta_copilot`, `vertirta_stl_monitor`, `vertirta_slz_bayes`, `vertirta_mavlink_rta_bridge`, and ROS 2 bridge packages) is open-source flight-control and run-time assurance software (not certified for live flight).

Within this codebase, VertiRTA explicitly models aviation cybersecurity and run-time assurance boundaries informed by **`ASTM F3269-21`** and **Commission Implementing Regulation (EU) 2023/203 (`Part-IS`)** (information security risk management for aviation and U-space service providers, applicable February 22, 2026).

---

## 2. Security Architecture & Trust Boundaries

| Boundary / Component | Trust Level | Security & Safety Guarantees Enforced in Code |
| :--- | :--- | :--- |
| **`libvertirta_flight_core` (C++17, 400 Hz)** | **Trusted Deterministic Core** | Zero dynamic heap (`nm` + `-Wl,--wrap=malloc`), bounded static stack DAG (`2,960 B` Cortex-M7), NaN/Inf guards on all IMU/ESC inputs, irreversible in-flight `LATCHED_CEL` latch. |
| **`libvertirta_rta_companion` (C++20, 100 Hz)** | **Trusted Safety Filter (`ASTM F3269`)** | FIPS 180-4 SHA-256 verification on `mmap` LUTs (`VRTALUT1`, `VRTACEL2`) rejecting tampered payloads; Castagnoli CRC-32C framing on `.vrta` headers (`128 B`) and frames (`256 B`); hard `LTFJ`/`LTBX` CTR exclusion barriers. |
| **Pillar 2 Co-Pilot (`vertirta_copilot`, `DAL E`)** | **Untrusted Advisory Complex Function** | Eager GBNF grammar clamping output to `<= 30` ASCII tokens; explicit rejection of `<think>` / `/think` prompt-injection payloads; fail-closed Z3 `QF_LRA` verifier (`rlimit=250000`, `12 ms` timeout); immutable `shield_active = true`; zero closed-loop actuation write path. |
| **Pillar 3 Visual SLZ (`vertirta_slz_bayes`, `DAL E`)** | **Untrusted Advisory Perception** | Zero-ShareAlike startup license gate; Free-Energy OOD variance floor + precision cap preventing adversarial tarp/clutter false-safe collapse; hard `> 250 ms` frame staleness rejection; isolated from 100 Hz RTA via lock-free SPSC queues. |
| **Telemetry, MAVLink v2 & U-Space (`Part-IS` Alignment)** | **External Network Interface** | Local visualization ports isolated (`:5180` CesiumJS, `:8891` Foxglove, `:8888` MicroXRCE); `Part-IS (EU) 2023/203` `40 B` `VertiRtaSecureHeader` + `ReplayWindow128` + pinned `libsodium >= 1.0.21` + `RTO <= 10 s` FSM, `ASTM F3548-21`/`F3411-22a` DRIP, `MCUSUM` GNSS anti-spoofing, and clean-room MAVLink v2 CRC-16/deadman shield treat all external network/RF inputs as untrusted until authenticated and cross-checked. |

---

## 3. Threat Surface In Scope for Vulnerability Reports

We welcome security and safety-integrity reports covering:
- **Safety-Filter / Memory Bypass:** Any input causing heap allocation, unbounded stack growth, NaN propagation, integer wraparound (`FixedMonotonicWedge`, timestamp underflow), or barrier bypass in `core/flight_core/`, `core/rta_companion/`, or `core/stl/`.
- **Binary & Serialization Parsing:** Malformed `.vrta`, `.vrta_lut`, `.vrta_cel2d`, YAML, or JSON inputs causing out-of-bounds memory reads, CRC-32C/SHA-256 verification bypass, or denial-of-service crashes.
- **Prompt / Grammar / Verifier Bypass:** Inputs bypassing GBNF grammar constraints, inducing Z3 assertion-name collisions, disabling `shield_active`, or fabricating numeric telemetry past the Numeric Grounding Gate.
- **Supply-Chain & Secret Leaks:** Accidental inclusion of non-permissive/ShareAlike datasets or models, leaked credentials, or compromised CI workflow actions.

---

## 4. Reporting a Vulnerability

**Do NOT open a public GitHub issue for undisclosed security vulnerabilities.**

1. Use **GitHub Private Vulnerability Reporting** (`Security` -> `Advisories` -> `Report a vulnerability`) on the repository, or contact the maintainers privately via the security contact listed on the repository profile.
2. Include:
   - Affected component/file path and commit SHA.
   - Minimal reproduction input (e.g., failing Catch2/pytest case, malformed `.vrta` hex snippet, or adversarial prompt/state vector).
   - Expected invariant vs. observed behavior.
3. **Response Targets:**
   - Initial acknowledgment: within **72 hours**.
   - Triage & severity classification: within **7 days**.
   - Coordinated fix & regression test (added to `tests/` before public advisory): prioritized immediately for any core RTA, memory-safety, or verifier bypass finding.

---

## 5. Automated Hygiene Enforcement

Repository hygiene is verified on every change via `scripts/ci/check_public_hygiene.py`, which enforces zero leaked personal paths, zero hardcoded API keys/tokens, zero tracked local `.agents/` rules, and pure LF endings on all shell scripts.
