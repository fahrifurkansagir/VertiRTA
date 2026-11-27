# VertiRTA Rust Pilot — MAVLink v2 Offboard RTA Shield (`rust/vertirta-mavlink-shield/`)

Experimental side-by-side **Apache-2.0, zero-GPL, `#![forbid(unsafe_code)]`, zero-external-dependency** Rust implementation of VertiRTA's MAVLink v2 Offboard Run-Time Assurance (RTA) Shield, built to evaluate Option A ("Strangler-Fig Hybrid Rust") against the canonical C++20 reference in [`tools/mavlink_rta_bridge/`](../../tools/mavlink_rta_bridge/README.md).

## What This Pilot Proves

- **Zero External Dependencies (`[dependencies]` empty):** Zero GPL, zero crates.io supply-chain surface, offline builds on Windows & Linux.
- **`#![forbid(unsafe_code)]`:** Eliminates raw pointer arithmetic, `memcpy` wire framing, and POSIX/Winsock `#ifdef` divergence across the entire MAVLink v2 codec and UDP bridge.
- **Compile-Time `FiniteF64` / `Vec3Ned` Newtypes (`src/finite.rs`):** Enforces finiteness at deserialization boundaries so `NaN` / `±Inf` cannot propagate into Istanbul geofence or speed/altitude clamps.
- **Exact Wire & Safety Parity:** Implements `0xFD` MAVLink v2 framing, X.25 `CRC-16/MCRF4XX` with all 8 `common.xml` `CRC_EXTRA` seeds, `MAV_FRAME_LOCAL_NED` / `FORCE_SET` fail-closed guards, Istanbul `LTFJ` / `LTBX` / Kartal Hospital NFZ projection, airborne/non-finite disarm denial with `COMMAND_ACK (DENIED)`, and 500 ms deadman hover.

## Build, Lint & Live 2,500-Packet UDP Proof

```bash
cd rust/vertirta-mavlink-shield
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo run --release -- --packets 2500
```
