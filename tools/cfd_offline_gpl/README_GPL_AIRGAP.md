# Air-Gapped Offline GPL-3.0 OpenFOAM Coastal Wake & POD-RBF Toolchain

> **Disclaimer:** Research simulation pipeline only — not certified for real-world aeronautical navigation or operational wind-hazard forecasting (`EASA SC-VTOL`, `DO-178C`, or `DO-278A`).

## 1. Strict License Air-Gap (FSF GPLv3 Section 0)

- Everything inside `tools/cfd_offline_gpl/` is licensed under **`GPL-3.0-or-later`** (see `LICENSE.GPL3`).
- **Zero compile/link boundary:** Neither root `CMakeLists.txt` nor any Apache-2.0 library under `core/` ever `#include`s, links, or invokes OpenFOAM (`libOpenFOAM`, `libfiniteVolume`, `fvCFD.H`, etc.).
- **Pure numeric data bridge:** Offline CFD snapshots are compressed into pure numeric `VRTAWND1` binary lookup tables (`12,288 B` header + FIPS 180-4 SHA-256 + `64 B` Morton-swizzled `int16` POD-RBF bricks). Under FSF GPLv3 Section 0, pure numerical physical field tables are not derivative works of the solver executable.
- **Automated CI enforcement:** `scripts/ci/check_zero_gpl_symbols.py` runs `ldd`, `nm`, and C++ header scans across all Apache-2.0 targets to guarantee zero GPL symbol leakage.

## 2. Mandatory Istanbul Geodesy & OpenFOAM Errata Locks

1. **LTFJ CTR Hard Exclusion (`lon_deg < 29.1895° E`):** Anchored strictly to `kKartalDatum` (`40.8860° N, 29.1880° E, 12.0 m AMSL, EGM2008 N = +36.7 m`). Any domain vertex or SLZ nest at `lon_deg >= 29.1895° E` (including the erroneous `29.2057° E` İSTMarina coordinate from `DR-09`) is hard-rejected with `ValueError`.
2. **OpenFOAM `GL#3226` Lock:** Lateral and top domain boundaries strictly use `slip` (never `symmetryPlane` on curved/inconsistent boundary patches).
3. **OpenFOAM `GL#3236` Lock:** Offline containerization pins `openfoam-dev:apptainer` via `openfoam_apptainer.def` (never `dl.openfoam.com/add-debian-repo` apt repositories, and OpenFOAM is **never** installed or executed in GitHub Actions CI).
4. **Zero Multi-MB Blobs in Git:** Full-scale `VRTAWND1` tables (`14,561,280 B` = `13.887 MiB < 16 MiB`) are distributed via GitHub Releases / local scratch; CI runs `generate_ci_fixture_vrtawnd1.py` into temporary paths (`< 80 KiB`) in `< 1 s`.
