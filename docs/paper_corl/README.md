# Paper 2 — CoRL 2027 Double-Anonymous Manuscript Package

This folder holds the double-anonymous 8-page Conference on Robot Learning (CoRL 2027) manuscript (`vertirta_corl_manuscript.tex`), its manifest (`corl_metrics_manifest.json`), and its deterministic table/figure/verification scripts.

## Scientific Scope & Non-Certification Boundary
- **Advisory Layer Only:** Evaluates bilingual Turkish/English Z3-GRPO mission synthesis (`VertiBench-v2`), language- and cockpit-SNR-stratified Mondrian split conformal gating (`alpha = 0.05`), and zero-ShareAlike active visual Safe Landing Zone (SLZ) real-imagery transfer (`AeroScapes`, `ICG Drone`, `UAVScenes`, zero `UAVid`).
- **Neutral Deterministic RTA Cage:** Treats the underlying bare-metal HJI-CBVF-QP + INDI-WLS safety filter as an external deterministic verifier without revealing repository or author identity.
- **Honest Provenance & Phase-2 Anti-Fabrication:**
  - `VertiBench-v2` is tagged `PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2`.
  - Phase-3 `G4` batch slots remain `PENDING_PHASE3_G4_THUNDER_RUN` (`\text{--}`) until ingested via `--ingest-v400 artifacts/v4.0.0`.
  - Unrun external real-imagery datasets (`ICG Drone`, `UAVScenes`) remain `PENDING_EVAL_ONLY_REAL_IMAGERY` (`\text{--}`).
  - Research and software-verification testbed ONLY; no airworthiness certification claim.

## Reproducibility Commands
```bash
uv run --no-project --python 3.11 -- python docs/paper_corl/scripts/generate_corl_tables.py
uv run --no-project --python 3.11 --with matplotlib --with numpy --with scipy -- python docs/paper_corl/scripts/generate_corl_figures.py
uv run --no-project --python 3.11 -- python docs/paper_corl/scripts/verify_corl_numbers.py --allow-pending-g4
```

After Phase-3 `G4` artifacts are generated under `artifacts/v4.0.0/`:
```bash
uv run --no-project --python 3.11 -- python docs/paper_corl/scripts/generate_corl_tables.py --ingest-v400 artifacts/v4.0.0
uv run --no-project --python 3.11 --with matplotlib --with numpy --with scipy -- python docs/paper_corl/scripts/generate_corl_figures.py
uv run --no-project --python 3.11 -- python docs/paper_corl/scripts/verify_corl_numbers.py
```
