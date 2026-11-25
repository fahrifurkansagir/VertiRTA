## Summary

- **Linked Issue / Proposal:** #
- **What changed and why:**

## Verification Proof (Required — PRs without test evidence may be closed)

- [ ] Commits are signed off under **DCO 1.1** (`git commit -s`, Apache-2.0, no CLA).
- [ ] All changes are personally reviewed and tested (unreviewed AI-generated bulk PRs or unsolicited refactors are not accepted).
- [ ] `core/flight_core/` bare-metal constraints remain intact (`0` heap, bounded static stack, `< 256 KiB` Cortex-M7 Flash) or untouched.
- [ ] Zero GPL/LGPL/AGPL, CC BY-NC, or CC BY-SA dependencies or training datasets introduced (`check_zero_gpl_symbols.py`).
- [ ] Local checks passed:
  - [ ] `ctest --test-dir build/dev-linux-x64 --output-on-failure`
  - [ ] `python3 scripts/ci/check_arch_isolation.py --repo-root .`
  - [ ] `python3 scripts/ci/check_public_hygiene.py --repo-root .`
- [ ] Pure UTF-8 (no BOM), LF line endings, no large binaries (`>= 1 MB`) or `.mp4` files committed.
- [ ] No airworthiness certification or live-flight approval claims added.
