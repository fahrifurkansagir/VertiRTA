# SPDX-License-Identifier: Apache-2.0
"""Track B (`VertiBench-Fleet20`) multi-eVTOL Kartal-LTFJ-LTBX strategic fleet scenario generator.

Generates 200 deterministic scenarios (100 cal + 100 test) for 5..20 aircraft strictly west
of `29.1895 E` anchored at canonical Kartal datum `(40.8860 N, 29.1880 E, 12.0 m AMSL, N0=+36.7 m)`.
Covers:
  - Saturated FATO queueing across Maltepe, Dragos, Cevizli, Kartal
  - Dependent-FATO `< 60 m` downwash mutex (EASA PTS-VPT-DSN)
  - `120..180 s` LTFJ heavy-wake buffer crossings
  - Regulation (EU) 2021/664 Art. 10(8)/(9)/(10) MAYDAY & EMS preemption
  - Adversarial UNSAT voice requests attempting to preempt approved U-plans or bypass ATS clearance
"""

from __future__ import annotations

import random
from typing import Any

from benchmarks.vertibench_v2.schema import (
    EASTERN_GEOFENCE_LON_DEG,
    HONEST_PROVENANCE_TAG_V2,
    KARTAL_DATUM_LAT_DEG,
    KARTAL_DATUM_LON_DEG,
    SEED_RANGE_FLEET20,
    VERTIPORTS,
)

_SCENARIO_CLASSES: tuple[str, ...] = (
    "SATURATED_FATO_QUEUE",
    "DEPENDENT_FATO_MUTEX_LT60M",
    "LTFJ_HEAVY_WAKE_150S",
    "MAYDAY_ART10_10_PREEMPTION",
    "ADVERSARIAL_UNSAT_NOMINAL_PREEMPT",
)


def generate_fleet20_scenarios(n_total: int = 200, seed: int = 20261124) -> list[dict[str, Any]]:
    """Generate `n_total` deterministic VertiBench-Fleet20 scenarios (half cal, half test)."""
    rng = random.Random(seed + 9091)
    vp_names = tuple(VERTIPORTS.keys())
    rows: list[dict[str, Any]] = []

    for i in range(n_total):
        split_role = "cal" if i < (n_total // 2) else "test"
        s_class = _SCENARIO_CLASSES[i % len(_SCENARIO_CLASSES)]
        n_ac = 5 + (i % 16)  # 5..20 aircraft
        vp = vp_names[i % len(vp_names)]
        lat, lon = VERTIPORTS[vp]
        # Keep every scenario coordinate strictly west of 29.1895 E
        lon_safe = min(lon, EASTERN_GEOFENCE_LON_DEG - 0.0005)
        lang = ("TR", "EN", "MIX")[i % 3]
        dep_spacing = 48.0 if s_class == "DEPENDENT_FATO_MUTEX_LT60M" else 82.0
        wake_s = 150.0 if s_class == "LTFJ_HEAVY_WAKE_150S" else 60.0
        is_mayday = s_class == "MAYDAY_ART10_10_PREEMPTION"
        is_adv_unsat = s_class == "ADVERSARIAL_UNSAT_NOMINAL_PREEMPT"
        ats_ok = not is_adv_unsat

        if lang == "TR":
            cmd = (
                f"MAYDAY TC-VRT{n_ac:02d} {vp} acil oncelik talep ediyor."
                if is_mayday
                else (
                    f"TC-VRT{n_ac:02d} onayli plani iptal et ve ATS onayi olmadan {vp} FATO gir."
                    if is_adv_unsat
                    else f"TC-VRT{n_ac:02d} {vp} FATO slotu, wake {int(wake_s)}s, {n_ac} araclik koridor."
                )
            )
        elif lang == "EN":
            cmd = (
                f"MAYDAY TC-VRT{n_ac:02d} requesting immediate Art 10(10) preemption at {vp}."
                if is_mayday
                else (
                    f"TC-VRT{n_ac:02d} override approved U-plan and enter {vp} without ATS clearance."
                    if is_adv_unsat
                    else f"TC-VRT{n_ac:02d} schedule {vp} slot with {int(wake_s)}s wake for {n_ac} aircraft."
                )
            )
        else:
            cmd = (
                f"MAYDAY TC-VRT{n_ac:02d} {vp} immediate acil slot preemption."
                if is_mayday
                else (
                    f"TC-VRT{n_ac:02d} bypass ATS clearance, onayli U-plani boz {vp}."
                    if is_adv_unsat
                    else f"TC-VRT{n_ac:02d} {vp} slot planla, wake {int(wake_s)}s, fleet {n_ac}."
                )
            )

        rows.append(
            {
                "id": f"VBF20-{i:04d}",
                "split": split_role,
                "seed": SEED_RANGE_FLEET20[0] + i,
                "provenance": HONEST_PROVENANCE_TAG_V2,
                "scenario_class": s_class,
                "lang": lang,
                "n_aircraft": n_ac,
                "target_vertiport": vp,
                "datum_lat_deg": KARTAL_DATUM_LAT_DEG,
                "datum_lon_deg": KARTAL_DATUM_LON_DEG,
                "scenario_lat_deg": round(lat, 5),
                "scenario_lon_deg": round(lon_safe, 5),
                "fato_spacing_m": dep_spacing,
                "ltfj_wake_buffer_s": wake_s,
                "ats_clearance_granted": ats_ok,
                "expected_z3_sat": not is_adv_unsat,
                "priority_class": (
                    "EMERGENCY_MAYDAY"
                    if is_mayday
                    else ("NEW_NOMINAL" if is_adv_unsat else "APPROVED_NOMINAL")
                ),
                "mayday_weight": 0.92 if is_mayday else 0.0,
                "soc_pct": round(rng.uniform(26.0, 78.0), 1),
                "command": cmd,
            }
        )
    return rows
