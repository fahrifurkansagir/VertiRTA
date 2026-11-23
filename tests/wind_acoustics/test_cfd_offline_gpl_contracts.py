# SPDX-License-Identifier: Apache-2.0
"""Contract tests for air-gapped GPL-3.0 OpenFOAM coastal wake generator & VRTAWND1 packer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.cfd_offline_gpl.generate_ci_fixture_vrtawnd1 import generate_fixture_vrtawnd1
from tools.cfd_offline_gpl.openfoam_case_gen import (
    generate_coastal_openfoam_case,
    roughness_z0_for_direction,
)
from tools.cfd_offline_gpl.pod_rbf_compressor import morton_encode_4x4x4

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_openfoam_airgap_rejects_ltfj_breach_and_avoids_gl3226_gl3236(tmp_path: Path) -> None:
    # Must reject DR-09 ISTMarina (29.2057 E, +1.14 km inside LTFJ CTR!)
    with pytest.raises(ValueError, match="29.1895"):
        generate_coastal_openfoam_case(tmp_path / "bad_istmarina", slz_lat_deg=40.8846, slz_lon_deg=29.2057)

    # Also reject a valid SLZ whose eastern mesh extent crosses 29.1895 E
    with pytest.raises(ValueError, match="29.1895"):
        generate_coastal_openfoam_case(
            tmp_path / "bad_east_extent",
            slz_lat_deg=40.8860,
            slz_lon_deg=29.1880,
            east_extent_m=500.0,
        )

    case_dir = generate_coastal_openfoam_case(
        tmp_path / "dragos_kartal",
        slz_lat_deg=40.8860,
        slz_lon_deg=29.1880,
        wind_from_deg=225.0,
        mode="rans",
    )
    u_bc = (case_dir / "0" / "U").read_text(encoding="utf-8")
    assert "slip" in u_bc and "symmetryPlane" not in u_bc  # OpenFOAM GL#3226 lock
    assert "atmBoundaryLayerInletVelocity" in u_bc
    nut_bc = (case_dir / "0" / "nut").read_text(encoding="utf-8")
    assert "atmNutkWallFunction" in nut_bc and "symmetryPlane" not in nut_bc

    # Onshore SW Lodos sea fetch vs offshore NE Poyraz urban roughness
    assert roughness_z0_for_direction(225.0) == pytest.approx(0.002)
    assert roughness_z0_for_direction(45.0) == pytest.approx(0.50)

    # IDDES mode emits pimpleFoam + kOmegaSSTIDDES
    iddes_dir = generate_coastal_openfoam_case(
        tmp_path / "dragos_iddes",
        slz_lat_deg=40.8860,
        slz_lon_deg=29.1880,
        wind_from_deg=45.0,
        mode="iddes",
    )
    assert "pimpleFoam" in (iddes_dir / "system" / "controlDict").read_text(encoding="utf-8")
    assert "kOmegaSSTIDDES" in (iddes_dir / "constant" / "turbulenceProperties").read_text(
        encoding="utf-8"
    )

    apptainer = (REPO_ROOT / "tools" / "cfd_offline_gpl" / "openfoam_apptainer.def").read_text(
        encoding="utf-8"
    )
    assert "openfoam-dev" in apptainer and "dl.openfoam.com/add-debian-repo" not in apptainer
    assert (REPO_ROOT / "tools" / "cfd_offline_gpl" / "LICENSE.GPL3").is_file()
    assert (REPO_ROOT / "tools" / "cfd_offline_gpl" / "README_GPL_AIRGAP.md").is_file()


def test_pod_rbf_compressor_emits_valid_sha256_morton_vrtawnd1_fixture(tmp_path: Path) -> None:
    # Verify 4x4x4 Morton encoder is a bijection onto [0..63]
    seen = {
        morton_encode_4x4x4(lx, ly, lz)
        for lx in range(4)
        for ly in range(4)
        for lz in range(4)
    }
    assert seen == set(range(64))

    out_bin = tmp_path / "ci_coastal_fixture.vrtawnd1"
    meta = generate_fixture_vrtawnd1(out_bin)
    raw = out_bin.read_bytes()
    assert raw[:8] == b"VRTAWND1"
    assert meta["header_bytes"] == 12288
    assert meta["node_bytes"] == 64
    assert meta["total_bytes"] == len(raw) and len(raw) < 80 * 1024
    assert meta["full_scale_size_bytes"] == 14561280  # 13.887 MiB < 16 MiB
    assert meta["max_lon_deg"] < 29.1895
    assert meta["eps_rom_p95_max_mps"] > 0.0 and len(meta["payload_sha256"]) == 64

    # Optional Python -> C++ ABI parity check when VERTIRTA_VERIFY_VRTAWND1_BIN is provided
    verifier_bin = os.environ.get("VERTIRTA_VERIFY_VRTAWND1_BIN", "").strip()
    if verifier_bin:
        proc = subprocess.run(
            [verifier_bin, str(out_bin)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"C++ verify_vrtawnd1_load failed:\n{proc.stdout}\n{proc.stderr}"
        assert "VRTAWND1_OK" in proc.stdout
