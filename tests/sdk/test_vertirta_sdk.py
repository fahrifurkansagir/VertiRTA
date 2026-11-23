"""Unit & integration tests for VertiRTA nanobind Python SDK, Gymnasium v1.1+, Typer CLI & cibuildwheel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tomllib

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import numpy as np
import pytest
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]
SDK_DIR = REPO / "sdk" / "python"
# Append (never prepend!) source fallback so an installed compiled wheel in site-packages wins.
if str(SDK_DIR) not in sys.path:
    sys.path.append(str(SDK_DIR))

import vertirta  # noqa: E402
from vertirta import VertiFleetSim, VertiLiftCruiseSim  # noqa: E402
from vertirta.cli import app  # noqa: E402
from vertirta.env import VertiRtaCoaxialX8Env, crc32c_py, make_vrta_header_bytes, verify_vrta_frame_bytes  # noqa: E402
from vertirta.vec_env import VertiVecEnv  # noqa: E402


def test_pyproject_locks_nanobind_abi3_and_skips_free_threaded_cp314t() -> None:
    pyproject = REPO / "pyproject.toml"
    raw = pyproject.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    cfg = tomllib.loads(raw.decode("utf-8"))

    reqs = " ".join(cfg["build-system"]["requires"])
    assert "scikit-build-core" in reqs
    assert "nanobind" in reqs
    assert cfg["tool"]["scikit-build"]["wheel"]["py-api"] == "cp312"

    cibw = cfg["tool"]["cibuildwheel"]
    assert "cp31?t-*" in cibw["skip"]
    assert cibw["build-frontend"] == "uv"


def test_gymnasium_single_and_vec_env_determinism_and_autoreset_modes() -> None:
    if os.environ.get("VERTIRTA_REQUIRE_NATIVE") == "1":
        assert vertirta.get_backend_name() == "cpp_nanobind", (
            f"VERTIRTA_REQUIRE_NATIVE=1 expected compiled cpp_nanobind, got {vertirta.get_backend_name()!r} "
            f"(loaded from {vertirta.__file__})"
        )

    # Registered ID check
    env_reg = gym.make("VertiRTA-CoaxialX8-v0")
    obs_reg, info_reg = env_reg.reset(seed=123)
    assert obs_reg.shape == (18,) and obs_reg.dtype == np.float32
    assert info_reg["backend"] in ("cpp_nanobind", "python_fallback")
    env_reg.close()

    # Exact deterministic seeding & Gymnasium v1.1+ reset(seed=None) non-reset contract
    env1 = VertiRtaCoaxialX8Env(fault_mode="none")
    env2 = VertiRtaCoaxialX8Env(fault_mode="none")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)

    obs_a, _ = env1.reset(seed=None)
    obs_b, _ = env1.reset(seed=None)
    assert obs_a.shape == obs1.shape
    assert not np.array_equal(obs_a, obs_b), "reset(seed=None) must advance RNG stream, not freeze to constant"

    # Fault-injection CEL check + VRTA0001 CRC-32C frame integrity
    env_fault = VertiRtaCoaxialX8Env(fault_mode="same_arm_dual", substeps_400hz=4, max_steps=200)
    env_fault.reset(seed=9, options={"inject_immediately": True})
    cel_seen = False
    for _ in range(30):
        _, _, term, _, step_info = env_fault.step(np.zeros((4,), dtype=np.float32))
        if step_info["cel_latched"]:
            cel_seen = True
            break
        if term:
            break
    assert cel_seen is True, "Same-arm dual loss must trigger CEL active-spin latch"
    frame_bytes = env_fault.latest_vrta_frame_bytes()
    assert len(frame_bytes) == 256 and verify_vrta_frame_bytes(frame_bytes)
    assert crc32c_py(b"123456789") == 0xE3069283
    assert len(make_vrta_header_bytes()) == 128

    # Vectorized env: test all 3 Gymnasium v1.1+ AutoresetModes
    for mode in (AutoresetMode.SAME_STEP, AutoresetMode.NEXT_STEP, AutoresetMode.DISABLED):
        vec = VertiVecEnv(num_envs=4, autoreset_mode=mode, max_steps=5)
        assert vec.metadata["autoreset_mode"] == mode
        vobs, vinfo = vec.reset(seed=7)
        assert vobs.shape == (4, env1.observation_space.shape[0])
        assert vinfo["backend"] == vertirta.get_backend_name()
        next_obs, rew, term, trunc, info = vec.step(np.zeros((4, 4), dtype=np.float32))
        assert next_obs.shape == vobs.shape and rew.shape == (4,)
        assert term.shape == (4,) and trunc.shape == (4,)
        assert "backend" in info
        vec.close()

    # Verify reset_mask in DISABLED mode selectively resets only masked indices
    vec_dis = VertiVecEnv(num_envs=4, autoreset_mode=AutoresetMode.DISABLED, max_steps=3)
    o0, _ = vec_dis.reset(seed=11)
    for _ in range(3):
        o_end, _, _, trunc_end, _ = vec_dis.step(np.zeros((4, 4), dtype=np.float32))
    assert bool(trunc_end.all())
    o_partial, _ = vec_dis.reset(seed=99, options={"reset_mask": [True, False, True, False]})
    np.testing.assert_array_equal(o_partial[1], o_end[1])
    np.testing.assert_array_equal(o_partial[3], o_end[3])
    assert not np.array_equal(o_partial[0], o_end[0])
    vec_dis.close()


def test_vertirta_cli_subcommands_and_port_guard(tmp_path: Path) -> None:
    runner = CliRunner()
    out_vrta = tmp_path / "smoke.vrta"

    res = runner.invoke(app, ["sim", "--airframe", "x8", "--duration", "0.5", "--out", str(out_vrta)])
    assert res.exit_code == 0, res.stdout
    assert out_vrta.exists() and out_vrta.stat().st_size == 128 + 50 * 256

    res_insp = runner.invoke(app, ["inspect", str(out_vrta), "--summary"])
    assert res_insp.exit_code == 0 and "VRTA0001" in res_insp.stdout

    res_rep = runner.invoke(app, ["replay", str(out_vrta)])
    assert res_rep.exit_code == 0
    rep_data = json.loads(res_rep.stdout)
    assert rep_data["frames"] == 50 and rep_data["bad_crc_frames"] == 0

    res_ver = runner.invoke(app, ["verify", str(out_vrta)])
    assert res_ver.exit_code == 0
    assert json.loads(res_ver.stdout)["passed"] is True

    res_stl = runner.invoke(app, ["verify-stl", str(out_vrta)])
    assert res_stl.exit_code == 0
    assert json.loads(res_stl.stdout)["satisfied"] is True

    res_bench = runner.invoke(app, ["bench", "--num-envs", "4", "--steps", "10"])
    assert res_bench.exit_code == 0
    assert json.loads(res_bench.stdout)["env_steps_per_sec"] > 0.0

    # Port isolation guard (5180/8891 allowed; 4173/4174/8765/9222/9223 rejected)
    res_ok_port = runner.invoke(app, ["gcs", "--port", "5180", "--ws-port", "8891", "--dry-run"])
    assert res_ok_port.exit_code == 0
    for bad_port in ("4173", "4174", "8765", "9222", "9223"):
        res_bad = runner.invoke(app, ["gcs", "--port", bad_port, "--dry-run"])
        assert res_bad.exit_code != 0

    # Zero out-of-ODD Bosphorus examples in CLI help
    res_help = runner.invoke(app, ["verify-stl", "--help"])
    assert res_help.exit_code == 0
    assert "Bo\u011faz" not in res_help.stdout and "Bosphorus" not in res_help.stdout
    assert "Kartal" in res_help.stdout


def test_m2_fleet_sim_and_lift_cruise_sdk_bindings() -> None:
    fleet = VertiFleetSim(n_agents=20, threads=2)
    lc8 = VertiLiftCruiseSim(layout="8+1")
    lc4 = VertiLiftCruiseSim(layout="4+1")

    if os.environ.get("VERTIRTA_REQUIRE_NATIVE") == "1":
        assert fleet.backend == "cpp_nanobind"
        assert lc8.backend == "cpp_nanobind"

    assert fleet.agent_count == 20
    fleet.reset(seed=20261123)
    fleet.inject_same_arm_cel(agent_idx=0, arm_idx=0)
    summary = fleet.step_100hz()
    assert summary["active_agents"] == 20
    assert summary["airspace_breaches"] == 0
    assert summary["max_rta_hazard_rows"] <= 14
    assert fleet.agent_cel_latched(0) is True

    # Part-IS spoofed USSP into LTFJ CTR (>= 29.1895 E) -> quarantine -> <= 10 s autonomous RTB
    q = fleet.check_ussp_geofence(sig_valid=True, cmd_lon_deg=29.2100, cmd_agl_m=80.0, t_s=10.0)
    assert q["cyber_state"] == "SEC_USSP_QUARANTINE"
    assert q["drop_ussp_directives"] is True
    assert q["emit_is_or_230_draft"] is True
    rtb = fleet.step_cyber_time(t_s=20.0, ussp_auth_restored=False)
    assert rtb["cyber_state"] == "SEC_AUTONOMOUS_RTB"
    assert rtb["divert_to_nearest_fato_emg"] is True

    # Lift+Cruise 8+1 hover OEI <= 79.2% T_max vs 4+1 rank-3 hover -> rank-4 transition
    assert lc8.actuator_count == 12 and lc8.lift_rotor_count == 8
    assert lc8.cruise_airspeed_mps == pytest.approx(38.0)
    oei8 = lc8.audit_oei_trim(0.0, failed_lift_rotor=0)
    assert oei8["trim_feasible_4dof"] is True
    assert oei8["peak_thrust_ratio"] <= 0.793

    assert lc4.actuator_count == 8 and lc4.lift_rotor_count == 4
    oei4_hover = lc4.audit_oei_trim(0.0, failed_lift_rotor=0)
    assert oei4_hover["trim_feasible_4dof"] is False
    assert oei4_hover["effective_rank"] == 3
    oei4_trans = lc4.audit_oei_trim(14.0, failed_lift_rotor=0)
    assert oei4_trans["trim_feasible_4dof"] is True
