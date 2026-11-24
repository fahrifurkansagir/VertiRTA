# SPDX-License-Identifier: Apache-2.0
"""Contract and license-firewall tests for ROS 2 vertirta_voice_node (M3 Plan 6B Task 5)."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

REPO = Path(__file__).resolve().parents[2]
VOICE_NODE_DIR = REPO / "ros2_ws" / "src" / "vertirta_voice_node"

BANNED_VOICE_TOKENS = (
    "piper1-gpl",
    "piper1_gpl",
    "espeak-ng",
    "espeak_ng",
    "libespeak",
    "piper_phonemize",
    "piper-phonemize",
    "xtts-v2",
    "uwb-atcc",
    "atco2",
    "esc-50",
    "demand",
    "tr_tr-dfki",
    "tr_tr-fahrettin",
    "tr_tr-fettah",
    "mms-tts",
)


def test_voice_node_package_yaml_and_cmake_are_utf8_lf_and_zero_gpl() -> None:
    expected_files = [
        VOICE_NODE_DIR / "CMakeLists.txt",
        VOICE_NODE_DIR / "package.xml",
        VOICE_NODE_DIR / "config" / "voice_node.yaml",
        VOICE_NODE_DIR / "include" / "vertirta_voice_node" / "voice_node_core.hpp",
        VOICE_NODE_DIR / "src" / "voice_node_core.cpp",
        VOICE_NODE_DIR / "src" / "voice_node.cpp",
        VOICE_NODE_DIR / "test" / "CMakeLists.txt",
        VOICE_NODE_DIR / "test" / "test_voice_node_core.cpp",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing required vertirta_voice_node file: {path}"
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM forbidden: {path}"
        assert b"\r" not in raw, f"CRLF forbidden: {path}"
        text_lower = raw.decode("utf-8").lower()
        for banned in BANNED_VOICE_TOKENS:
            assert banned not in text_lower, f"Banned voice/license token {banned!r} found in {path}"

    pkg_root = ET.fromstring((VOICE_NODE_DIR / "package.xml").read_text(encoding="utf-8"))
    assert pkg_root.findtext("name") == "vertirta_voice_node"
    assert pkg_root.findtext("license") == "Apache-2.0"


def test_voice_node_yaml_locks_topics_sherpa_v2_and_canonical_snr_strata() -> None:
    cfg = yaml.safe_load((VOICE_NODE_DIR / "config" / "voice_node.yaml").read_text(encoding="utf-8"))
    params = cfg["vertirta_voice_node"]["ros__parameters"]

    assert params["topic_audio_16khz"] == "/vertirta/voice/audio_16khz"
    assert params["topic_rta_state_snapshot"] == "/vertirta/rta/state_snapshot"
    assert params["topic_dsl_command"] == "/vertirta/voice/dsl_command"
    assert params["topic_tts_readback"] == "/vertirta/voice/tts_readback"
    assert params["topic_sidecar_event"] == "/vertirta/forensics/sidecar_event"

    assert params["asr_translate"] is False
    assert params["asr_no_context"] is True
    assert params["asr_carry_initial_prompt"] is True
    assert 256 <= int(params["asr_audio_ctx"]) <= 384

    assert params["tts_engine"] == "sherpa_onnx_v2"
    assert float(params["tts_ttfa_budget_ms"]) <= 250.0
    assert params["snr_strata"] == ["S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"]
    assert float(params["mayday_weight_min"]) >= 0.15
    assert float(params["liveness_emergency_soc_pct"]) == 5.0
