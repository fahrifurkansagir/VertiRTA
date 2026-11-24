# SPDX-License-Identifier: Apache-2.0
"""Apache-2.0 Procedural 8-Rotor Coaxial BVI Cockpit Noise DSP Synthesizer.

Strictly parameterized from frozen `config/vehicle_x8_240.yaml` (01-DR08-13 Override #1):
  - 8 coaxial rotors (4 arms, coax_spacing_m = 0.35 m), B = 3 blades/rotor, R = 0.75 m (D = 1.50 m)
  - omega_hover = 151.8 rad/s (1,449.6 RPM, BPF_hover = 72.5 Hz)
  - omega_max   = 240.0 rad/s (2,291.8 RPM, BPF_max   = 114.6 Hz, ~115 Hz)
  - Operating sweep: 1,100..2,292 RPM (explicitly overriding DR-12's stale 1,850 RPM cap)

Consumed by Plan 6B (`benchmarks/vertibench_v2/`) and Plan 6C (`job_whisper_bvi_lora.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SNR_STRATA: tuple[str, ...] = ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")


@dataclass(frozen=True)
class CoaxialAcousticParams:
    rotor_count: int
    blades_per_rotor: int
    rotor_radius_m: float
    coax_spacing_m: float
    omega_hover_rad_s: float
    omega_max_rad_s: float
    rpm_min: float = 1100.0

    @property
    def rpm_hover(self) -> float:
        return self.omega_hover_rad_s * 60.0 / (2.0 * math.pi)

    @property
    def rpm_max(self) -> float:
        return self.omega_max_rad_s * 60.0 / (2.0 * math.pi)

    @property
    def bpf_hover_hz(self) -> float:
        return self.blades_per_rotor * (self.rpm_hover / 60.0)

    @property
    def bpf_max_hz(self) -> float:
        return self.blades_per_rotor * (self.rpm_max / 60.0)

    @property
    def coax_crossover_max_hz(self) -> float:
        return 2.0 * self.bpf_max_hz


def load_x8_240_acoustic_params(config_path: Path | None = None) -> CoaxialAcousticParams:
    """Load canonical acoustic constants directly from frozen config/vehicle_x8_240.yaml."""
    cfg_path = config_path or (REPO_ROOT / "config/vehicle_x8_240.yaml")
    data: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    geom = data["geometry"]
    prop = data["propulsion"]
    return CoaxialAcousticParams(
        rotor_count=int(geom["rotor_count"]),
        blades_per_rotor=int(prop["blades_per_rotor"]),
        rotor_radius_m=float(geom["rotor_radius_m"]),
        coax_spacing_m=float(geom["coax_spacing_m"]),
        omega_hover_rad_s=float(prop["omega_hover_rad_s"]),
        omega_max_rad_s=float(prop["omega_max_rad_s"]),
        rpm_min=1100.0,
    )


X8_240_ACOUSTIC_PARAMS: CoaxialAcousticParams = load_x8_240_acoustic_params()


def classify_snr_stratum(snr_db: float) -> str:
    """Map SNR (dB) to canonical 4-stratum cockpit taxonomy (S0_Clean..S3_Garbled)."""
    if snr_db > 15.0:
        return "S0_Clean"
    if snr_db >= 8.0:
        return "S1_Cruise"
    if snr_db >= 2.0:
        return "S2_Hover"
    return "S3_Garbled"


def _apply_fft_bandpass(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Zero-dependency pure-NumPy smooth FFT bandpass (300..3400 Hz VHF channel)."""
    n = signal.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    spec = np.fft.rfft(signal)
    # 4th-order Butterworth-like smooth magnitude response
    lo_mask = 1.0 / np.sqrt(1.0 + (lo_hz / np.maximum(freqs, 1e-3)) ** 8)
    hi_mask = 1.0 / np.sqrt(1.0 + (freqs / max(hi_hz, 1.0)) ** 8)
    mask = lo_mask * hi_mask
    mask[0] = 0.0
    return np.fft.irfft(spec * mask, n=n).astype(np.float32)


def _apply_cabin_rir(signal: np.ndarray, sr: int, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    """Apply short eVTOL cabin impulse response (RT60 in [0.12, 0.25] s)."""
    rt60 = float(np.clip(rt60_s, 0.08, 0.35))
    ir_len = max(64, int(round(min(rt60 * 1.1, 0.30) * sr)))
    t_ir = np.arange(ir_len, dtype=np.float64) / float(sr)
    decay = np.exp(-6.907755 * t_ir / rt60)
    ir = rng.normal(0.0, 0.08, size=ir_len) * decay
    ir[0] = 1.0
    # Add 3 discrete cabin bulkhead reflections
    for delay_ms, gain in ((3.2, 0.28), (7.8, 0.17), (14.5, 0.09)):
        idx = int(round((delay_ms / 1000.0) * sr))
        if idx < ir_len:
            ir[idx] += gain
    ir /= np.sqrt(np.sum(ir * ir)) + 1e-12
    conv = np.convolve(signal.astype(np.float64), ir, mode="full")[: signal.shape[0]]
    return conv.astype(np.float32)


def synthesize_bvi_cockpit_noise(
    duration_s: float,
    rpm: float = 1650.0,
    sr: int = 16000,
    seed: int = 0,
    *,
    rt60_s: float = 0.18,
    vhf_bandlimit: bool = False,
    bvi_intensity: float = 0.45,
    params: CoaxialAcousticParams = X8_240_ACOUSTIC_PARAMS,
) -> np.ndarray:
    """Synthesize deterministic 8-rotor coaxial BVI + PMSM + cabin turbulence noise."""
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    rpm_clamped = float(np.clip(rpm, params.rpm_min, params.rpm_max))
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(seed)

    # 1) 8-rotor fundamental BPF & harmonics (k=1..6) with slight per-arm beat detuning (±0.8%)
    f_shaft = rpm_clamped / 60.0
    f_bpf = params.blades_per_rotor * f_shaft
    rotor_sig = np.zeros(n, dtype=np.float64)
    for arm in range(params.rotor_count // 2):
        detune = 1.0 + rng.uniform(-0.008, 0.008)
        phase0 = rng.uniform(0.0, 2.0 * math.pi)
        f_arm = f_bpf * detune
        for k in range(1, 7):
            amp = 1.0 / (float(k) ** 1.25)
            rotor_sig += amp * np.sin(2.0 * math.pi * (k * f_arm) * t + phase0 * k)

    # 2) Coaxial upper-lower blade crossover BVI impulsive modulation (2 * B * f_shaft)
    f_cross = 2.0 * f_bpf
    bvi_carrier = np.power(0.5 * (1.0 + np.cos(2.0 * math.pi * f_cross * t)), 4.0)
    bvi_burst = bvi_intensity * bvi_carrier * (
        0.7 * np.sin(2.0 * math.pi * (3.0 * f_bpf) * t + 0.4)
        + 0.3 * np.sin(2.0 * math.pi * (5.0 * f_bpf) * t + 1.1)
    )

    # 3) 10-pole-pair PMSM electrical inverter whine (200..3500 Hz)
    f_pmsm = float(np.clip(10.0 * f_shaft * 4.0, 350.0, 3500.0))
    pmsm_sig = 0.14 * np.sin(2.0 * math.pi * f_pmsm * t + rng.uniform(0.0, 2.0 * math.pi))

    # 4) Broadband pink-tilted aerodynamic turbulence floor
    white = rng.normal(0.0, 1.0, size=n)
    w_spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    pink_filter = 1.0 / np.sqrt(np.maximum(freqs, 25.0))
    pink_filter[0] = 0.0
    pink = np.fft.irfft(w_spec * pink_filter, n=n)
    pink /= np.std(pink) + 1e-12
    aero_sig = 0.35 * pink

    raw = (0.45 * rotor_sig / (np.std(rotor_sig) + 1e-12)) + bvi_burst + pmsm_sig + aero_sig
    out = _apply_cabin_rir(raw.astype(np.float32), sr=sr, rt60_s=rt60_s, rng=rng)
    if vhf_bandlimit:
        out = _apply_fft_bandpass(out, sr=sr, lo_hz=300.0, hi_hz=3400.0)
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) + 1e-12
    return (0.20 * out / rms).astype(np.float32)


def measure_snr_db(clean_speech: np.ndarray, scaled_noise: np.ndarray) -> float:
    """Compute exact RMS SNR in dB between clean speech and added noise."""
    p_sig = float(np.mean(clean_speech.astype(np.float64) ** 2))
    p_noise = float(np.mean(scaled_noise.astype(np.float64) ** 2))
    return 10.0 * math.log10(max(p_sig, 1e-15) / max(p_noise, 1e-15))


def mix_at_snr(
    clean_speech: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, float]:
    """Mix clean speech and BVI noise at exact target SNR (dB), returning (mixed, measured_snr_db)."""
    s = np.asarray(clean_speech, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if s.shape[0] == 0:
        raise ValueError("clean_speech must be non-empty")
    if n.shape[0] < s.shape[0]:
        reps = int(math.ceil(s.shape[0] / max(1, n.shape[0])))
        n = np.tile(n, reps)
    n = n[: s.shape[0]]

    p_sig = float(np.mean(s * s))
    p_noise = float(np.mean(n * n))
    if p_sig <= 1e-15 or p_noise <= 1e-15:
        raise ValueError("clean_speech and noise must have non-zero power")

    desired_noise_power = p_sig / (10.0 ** (float(target_snr_db) / 10.0))
    scale = math.sqrt(desired_noise_power / p_noise)
    scaled_noise = n * scale
    mixed = (s + scaled_noise).astype(np.float32)
    actual_snr = measure_snr_db(s, scaled_noise)
    return mixed, actual_snr
