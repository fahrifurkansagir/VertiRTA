"""Gymnasium v1.1+ vectorized multi-threaded C++20 nanobind environment (VertiVecEnv)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np

from .env import ACT_DIM, FAULT_MODE_MAP, OBS_DIM, _PythonFallbackSim

try:
    from . import _vertirta_native as _native
except ImportError:
    _native = None


class VertiVecEnv(VectorEnv):
    """High-throughput vectorized VertiRTA-X8-240 environment supporting all 3 Gymnasium v1.1+ AutoresetModes."""

    def __init__(
        self,
        num_envs: int = 8,
        *,
        fault_mode: str = "none",
        substeps_400hz: int = 4,
        max_steps: int = 500,
        num_threads: int = 0,
        autoreset_mode: AutoresetMode = AutoresetMode.NEXT_STEP,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if fault_mode not in FAULT_MODE_MAP:
            raise ValueError(
                f"Unknown fault_mode {fault_mode!r}; expected one of {sorted(FAULT_MODE_MAP)}"
            )
        if not isinstance(autoreset_mode, AutoresetMode):
            raise TypeError(f"Expected gymnasium.vector.AutoresetMode, got {type(autoreset_mode)}")

        self.num_envs = int(num_envs)
        self.fault_mode = fault_mode
        self.substeps_400hz = int(max(1, substeps_400hz))
        self.max_steps = int(max(1, max_steps))
        self.autoreset_mode = autoreset_mode
        self.metadata = {
            "render_modes": [],
            "autoreset_mode": autoreset_mode,
        }

        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        if _native is not None:
            self._vec_sim: Any = _native.NativeVecSim(self.num_envs, int(max(0, num_threads)))
            self.backend: str = "cpp_nanobind"
        else:
            self._vec_sim = None
            self._py_sims = [_PythonFallbackSim(True) for _ in range(self.num_envs)]
            self.backend = "python_fallback"

        self._rngs: list[np.random.Generator] = [
            np.random.default_rng() for _ in range(self.num_envs)
        ]
        self._obs_buf = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        self._rew_buf = np.zeros((self.num_envs,), dtype=np.float32)
        self._term_u8 = np.zeros((self.num_envs,), dtype=np.uint8)
        self._cel_u8 = np.zeros((self.num_envs,), dtype=np.uint8)
        self._steps = np.zeros((self.num_envs,), dtype=np.int32)
        self._needs_reset = np.zeros((self.num_envs,), dtype=bool)
        self._fault_injected = np.zeros((self.num_envs,), dtype=bool)

    def _reset_env_index(self, idx: int) -> np.ndarray:
        rng = self._rngs[idx]
        jitter_xy = rng.uniform(-0.25, 0.25, size=2)
        wind_xy = rng.uniform(-0.5, 0.5, size=2)
        if self._vec_sim is not None:
            obs = np.asarray(
                self._vec_sim.reset_one(
                    idx,
                    (float(jitter_xy[0]), float(jitter_xy[1]), 35.0),
                    (0.0, 0.0, 0.0),
                    0.80,
                    0,
                    (float(wind_xy[0]), float(wind_xy[1]), 0.0),
                ),
                dtype=np.float32,
            )
        else:
            obs = self._py_sims[idx].reset(
                (float(jitter_xy[0]), float(jitter_xy[1]), 35.0),
                (0.0, 0.0, 0.0),
                0.80,
                0,
                (float(wind_xy[0]), float(wind_xy[1]), 0.0),
            )
        self._obs_buf[idx] = obs
        self._steps[idx] = 0
        self._needs_reset[idx] = False
        self._fault_injected[idx] = False
        return obs

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        opts = options or {}
        mask_raw = opts.get("reset_mask", None)
        if mask_raw is None:
            reset_mask = np.ones((self.num_envs,), dtype=bool)
        else:
            reset_mask = np.asarray(mask_raw, dtype=bool).reshape((self.num_envs,))

        if isinstance(seed, int):
            for i in range(self.num_envs):
                if reset_mask[i]:
                    self._rngs[i] = np.random.default_rng(seed + i)
        elif isinstance(seed, list):
            if len(seed) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} seeds, got {len(seed)}")
            for i, s_i in enumerate(seed):
                if reset_mask[i] and s_i is not None:
                    self._rngs[i] = np.random.default_rng(int(s_i))

        for i in range(self.num_envs):
            if reset_mask[i]:
                self._reset_env_index(i)

        return self._obs_buf.copy(), {
            "backend": self.backend,
            "autoreset_mode": self.autoreset_mode,
        }

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        act = np.ascontiguousarray(
            np.clip(np.asarray(actions, dtype=np.float32).reshape((self.num_envs, ACT_DIM)), -1.0, 1.0)
        )

        # NEXT_STEP mode resets previously done environments at the start of the next step()
        stepped_mask = np.ones((self.num_envs,), dtype=bool)
        if self.autoreset_mode == AutoresetMode.NEXT_STEP and bool(self._needs_reset.any()):
            for i in range(self.num_envs):
                if self._needs_reset[i]:
                    self._reset_env_index(i)
                    stepped_mask[i] = False

        if self.fault_mode != "none":
            fault_code = FAULT_MODE_MAP[self.fault_mode]
            for i in range(self.num_envs):
                if stepped_mask[i] and (not self._fault_injected[i]) and self._steps[i] >= 10:
                    if self._vec_sim is not None:
                        self._vec_sim.inject_fault_one(i, fault_code, 0)
                    else:
                        self._py_sims[i].inject_fault(fault_code, 0)
                    self._fault_injected[i] = True

        if self._vec_sim is not None:
            saved_reset_obs = self._obs_buf[~stepped_mask].copy() if not bool(stepped_mask.all()) else None
            self._vec_sim.step_batch(
                act,
                self.substeps_400hz,
                self._obs_buf,
                self._rew_buf,
                self._term_u8,
                self._cel_u8,
            )
            if saved_reset_obs is not None:
                self._obs_buf[~stepped_mask] = saved_reset_obs
                self._rew_buf[~stepped_mask] = 0.0
                self._term_u8[~stepped_mask] = 0
                self._cel_u8[~stepped_mask] = 0
        else:
            for i in range(self.num_envs):
                if not stepped_mask[i]:
                    self._rew_buf[i] = 0.0
                    self._term_u8[i] = 0
                    self._cel_u8[i] = 0
                    continue
                obs_i, rew_i, term_i, cel_i, _ = self._py_sims[i].step_400hz(
                    self.substeps_400hz,
                    float(act[i, 0]) * 4.0,
                    float(act[i, 1]) * 4.0,
                    float(act[i, 2]) * 3.0,
                    float(act[i, 3]) * 1.0,
                )
                self._obs_buf[i] = obs_i
                self._rew_buf[i] = rew_i
                self._term_u8[i] = 1 if term_i else 0
                self._cel_u8[i] = 1 if cel_i else 0

        self._steps[stepped_mask] += 1
        terminated = self._term_u8.astype(bool)
        truncated = (~terminated) & (self._steps >= self.max_steps) & stepped_mask
        done = terminated | truncated

        info: dict[str, Any] = {
            "backend": self.backend,
            "cel_latched": self._cel_u8.astype(bool).copy(),
        }

        if self.autoreset_mode == AutoresetMode.SAME_STEP and bool(done.any()):
            final_obs = self._obs_buf.copy()
            info["final_obs"] = final_obs
            info["_final_obs"] = done.copy()
            for i in range(self.num_envs):
                if done[i]:
                    self._reset_env_index(i)
        elif self.autoreset_mode == AutoresetMode.NEXT_STEP:
            self._needs_reset = done.copy()

        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            terminated,
            truncated,
            info,
        )

    def close(self, **kwargs: Any) -> None:
        super().close(**kwargs)
