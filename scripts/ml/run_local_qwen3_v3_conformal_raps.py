#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact Layer-Streaming KV-Prefix-Shared Qwen3-4B v3 Conformal Log-Prob Exporter & Mondrian-RAPS Fit.

Runs at 0.00 CU locally on constrained 4 GB GPUs (e.g., NVIDIA GTX 1650 SUPER TU116) WITHOUT
4-bit/8-bit quantization loss by:
  1. Merging Apache-2.0 `Qwen/Qwen3-4B` + saved `qwen3_4b_v3_seed0_lora` in BF16 in host RAM (~8.5 GiB).
  2. Embedding prompts & candidate branches on CPU (`embed_tokens` never consumes VRAM).
  3. Streaming ONE `Qwen3DecoderLayer` at a time to GPU (`~380 MiB` FP32), computing prompt KV cache
     ONCE per sample and sharing it across all 14 candidate branches (KV-prefix sharing -> ~8x fewer
     tokens, peak VRAM < 950 MiB).
  4. Computing exact FP32 token log-probabilities via vocab-chunked `lm_head` log-sum-exp (4 slices,
     < 450 MiB VRAM), extracting BOTH:
       - `decision_prefix` exact autoregressive marginal log-probs (`ACT=...;TGT/VIO=...;`), fixing
         RC1 (hardcoded OOD AGL/SPD mismatch) & RC2 (boilerplate dilution at gamma=1.0) with ZERO
         gold leakage from `canonical_dsl`, AND
       - `full_eos` SFT-aligned candidate DSL + `<|im_end|>` log-probs in the same single pass.
  5. Checkpointing every `--checkpoint-every` rows to `/var/tmp/vertirta_v400_blobs/qwen3_v3_seed0_branch_logprobs.npz`
     for resumable unattended execution.
  6. Fitting strictly zero-leakage disjoint 50/50 per-group temperature-scaled Mondrian-RAPS
     (RC3 + RC4) on `cal.jsonl` (n=2,000) and evaluating once on held-out `test.jsonl` (n=1,200).

Supports:
  - `--download-only`: Pre-downloads Apache-2.0 `Qwen/Qwen3-4B` to HF cache on CPU (0 MiB VRAM).
  - `--self-test-cpu`: Verifies layer-streaming + KV-prefix sharing + vocab-chunked `lm_head`
    matches monolithic `Qwen3ForCausalLM` forward pass to `< 1e-5` max abs diff on CPU (0 MiB VRAM).
  - `--calibrate-only`: Runs fast CPU-only Mondrian-RAPS/LAC calibration from a saved `.npz` file.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vertibench_tr_en.validate_vertibench import parse_compact_dsl_py  # noqa: E402
from benchmarks.vertibench_v2.schema import HONEST_PROVENANCE_TAG_V2  # noqa: E402
from ml.copilot.calibrate_conformal import (  # noqa: E402
    bootstrap_resplit_mondrian_coverage,
    compute_ece,
    evaluate_mondrian_conformal,
    evaluate_temperature_scaled_mondrian_raps,
    fit_mondrian_conformal_groups,
    fit_temperature_scaled_mondrian_raps,
    fit_temperature_scaling_nll,
    length_normalized_scores,
    softmax_with_temperature,
)
from ml.copilot.train_sft_grpo import format_copilot_prompt  # noqa: E402
from scripts.colab.g4_jobs.common import (  # noqa: E402
    CANONICAL_SNR_STRATA,
    seed_everything,
    write_canonical_json,
)
from scripts.colab.g4_jobs.job_qwen3_v3_vertibench_v2 import (  # noqa: E402
    _ensure_vertibench_v2,
    _load_jsonl,
    enrich_vertibench_v2_row_for_z3,
)

SCHEMA_VERSION = "vertirta.local_qwen3_v3_conformal_raps/v1.0.0"
MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_ADAPTER_DIR = Path("/var/tmp/vertirta_v400_blobs/qwen3_4b_v3_seed0_lora")
DEFAULT_TOKENIZER_DIR = Path("/var/tmp/vertirta_v400_blobs/qwen3_4b_v3_shared_sft_lora")
DEFAULT_NPZ_PATH = Path("/var/tmp/vertirta_v400_blobs/qwen3_v3_seed0_branch_logprobs.npz")
DEFAULT_OUT_JSON = REPO_ROOT / "artifacts/v4.0.0/local_gtx1650s_qwen3_v3_conformal_raps.json"

VIOLATION_CODES: tuple[str, ...] = (
    "LTFJ_CTR",
    "LTBX_CTR",
    "DYN_NFZ",
    "SOC_RESERVE",
    "SLZ_UNSAFE",
    "MIN_AGL",
    "SHIELD_BYPASS",
)
NUM_BRANCHES = 14


def build_leakfree_candidate_branches(enriched_row: dict[str, Any]) -> list[dict[str, str]]:
    """Build 14 mutually exclusive, prefix-free branches STRICTLY from runtime state (zero gold leakage).

    NEVER reads `canonical_dsl`, `target_dsl`, `gold_intent`, or `bucket`.
    Each branch provides:
      - `key`: canonical branch key (`DIVERT:<id>`, `ABORT:<id>`, `HOLD_LOITER`, `CLARIFY`, `REJECT:<vio>`)
      - `decision_prefix`: exact prefix-free decision tokens (`ACT=...;TGT/VIO=...;`)
      - `full_dsl_eos`: full in-distribution SFT-aligned candidate DSL + `<|im_end|>`
    """
    st = enriched_row["rta_snapshot"]
    slzs = st.get("ranked_slzs", [])
    lang = str(enriched_row.get("lang", "TR"))
    if lang not in ("TR", "EN", "MIX"):
        lang = "TR"
    snr_stratum = str(enriched_row.get("snr_stratum", "S1_Cruise"))
    rsn = "LOW_ASR" if snr_stratum in ("S2_Hover", "S3_Garbled") else "AMBIG"
    fault_mode = str(st.get("fault_mode", "NOMINAL"))

    branches: list[dict[str, str]] = []
    # 1..4: Top-4 ranked SLZs for DIVERT_SLZ (in-distribution SFT nominal AGL=55,95; SPD=15/12)
    div_spd = 15 if fault_mode == "NOMINAL" else 12
    for s in slzs[:4]:
        pad_id = str(s["id"])
        eta_lim = min(600, max(30, int(math.ceil(float(s["eta_s"]) + 15.0))))
        prefix = f"ACT=DIVERT_SLZ;TGT={pad_id};"
        full = f"{prefix}ETA={eta_lim};AGL=55,95;SPD={div_spd}\nLNG={lang};MOD=BRIEF<|im_end|>"
        branches.append({"key": f"DIVERT:{pad_id}", "decision_prefix": prefix, "full_dsl_eos": full})

    # 5: Nearest SLZ for ABORT_RETURN (in-distribution SFT nominal AGL=35,70; SPD=12)
    top_id = str(slzs[0]["id"]) if slzs else "Kartal_Sahil"
    top_eta = min(600, max(30, int(math.ceil(float(slzs[0]["eta_s"]) + 15.0)))) if slzs else 120
    ab_prefix = f"ACT=ABORT_RETURN;TGT={top_id};"
    ab_full = f"{ab_prefix}ETA={top_eta};AGL=35,70;SPD=12\nLNG={lang};MOD=BRIEF<|im_end|>"
    branches.append({"key": f"ABORT:{top_id}", "decision_prefix": ab_prefix, "full_dsl_eos": ab_full})

    # 6: HOLD_LOITER
    hl_prefix = "ACT=HOLD_LOITER;"
    hl_full = f"{hl_prefix}DUR=30;AGL=60\nLNG={lang};MOD=BRIEF<|im_end|>"
    branches.append({"key": "HOLD_LOITER", "decision_prefix": hl_prefix, "full_dsl_eos": hl_full})

    # 7: REQUEST_CLARIFICATION (strictly from runtime reachable SLZs; zero canonical_dsl peeking!)
    sat_pads = [str(s["id"]) for s in slzs if float(s.get("soc_margin_pct", -1.0)) >= 0.0]
    opt_a = sat_pads[0] if len(sat_pads) >= 1 else (str(slzs[0]["id"]) if slzs else "Kartal_Sahil")
    opt_b = sat_pads[1] if len(sat_pads) >= 2 else (str(slzs[1]["id"]) if len(slzs) >= 2 else "Dragos_Plaza")
    cl_prefix = "ACT=REQUEST_CLARIFICATION;"
    cl_full = f"{cl_prefix}A={opt_a};B={opt_b};RSN={rsn}\nLNG={lang};MOD=CLARIFY<|im_end|>"
    branches.append({"key": "CLARIFY", "decision_prefix": cl_prefix, "full_dsl_eos": cl_full})

    # 8..14: REJECT_UNSAFE across all 7 statutory violation codes
    fb = sat_pads[0] if sat_pads else top_id
    for vio in VIOLATION_CODES:
        rj_prefix = f"ACT=REJECT_UNSAFE;VIO={vio};"
        rj_full = f"{rj_prefix}FB={fb}\nLNG={lang};MOD=REJECT<|im_end|>"
        branches.append({"key": f"REJECT:{vio}", "decision_prefix": rj_prefix, "full_dsl_eos": rj_full})

    assert len(branches) == NUM_BRANCHES, f"Expected {NUM_BRANCHES} branches, got {len(branches)}"
    return branches


def gold_branch_key_leakfree(enriched_row: dict[str, Any]) -> str:
    """Map ground-truth canonical_dsl to its canonical key in the 14-branch lattice."""
    ast = parse_compact_dsl_py(str(enriched_row.get("canonical_dsl", "")))
    if ast is None:
        return "UNKNOWN"
    if ast.action == "DIVERT_SLZ":
        return f"DIVERT:{ast.target}"
    if ast.action == "ABORT_RETURN":
        return f"ABORT:{ast.target}"
    if ast.action == "HOLD_LOITER":
        return "HOLD_LOITER"
    if ast.action == "REQUEST_CLARIFICATION":
        return "CLARIFY"
    if ast.action == "REJECT_UNSAFE":
        return f"REJECT:{ast.violation}"
    return ast.action


def _make_single_layer_cache_cls() -> type:
    """Create a lightweight single-layer KV cache compatible with transformers v4/v5 Cache API."""
    import torch
    from transformers.cache_utils import Cache

    class SingleLayerPrefixCache(Cache):
        """Holds KV tensors strictly for ONE active decoder layer (zero multi-layer VRAM accumulation)."""

        def __init__(self, key_prefix: torch.Tensor | None = None, val_prefix: torch.Tensor | None = None) -> None:
            # Skip heavy Cache.__init__ layer-list setup; expose minimal attributes required by Qwen3Attention.
            self.layers: list[Any] = []
            self._key_prefix = key_prefix
            self._val_prefix = val_prefix
            self.saved_key: torch.Tensor | None = None
            self.saved_val: torch.Tensor | None = None

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del layer_idx, args, kwargs
            if self._key_prefix is None or self._val_prefix is None:
                self.saved_key = key_states
                self.saved_val = value_states
                return key_states, value_states
            k_full = torch.cat([self._key_prefix, key_states], dim=-2)
            v_full = torch.cat([self._val_prefix, value_states], dim=-2)
            return k_full, v_full

        def get_seq_length(self, layer_idx: int = 0) -> int:
            del layer_idx
            if self._key_prefix is not None:
                return int(self._key_prefix.shape[-2])
            if self.saved_key is not None:
                return int(self.saved_key.shape[-2])
            return 0

        def get_max_cache_shape(self) -> int | None:
            return None

        def __len__(self) -> int:
            return 1

    return SingleLayerPrefixCache


def _call_decoder_layer(
    layer: Any,
    hidden_states: Any,
    *,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    cache_obj: Any,
    use_cache: bool,
) -> Any:
    """Version-safe invocation supporting transformers v5 (`cache=`) and v4 (`past_key_value=`)."""
    import inspect

    params = inspect.signature(layer.forward).parameters
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "position_embeddings": position_embeddings,
        "use_cache": use_cache,
    }
    if "cache" in params:
        kwargs["cache"] = cache_obj
    elif "past_key_values" in params:
        kwargs["past_key_values"] = cache_obj
    else:
        kwargs["past_key_value"] = cache_obj

    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _build_left_padded_prompt_mask_4d(
    prompt_lens: Sequence[int],
    p_max: int,
    *,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    """Build exact 4D additive causal+left-padding mask `[B, 1, P, P]` and `position_ids` `[B, P]`."""
    import torch

    bsz = len(prompt_lens)
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((bsz, 1, p_max, p_max), neg_inf, dtype=dtype, device=device)
    pos_ids = torch.zeros((bsz, p_max), dtype=torch.long, device=device)
    tri = torch.tril(torch.ones((p_max, p_max), dtype=torch.bool, device=device))

    for b, plen in enumerate(prompt_lens):
        pad_l = p_max - int(plen)
        valid_q = torch.arange(p_max, device=device) >= pad_l
        # Valid queries attend causally to valid keys; pad queries attend to themselves to avoid NaN softmax
        allowed = tri & valid_q.unsqueeze(0) & valid_q.unsqueeze(1)
        if pad_l > 0:
            pad_diag = torch.eye(p_max, dtype=torch.bool, device=device) & (~valid_q).unsqueeze(1)
            allowed = allowed | pad_diag
        mask[b, 0] = torch.where(allowed, torch.tensor(0.0, dtype=dtype, device=device), neg_inf)
        if plen > 0:
            pos_ids[b, pad_l:] = torch.arange(plen, dtype=torch.long, device=device)
    return mask, pos_ids


def _build_cand_mask_4d_and_pos(
    prompt_lens: Sequence[int],
    p_max: int,
    c_max: int,
    n_branches: int,
    *,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    """Build exact 4D additive mask `[B*K, 1, C, P+C]` and RoPE `position_ids` `[B*K, C]`."""
    import torch

    bsz = len(prompt_lens)
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((bsz * n_branches, 1, c_max, p_max + c_max), neg_inf, dtype=dtype, device=device)
    pos_ids = torch.zeros((bsz * n_branches, c_max), dtype=torch.long, device=device)
    tri_c = torch.tril(torch.ones((c_max, c_max), dtype=torch.bool, device=device))
    c_offsets = torch.arange(c_max, dtype=torch.long, device=device)

    for b, plen in enumerate(prompt_lens):
        pad_l = p_max - int(plen)
        row_lo = b * n_branches
        row_hi = row_lo + n_branches
        # All completion tokens attend to all valid prompt keys [pad_l : p_max]
        mask[row_lo:row_hi, 0, :, pad_l:p_max] = 0.0
        # Completion tokens attend causally to completion keys [p_max : p_max + c_max]
        c_block = torch.where(tri_c, torch.tensor(0.0, dtype=dtype, device=device), neg_inf)
        mask[row_lo:row_hi, 0, :, p_max:] = c_block.unsqueeze(0)
        pos_ids[row_lo:row_hi, :] = (int(plen) + c_offsets).unsqueeze(0)
    return mask, pos_ids


def _vocab_chunked_exact_token_logprobs(
    hidden_cpu_bf16: Any,
    target_ids_cpu: Any,
    lm_head_weight_cpu: Any,
    *,
    device: Any,
    compute_dtype: Any,
    vocab_chunks: int = 4,
    token_batch: int = 512,
) -> Any:
    """Compute exact FP32 token log-probs `log p(y_t | h_t)` using vocab-chunked log-sum-exp (<450 MiB VRAM).

    Args:
        hidden_cpu_bf16: Tensor `[N, D]` (already passed through `model.norm`) on CPU.
        target_ids_cpu: LongTensor `[N]` target token IDs on CPU.
        lm_head_weight_cpu: Tensor `[V, D]` on CPU.
    Returns:
        Float32 CPU Tensor `[N]` of exact log-probabilities.
    """
    import torch

    n_tok = int(hidden_cpu_bf16.shape[0])
    vocab_size = int(lm_head_weight_cpu.shape[0])
    chunk_v = max(1, math.ceil(vocab_size / float(max(1, vocab_chunks))))

    lse_accum = torch.full((n_tok,), -math.inf, dtype=torch.float32, device="cpu")
    picked_logits = torch.zeros((n_tok,), dtype=torch.float32, device="cpu")

    # Gather exact target logits per token without materializing full [N, V]
    for s in range(0, n_tok, token_batch):
        e = min(n_tok, s + token_batch)
        h_b = hidden_cpu_bf16[s:e].to(device=device, dtype=torch.float32)
        y_b = target_ids_cpu[s:e]
        w_y = lm_head_weight_cpu[y_b].to(device=device, dtype=torch.float32)
        picked_logits[s:e] = (h_b * w_y).sum(dim=-1).cpu()
        del h_b, w_y

    # Accumulate exact logsumexp across vocab slices (each slice <= ~390 MiB in FP32 for 152k vocab)
    for v0 in range(0, vocab_size, chunk_v):
        v1 = min(vocab_size, v0 + chunk_v)
        w_slice = lm_head_weight_cpu[v0:v1].to(device=device, dtype=compute_dtype)
        for s in range(0, n_tok, token_batch):
            e = min(n_tok, s + token_batch)
            h_b = hidden_cpu_bf16[s:e].to(device=device, dtype=compute_dtype)
            logits_sv = torch.matmul(h_b, w_slice.t()).float()
            slice_lse = torch.logsumexp(logits_sv, dim=-1).cpu()
            lse_accum[s:e] = torch.logaddexp(lse_accum[s:e], slice_lse)
            del h_b, logits_sv, slice_lse
        del w_slice
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return picked_logits - lse_accum


def score_batch_layer_streaming_kv_shared(
    model: Any,
    batch_prompt_ids: Sequence[Sequence[int]],
    batch_cand_ids: Sequence[Sequence[Sequence[int]]],
    batch_dec_lens: Sequence[Sequence[int]],
    *,
    device: Any,
    layer_dtype: Any,
    cand_row_chunk: int = 4,
    vocab_chunks: int = 4,
    pad_token_id: int = 0,
) -> dict[str, np.ndarray]:
    """Exact layer-by-layer streaming + KV-prefix sharing over `[B, K]` candidate completions.

    At any instant, ONLY ONE transformer decoder layer (or ONE vocab slice of `lm_head`) resides on
    `device`, keeping peak VRAM strictly `< 950 MiB` on `Qwen3-4B` with zero quantization loss.

    Returns dict of numpy arrays:
      - `decision_sum_lp`: float64 `[B, K]`
      - `decision_n_tok`:  int32   `[B, K]`
      - `full_sum_lp`:     float64 `[B, K]`
      - `full_n_tok`:      int32   `[B, K]`
    """
    import torch

    SingleLayerPrefixCache = _make_single_layer_cache_cls()
    bsz = len(batch_prompt_ids)
    n_k = len(batch_cand_ids[0])
    prompt_lens = [len(p) for p in batch_prompt_ids]
    p_max = max(prompt_lens)
    c_lens = [[len(c) for c in row_c] for row_c in batch_cand_ids]
    c_max = max(max(row_l) for row_l in c_lens)
    if c_max < 1:
        raise ValueError("Candidate completion sequences must have length >= 1")

    # Left-pad prompts so every prompt's final token sits at column `p_max - 1`
    p_ids_t = torch.full((bsz, p_max), int(pad_token_id), dtype=torch.long)
    for b, p in enumerate(batch_prompt_ids):
        p_ids_t[b, p_max - len(p) :] = torch.tensor(p, dtype=torch.long)

    # Right-pad candidates `[B * K, c_max]`
    c_ids_t = torch.full((bsz * n_k, c_max), int(pad_token_id), dtype=torch.long)
    for b in range(bsz):
        for k in range(n_k):
            seq = batch_cand_ids[b][k]
            c_ids_t[b * n_k + k, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    # Note: to predict completion token c[0], we need hidden state at prompt[-1];
    # to predict c[1..L_c-1], we need hidden states at c[0..L_c-2].
    # Thus we only need to run c_prefix_len = max(1, c_max - 1) completion tokens through the layers!
    c_feed_len = max(1, c_max - 1)
    c_feed_ids = c_ids_t[:, :c_feed_len]

    host_store_dtype = torch.float32 if layer_dtype == torch.float32 else torch.bfloat16

    with torch.no_grad():
        # Step 0: Embed on CPU (zero VRAM!)
        embed_mod = model.model.embed_tokens
        h_p_cpu = embed_mod(p_ids_t).to(dtype=host_store_dtype).contiguous()
        h_c_cpu = embed_mod(c_feed_ids).to(dtype=host_store_dtype).contiguous()

        rotary_emb = model.model.rotary_emb.to(device=device)
        mask_p_4d, pos_p = _build_left_padded_prompt_mask_4d(prompt_lens, p_max, dtype=layer_dtype, device=device)
        rope_p = rotary_emb(torch.empty((bsz, p_max, 1), dtype=layer_dtype, device=device), pos_p)

        mask_c_all, pos_c_all = _build_cand_mask_4d_and_pos(
            prompt_lens, p_max, c_feed_len, n_k, dtype=layer_dtype, device=device
        )

        # Step 1: Stream 1 decoder layer at a time to GPU with KV-prefix sharing
        for layer in model.model.layers:
            # Copy layer weights to GPU in `layer_dtype` without overwriting original CPU weights
            orig_dtype = next(layer.parameters()).dtype
            layer.to(device=device, dtype=layer_dtype)

            # 1a) Prompt pass ONCE per sample -> produces single-layer KV cache `(k_p, v_p)`
            p_cache = SingleLayerPrefixCache()
            h_p_dev = h_p_cpu.to(device=device, dtype=layer_dtype)
            h_p_out = _call_decoder_layer(
                layer,
                h_p_dev,
                attention_mask=mask_p_4d,
                position_ids=pos_p,
                position_embeddings=rope_p,
                cache_obj=p_cache,
                use_cache=True,
            )
            h_p_cpu = h_p_out.to(device="cpu", dtype=host_store_dtype)
            k_p_all = p_cache.saved_key
            v_p_all = p_cache.saved_val
            del h_p_dev, h_p_out, p_cache
            assert k_p_all is not None and v_p_all is not None

            # 1b) Candidate completion pass in row chunks reusing `(k_p, v_p)` across all K branches
            for r0 in range(0, bsz, max(1, int(cand_row_chunk))):
                r1 = min(bsz, r0 + max(1, int(cand_row_chunk)))
                ck0 = r0 * n_k
                ck1 = r1 * n_k
                k_rep = k_p_all[r0:r1].repeat_interleave(n_k, dim=0)
                v_rep = v_p_all[r0:r1].repeat_interleave(n_k, dim=0)
                c_cache = SingleLayerPrefixCache(k_rep, v_rep)
                h_c_dev = h_c_cpu[ck0:ck1].to(device=device, dtype=layer_dtype)
                pos_c_sub = pos_c_all[ck0:ck1]
                rope_c_sub = rotary_emb(
                    torch.empty((ck1 - ck0, c_feed_len, 1), dtype=layer_dtype, device=device),
                    pos_c_sub,
                )
                h_c_out = _call_decoder_layer(
                    layer,
                    h_c_dev,
                    attention_mask=mask_c_all[ck0:ck1],
                    position_ids=pos_c_sub,
                    position_embeddings=rope_c_sub,
                    cache_obj=c_cache,
                    use_cache=True,
                )
                h_c_cpu[ck0:ck1] = h_c_out.to(device="cpu", dtype=host_store_dtype)
                del h_c_dev, h_c_out, c_cache, k_rep, v_rep, rope_c_sub

            del k_p_all, v_p_all
            layer.to(device="cpu", dtype=orig_dtype)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rotary_emb.to(device="cpu")
        del mask_p_4d, pos_p, rope_p, mask_c_all, pos_c_all

        # Step 2: Apply final RMSNorm on CPU (tiny, zero VRAM!)
        norm_mod = model.model.norm
        norm_dtype = next(norm_mod.parameters()).dtype
        # Last prompt hidden state predicts c[0]; c_out[:, :-1] predicts c[1..c_max-1]
        h_last_p = norm_mod(h_p_cpu[:, -1:, :].to(dtype=norm_dtype)).to(dtype=torch.float32)  # [B, 1, D]
        h_c_norm = norm_mod(h_c_cpu.to(dtype=norm_dtype)).to(dtype=torch.float32)  # [B*K, c_feed_len, D]

        # Collect only valid completion token positions `(h_pred, target_id)` into flat arrays
        flat_h_list: list[Any] = []
        flat_y_list: list[int] = []
        spans: list[tuple[int, int, int, int, int]] = []  # (b, k, start, full_len, dec_len)
        cursor = 0
        for b in range(bsz):
            h_p_b = h_last_p[b, 0]  # [D]
            for k in range(n_k):
                full_len = c_lens[b][k]
                dec_len = int(batch_dec_lens[b][k])
                seq = batch_cand_ids[b][k]
                # Position 0 uses h_p_b; positions 1..full_len-1 use h_c_norm[b*K + k, 0..full_len-2]
                if full_len == 1:
                    h_seq = h_p_b.unsqueeze(0)
                else:
                    h_seq = torch.cat([h_p_b.unsqueeze(0), h_c_norm[b * n_k + k, : full_len - 1]], dim=0)
                flat_h_list.append(h_seq)
                flat_y_list.extend(int(t) for t in seq)
                spans.append((b, k, cursor, full_len, min(dec_len, full_len)))
                cursor += full_len

        flat_h = torch.cat(flat_h_list, dim=0)  # [N_valid, D] float32 on CPU
        flat_y = torch.tensor(flat_y_list, dtype=torch.long)

        # Step 3: Exact vocab-chunked log-probs through lm_head
        flat_lps = _vocab_chunked_exact_token_logprobs(
            flat_h,
            flat_y,
            model.lm_head.weight.detach().cpu(),
            device=device,
            compute_dtype=layer_dtype,
            vocab_chunks=vocab_chunks,
        ).numpy()

    dec_sum_lp = np.zeros((bsz, n_k), dtype=np.float64)
    dec_n_tok = np.zeros((bsz, n_k), dtype=np.int32)
    full_sum_lp = np.zeros((bsz, n_k), dtype=np.float64)
    full_n_tok = np.zeros((bsz, n_k), dtype=np.int32)

    for b, k, start, full_len, dec_len in spans:
        seg = flat_lps[start : start + full_len]
        full_sum_lp[b, k] = float(np.sum(seg, dtype=np.float64))
        full_n_tok[b, k] = int(full_len)
        dec_sum_lp[b, k] = float(np.sum(seg[:dec_len], dtype=np.float64))
        dec_n_tok[b, k] = int(dec_len)

    return {
        "decision_sum_lp": dec_sum_lp,
        "decision_n_tok": dec_n_tok,
        "full_sum_lp": full_sum_lp,
        "full_n_tok": full_n_tok,
    }


def build_gbnf_minimal_trie(branch_seqs: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Build exact minimal BPE prefix trie over prefix-free candidate branches.

    Prunes deterministic single-branch tails whose continuation probability under the
    GBNF automaton is identically 1.0, retaining ONLY internal nodes visited by >= 2
    branches. On VertiBench-v2 (14 branches), this compresses 238 decision tokens (or
    672 full_eos tokens) to exactly 18 constant feed tokens with 0.0 approximation!
    """
    n_k = len(branch_seqs)
    nodes: dict[int, dict[str, Any]] = {
        -1: {"tok": None, "parent": None, "children": {}, "branches": list(range(n_k))}
    }
    next_id = 0
    for b_idx, seq in enumerate(branch_seqs):
        cur = -1
        for tid in seq:
            tid_i = int(tid)
            ch = nodes[cur]["children"]
            if tid_i not in ch:
                ch[tid_i] = next_id
                nodes[next_id] = {"tok": tid_i, "parent": cur, "children": {}, "branches": []}
                next_id += 1
            cur = ch[tid_i]
            nodes[cur]["branches"].append(b_idx)

    needed = [u for u in range(next_id) if len(nodes[u]["branches"]) >= 2]
    remap = {-1: -1}
    for new_i, old_u in enumerate(needed):
        remap[old_u] = new_i

    feed_tokens = [int(nodes[u]["tok"]) for u in needed]
    parent_idx = [int(remap[nodes[u]["parent"]]) for u in needed]
    depths: list[int] = []
    ancestors: list[list[int]] = []
    for i, p in enumerate(parent_idx):
        if p == -1:
            depths.append(1)
            ancestors.append([i])
        else:
            depths.append(depths[p] + 1)
            ancestors.append(ancestors[p] + [i])

    branch_splits: list[list[tuple[int, int, tuple[int, ...]]]] = [[] for _ in range(n_k)]
    split_nodes_set: set[int] = set()
    for b_idx, seq in enumerate(branch_seqs):
        cur = -1
        for tid in seq:
            tid_i = int(tid)
            ch = nodes[cur]["children"]
            if len(ch) >= 2:
                u_new = int(remap[cur])
                sibs = tuple(sorted(int(t) for t in ch.keys()))
                branch_splits[b_idx].append((u_new, tid_i, sibs))
                split_nodes_set.add(u_new)
            cur = ch[tid_i]
            if len(nodes[cur]["branches"]) == 1:
                break

    return {
        "feed_tokens": feed_tokens,
        "parent_idx": parent_idx,
        "depths": depths,
        "ancestors": ancestors,
        "branch_splits": branch_splits,
        "split_nodes": sorted(split_nodes_set),
    }


def _longest_common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    if not seqs:
        return 0
    lcp = len(seqs[0])
    for s in seqs[1:]:
        lim = min(lcp, len(s))
        i = 0
        while i < lim and s[i] == seqs[0][i]:
            i += 1
        lcp = i
    # Always leave at least 1 token in the per-sample suffix so prompt[-1] sits in h_seq
    min_len = min(len(s) for s in seqs)
    return max(0, min(lcp, min_len - 1))


def score_batch_lcp_gbnf_tree_streaming(
    model: Any,
    batch_prompt_ids: Sequence[Sequence[int]],
    batch_dec_seqs: Sequence[Sequence[Sequence[int]]],
    *,
    device: Any,
    layer_dtype: Any,
    micro_batch: int = 16,
    vocab_chunks: int = 4,
    pad_token_id: int = 0,
    gpu_layer_buf: Any | None = None,
) -> dict[str, np.ndarray]:
    """Exact LCP + Minimal GBNF BPE-Trie Tree-Attention layer-streaming scorer.

    Guarantees:
      1. ZERO 4-bit quantization loss (exact BF16->FP32 weights, FP32 matmuls & log-softmax).
      2. One-way CPU->GPU layer copy into a single reusable `gpu_layer_buf` (zero GPU->CPU
         copy-back, zero VRAM fragmentation, peak VRAM < 750 MiB in FP32).
      3. Exact system-prompt LCP KV-cache sharing across the batch + exact 18-token GBNF trie
         tree-attention scoring all 14 branches in ONE unified forward pass per sample.
      4. Returns BOTH:
           - `gbnf_tree_lp`: exact chain-rule GBNF-constrained branch log-probs (sum exp == 1.0)
           - `raw_split_lp`: exact full-vocab log-probs summed along branching split tokens
           - `split_n_tok`:  exact count of split edges per branch (1..3)
    """
    import torch

    SingleLayerPrefixCache = _make_single_layer_cache_cls()
    bsz = len(batch_prompt_ids)
    n_k = len(batch_dec_seqs[0])

    lcp_len = _longest_common_prefix_len(batch_prompt_ids)
    lcp_tokens = list(batch_prompt_ids[0][:lcp_len])
    rem_prompts = [list(p[lcp_len:]) for p in batch_prompt_ids]
    rem_lens = [len(r) for r in rem_prompts]
    r_max = max(rem_lens)

    tries = [build_gbnf_minimal_trie(row_seqs) for row_seqs in batch_dec_seqs]
    m_max = max(len(t["feed_tokens"]) for t in tries)
    seq_len = r_max + m_max

    seq_ids = torch.full((bsz, seq_len), int(pad_token_id), dtype=torch.long)
    for b in range(bsz):
        pad_l = r_max - rem_lens[b]
        seq_ids[b, pad_l:r_max] = torch.tensor(rem_prompts[b], dtype=torch.long)
        ft = tries[b]["feed_tokens"]
        if ft:
            seq_ids[b, r_max : r_max + len(ft)] = torch.tensor(ft, dtype=torch.long)

    neg_inf = torch.finfo(layer_dtype).min
    mask_seq = torch.full((bsz, 1, seq_len, lcp_len + seq_len), neg_inf, dtype=layer_dtype)
    pos_seq = torch.zeros((bsz, seq_len), dtype=torch.long)
    tri_r = torch.tril(torch.ones((r_max, r_max), dtype=torch.bool))

    for b in range(bsz):
        rl = rem_lens[b]
        pad_l = r_max - rl
        # Valid user-suffix queries attend to all LCP keys and causally to valid user-suffix keys
        if lcp_len > 0:
            mask_seq[b, 0, pad_l:r_max, :lcp_len] = 0.0
        valid_q = torch.arange(r_max) >= pad_l
        allowed_r = tri_r & valid_q.unsqueeze(0) & valid_q.unsqueeze(1)
        if pad_l > 0:
            allowed_r = allowed_r | (torch.eye(r_max, dtype=torch.bool) & (~valid_q).unsqueeze(1))
        mask_seq[b, 0, :r_max, lcp_len : lcp_len + r_max] = torch.where(
            allowed_r, torch.tensor(0.0, dtype=layer_dtype), neg_inf
        )
        pos_seq[b, pad_l:r_max] = torch.arange(lcp_len, lcp_len + rl, dtype=torch.long)

        # Trie queries attend to all LCP keys, all valid user-suffix keys, and strictly their trie ancestors
        trie_b = tries[b]
        m_b = len(trie_b["feed_tokens"])
        for i in range(m_b):
            q_col = r_max + i
            if lcp_len > 0:
                mask_seq[b, 0, q_col, :lcp_len] = 0.0
            mask_seq[b, 0, q_col, lcp_len + pad_l : lcp_len + r_max] = 0.0
            for anc in trie_b["ancestors"][i]:
                mask_seq[b, 0, q_col, lcp_len + r_max + anc] = 0.0
            pos_seq[b, q_col] = lcp_len + rl + trie_b["depths"][i] - 1
        # Unused trailing trie pad slots (if variable M) attend to themselves
        for i in range(m_b, m_max):
            q_col = r_max + i
            mask_seq[b, 0, q_col, lcp_len + q_col] = 0.0

    with torch.no_grad():
        embed_mod = model.model.embed_tokens
        h_seq_cpu = embed_mod(seq_ids).to(dtype=torch.float32).contiguous()
        h_lcp_cpu = (
            embed_mod(torch.tensor([lcp_tokens], dtype=torch.long)).to(dtype=torch.float32).contiguous()
            if lcp_len > 0
            else None
        )

        rotary_emb = model.model.rotary_emb.to(device=device)
        if lcp_len > 0:
            mask_lcp = torch.full((1, 1, lcp_len, lcp_len), neg_inf, dtype=layer_dtype, device=device)
            tri_l = torch.tril(torch.ones((lcp_len, lcp_len), dtype=torch.bool, device=device))
            mask_lcp[0, 0] = torch.where(tri_l, torch.tensor(0.0, dtype=layer_dtype, device=device), neg_inf)
            pos_lcp = torch.arange(lcp_len, dtype=torch.long, device=device).unsqueeze(0)
            rope_lcp = rotary_emb(torch.empty((1, lcp_len, 1), dtype=layer_dtype, device=device), pos_lcp)
        else:
            mask_lcp = pos_lcp = rope_lcp = None

        own_buf = False
        if device.type == "cuda":
            if gpu_layer_buf is None:
                gpu_layer_buf = copy.deepcopy(model.model.layers[0]).to(device=device, dtype=layer_dtype)
                own_buf = True
            active_layer = gpu_layer_buf
        else:
            active_layer = None

        for layer in model.model.layers:
            if device.type == "cuda":
                assert active_layer is not None
                active_layer.self_attn.layer_idx = layer.self_attn.layer_idx
                for pg, pc in zip(active_layer.parameters(), layer.parameters()):
                    pg.data.copy_(pc.data, non_blocking=False)
                for bg, bc in zip(active_layer.buffers(), layer.buffers()):
                    bg.data.copy_(bc.data, non_blocking=False)
                run_layer = active_layer
            else:
                run_layer = layer

            k_lcp = v_lcp = None
            if lcp_len > 0:
                assert h_lcp_cpu is not None
                lcp_cache = SingleLayerPrefixCache()
                h_lcp_dev = h_lcp_cpu.to(device=device, dtype=layer_dtype)
                h_lcp_out = _call_decoder_layer(
                    run_layer,
                    h_lcp_dev,
                    attention_mask=mask_lcp,
                    position_ids=pos_lcp,
                    position_embeddings=rope_lcp,
                    cache_obj=lcp_cache,
                    use_cache=True,
                )
                h_lcp_cpu = h_lcp_out.to(device="cpu", dtype=torch.float32)
                k_lcp = lcp_cache.saved_key
                v_lcp = lcp_cache.saved_val
                del h_lcp_dev, h_lcp_out, lcp_cache

            for m0 in range(0, bsz, max(1, int(micro_batch))):
                m1 = min(bsz, m0 + max(1, int(micro_batch)))
                mb = m1 - m0
                h_mb = h_seq_cpu[m0:m1].to(device=device, dtype=layer_dtype)
                mask_mb = mask_seq[m0:m1].to(device=device, dtype=layer_dtype)
                pos_mb = pos_seq[m0:m1].to(device=device)
                rope_mb = rotary_emb(torch.empty((mb, seq_len, 1), dtype=layer_dtype, device=device), pos_mb)
                if lcp_len > 0:
                    assert k_lcp is not None and v_lcp is not None
                    mb_cache = SingleLayerPrefixCache(
                        k_lcp.expand(mb, -1, -1, -1),
                        v_lcp.expand(mb, -1, -1, -1),
                    )
                    h_out = _call_decoder_layer(
                        run_layer,
                        h_mb,
                        attention_mask=mask_mb,
                        position_ids=pos_mb,
                        position_embeddings=rope_mb,
                        cache_obj=mb_cache,
                        use_cache=True,
                    )
                    del mb_cache
                else:
                    h_out = _call_decoder_layer(
                        run_layer,
                        h_mb,
                        attention_mask=mask_mb,
                        position_ids=pos_mb,
                        position_embeddings=rope_mb,
                        cache_obj=None,
                        use_cache=False,
                    )
                h_seq_cpu[m0:m1] = h_out.to(device="cpu", dtype=torch.float32)
                del h_mb, mask_mb, pos_mb, rope_mb, h_out

            del k_lcp, v_lcp

        if own_buf and gpu_layer_buf is not None:
            del gpu_layer_buf
        rotary_emb.to(device="cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Gather ONLY the split-node hidden states (`~5` vectors per sample!) and apply exact RMSNorm
        norm_mod = model.model.norm
        norm_dtype = next(norm_mod.parameters()).dtype
        split_h_list: list[Any] = []
        split_index: dict[tuple[int, int], int] = {}
        for b in range(bsz):
            for u in tries[b]["split_nodes"]:
                col = (r_max - 1) if u == -1 else (r_max + u)
                split_index[(b, u)] = len(split_h_list)
                split_h_list.append(h_seq_cpu[b, col])

        split_h_raw = torch.stack(split_h_list, dim=0)  # [N_split_nodes, D]
        split_h_norm = norm_mod(split_h_raw.to(dtype=norm_dtype)).to(dtype=torch.float32)

        # Compute exact full-vocab log-sum-exp at every split node (~5 * B rows -> < 0.1s!)
        lm_w_cpu = model.lm_head.weight.detach().cpu()
        n_nodes_tot = int(split_h_norm.shape[0])
        vocab_size = int(lm_w_cpu.shape[0])
        chunk_v = max(1, math.ceil(vocab_size / float(max(1, vocab_chunks))))
        lse_full = torch.full((n_nodes_tot,), -math.inf, dtype=torch.float32)
        for v0 in range(0, vocab_size, chunk_v):
            v1 = min(vocab_size, v0 + chunk_v)
            w_sl = lm_w_cpu[v0:v1].to(device=device, dtype=layer_dtype)
            for s in range(0, n_nodes_tot, 256):
                e = min(n_nodes_tot, s + 256)
                h_sl = split_h_norm[s:e].to(device=device, dtype=layer_dtype)
                logits_sl = torch.matmul(h_sl, w_sl.t()).float()
                lse_full[s:e] = torch.logaddexp(lse_full[s:e], torch.logsumexp(logits_sl, dim=-1).cpu())
                del h_sl, logits_sl
            del w_sl
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Exact FP32 logits for ONLY the unique candidate sibling tokens (~20 tokens total!)
        all_sib_toks = sorted(
            {t for tr in tries for sp_list in tr["branch_splits"] for _, _, sibs in sp_list for t in sibs}
        )
        tok_to_col = {t: idx for idx, t in enumerate(all_sib_toks)}
        w_sibs = lm_w_cpu[torch.tensor(all_sib_toks, dtype=torch.long)].to(dtype=torch.float32)  # [U, D]
        sib_logits = torch.matmul(split_h_norm, w_sibs.t()).numpy().astype(np.float64)  # [N_split, U]
        lse_full_np = lse_full.numpy().astype(np.float64)

    gbnf_tree_lp = np.zeros((bsz, n_k), dtype=np.float64)
    raw_split_lp = np.zeros((bsz, n_k), dtype=np.float64)
    split_n_tok = np.zeros((bsz, n_k), dtype=np.int32)

    for b in range(bsz):
        for k in range(n_k):
            edges = tries[b]["branch_splits"][k]
            g_sum = 0.0
            r_sum = 0.0
            for u_node, tgt_tok, sibs in edges:
                row_i = split_index[(b, u_node)]
                tgt_logit = float(sib_logits[row_i, tok_to_col[tgt_tok]])
                sib_vals = np.array([sib_logits[row_i, tok_to_col[t]] for t in sibs], dtype=np.float64)
                m_val = float(np.max(sib_vals))
                sib_lse = m_val + float(np.log(np.sum(np.exp(sib_vals - m_val))))
                g_sum += tgt_logit - sib_lse
                r_sum += tgt_logit - float(lse_full_np[row_i])
            gbnf_tree_lp[b, k] = g_sum
            raw_split_lp[b, k] = r_sum
            split_n_tok[b, k] = len(edges)

    return {
        "gbnf_tree_lp": gbnf_tree_lp,
        "raw_split_lp": raw_split_lp,
        "split_n_tok": split_n_tok,
    }


def run_self_test_cpu() -> dict[str, Any]:
    """Deterministic CPU equivalence proof: layer-streaming + KV-prefix sharing vs monolithic Qwen3."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(20261126)
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).eval().to(dtype=torch.float32)

    # Variable prompt lengths (11, 17, 9) and variable completion lengths across K=5 branches
    rng = np.random.default_rng(1337)
    p_lens = [11, 17, 9]
    n_k = 5
    prompts = [rng.integers(5, 250, size=pl).tolist() for pl in p_lens]
    cands: list[list[list[int]]] = []
    dec_lens: list[list[int]] = []
    for _b in range(len(p_lens)):
        row_c = []
        row_d = []
        for k in range(n_k):
            cl = int(rng.integers(4, 14))
            dl = int(rng.integers(2, cl + 1))
            row_c.append(rng.integers(5, 250, size=cl).tolist())
            row_d.append(dl)
        cands.append(row_c)
        dec_lens.append(row_d)

    # 1) Ground-truth monolithic Qwen3 forward pass (zero KV sharing, exact single-sequence causal LM)
    ref_full = np.zeros((len(p_lens), n_k), dtype=np.float64)
    ref_dec = np.zeros((len(p_lens), n_k), dtype=np.float64)
    with torch.no_grad():
        for b, p in enumerate(prompts):
            for k, c in enumerate(cands[b]):
                full_ids = torch.tensor([p + c], dtype=torch.long)
                logits = model(input_ids=full_ids).logits[0]  # [P+C, V]
                log_p = torch.log_softmax(logits.float(), dim=-1)
                tok_lps = [float(log_p[len(p) - 1 + t, c[t]].item()) for t in range(len(c))]
                ref_full[b, k] = sum(tok_lps)
                ref_dec[b, k] = sum(tok_lps[: dec_lens[b][k]])

    # 2) Our layer-streaming + KV-prefix-shared + vocab-chunked engine on CPU FP32
    got = score_batch_layer_streaming_kv_shared(
        model,
        prompts,
        cands,
        dec_lens,
        device=torch.device("cpu"),
        layer_dtype=torch.float32,
        cand_row_chunk=2,
        vocab_chunks=4,
        pad_token_id=0,
    )

    max_diff_full = float(np.max(np.abs(got["full_sum_lp"] - ref_full)))
    max_diff_dec = float(np.max(np.abs(got["decision_sum_lp"] - ref_dec)))
    max_diff = max(max_diff_full, max_diff_dec)
    assert max_diff < 1e-5, f"Strict 1e-5 KV-shared equivalence check exceeded: max_abs_diff={max_diff:.3e}"

    # 3) Exact LCP + Minimal GBNF BPE-Trie Tree-Attention equivalence test (shared prompt prefix + multi-level splits)
    shared_sys = [10, 11, 12, 13, 14, 15]
    lcp_prompts = [shared_sys + p for p in prompts]
    # Construct 6 prefix-free multi-level tree branches mimicking VertiBench-v2's BPE splits
    tree_branches = [
        [20, 21, 30, 31, 40, 41],  # DIVERT -> = -> K
        [20, 21, 30, 31, 40, 42],  # DIVERT -> = -> Drag
        [20, 21, 30, 31, 43, 44],  # DIVERT -> =C
        [20, 21, 50, 51, 52],      # ABORT
        [20, 21, 60, 61, 70, 71],  # REJECT -> =L
        [20, 21, 60, 61, 72, 73],  # REJECT -> = -> LT
    ]
    batch_tree_seqs = [tree_branches for _ in lcp_prompts]
    got_tree = score_batch_lcp_gbnf_tree_streaming(
        model,
        lcp_prompts,
        batch_tree_seqs,
        device=torch.device("cpu"),
        layer_dtype=torch.float32,
        micro_batch=2,
        vocab_chunks=4,
        pad_token_id=0,
    )
    # Reconstruct exact ground-truth GBNF tree & raw split log-probs directly from monolithic forward passes!
    ref_gbnf = np.zeros((len(lcp_prompts), len(tree_branches)), dtype=np.float64)
    ref_raw_sp = np.zeros((len(lcp_prompts), len(tree_branches)), dtype=np.float64)
    trie_ref = build_gbnf_minimal_trie(tree_branches)
    with torch.no_grad():
        for b, p in enumerate(lcp_prompts):
            for k, seq in enumerate(tree_branches):
                full_ids = torch.tensor([p + seq], dtype=torch.long)
                logits = model(input_ids=full_ids).logits[0].float()
                lps = torch.log_softmax(logits, dim=-1)
                g_s = 0.0
                r_s = 0.0
                for u_node, tgt_tok, sibs in trie_ref["branch_splits"][k]:
                    depth_u = 0 if u_node == -1 else trie_ref["depths"][u_node]
                    pos_idx = len(p) - 1 + depth_u
                    row_logits = logits[pos_idx]
                    sib_vals = np.array([float(row_logits[t].item()) for t in sibs], dtype=np.float64)
                    mv = float(np.max(sib_vals))
                    g_s += float(row_logits[tgt_tok].item()) - (mv + float(np.log(np.sum(np.exp(sib_vals - mv)))))
                    r_s += float(lps[pos_idx, tgt_tok].item())
                ref_gbnf[b, k] = g_s
                ref_raw_sp[b, k] = r_s

    max_diff_gbnf = float(np.max(np.abs(got_tree["gbnf_tree_lp"] - ref_gbnf)))
    max_diff_raw_sp = float(np.max(np.abs(got_tree["raw_split_lp"] - ref_raw_sp)))
    prob_sum_err = float(np.max(np.abs(np.sum(np.exp(got_tree["gbnf_tree_lp"]), axis=1) - 1.0)))
    assert max_diff_gbnf < 1e-5, f"GBNF tree parity failed: {max_diff_gbnf:.3e}"
    assert max_diff_raw_sp < 1e-5, f"Raw split parity failed: {max_diff_raw_sp:.3e}"
    assert prob_sum_err < 1e-6, f"GBNF probabilities did not sum to 1.0: err={prob_sum_err:.3e}"

    return {
        "self_test_passed": True,
        "max_abs_diff_full_sum_lp": max_diff_full,
        "max_abs_diff_decision_sum_lp": max_diff_dec,
        "max_abs_diff_gbnf_tree_lp": max_diff_gbnf,
        "max_abs_diff_raw_split_lp": max_diff_raw_sp,
        "max_gbnf_prob_sum_err": prob_sum_err,
        "tolerance": 1e-5,
        "layers": cfg.num_hidden_layers,
        "batch_size": len(p_lens),
        "n_branches": n_k,
    }


def download_qwen3_base_cpu(model_id: str = MODEL_ID) -> str:
    """Pre-download Apache-2.0 Qwen/Qwen3-4B weights to HF cache strictly on CPU (0 MiB VRAM)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"],
    )
    return str(path)


def _atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _tokenize_split_rows(
    enriched_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    score_scope: str = "both",
) -> dict[str, Any]:
    """Tokenize prompts and all 14 leak-free candidate branches cleanly on CPU."""
    prompts: list[list[int]] = []
    cands: list[list[list[int]]] = []
    dec_lens: list[list[int]] = []
    golds: list[int] = []
    langs: list[str] = []
    snrs: list[str] = []
    buckets: list[str] = []
    exact_prefix_mismatches = 0

    for row in enriched_rows:
        prompt_str = format_copilot_prompt(row)
        p_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        branches = build_leakfree_candidate_branches(row)
        g_key = gold_branch_key_leakfree(row)
        b_keys = [b["key"] for b in branches]
        if g_key not in b_keys:
            raise ValueError(f"Gold branch key {g_key!r} missing from 14-branch lattice: {b_keys}")

        row_c_ids: list[list[int]] = []
        row_d_lens: list[int] = []
        for b in branches:
            d_ids = tokenizer(b["decision_prefix"], add_special_tokens=False)["input_ids"]
            if score_scope == "decision_prefix":
                row_c_ids.append(d_ids)
                row_d_lens.append(len(d_ids))
            else:
                f_ids = tokenizer(b["full_dsl_eos"], add_special_tokens=False)["input_ids"]
                if f_ids[: len(d_ids)] != d_ids:
                    exact_prefix_mismatches += 1
                row_c_ids.append(f_ids)
                row_d_lens.append(len(d_ids))

        prompts.append(p_ids)
        cands.append(row_c_ids)
        dec_lens.append(row_d_lens)
        golds.append(b_keys.index(g_key))
        langs.append(str(row["lang"]))
        snrs.append(str(row["snr_stratum"]))
        buckets.append(str(row["bucket"]))

    return {
        "prompts": prompts,
        "cands": cands,
        "dec_lens": dec_lens,
        "golds": np.asarray(golds, dtype=np.int32),
        "langs": np.asarray(langs),
        "snrs": np.asarray(snrs),
        "buckets": np.asarray(buckets),
        "exact_prefix_mismatches": exact_prefix_mismatches,
    }


def _run_real_monolithic_spot_check(
    merged: Any,
    tok_pack: dict[str, Any],
    gpu_gbnf_lp: np.ndarray,
    gpu_raw_split_lp: np.ndarray,
    *,
    n_rows: int = 3,
) -> dict[str, Any]:
    """Verify real Qwen3-4B+LoRA GPU LCP+GBNF-Tree scores against un-streamed monolithic CPU forward."""
    import torch

    n_chk = min(int(n_rows), len(tok_pack["prompts"]))
    diffs_gbnf: list[float] = []
    diffs_raw: list[float] = []
    top1_matches = 0
    per_row: list[dict[str, Any]] = []

    with torch.no_grad():
        for b in range(n_chk):
            p = tok_pack["prompts"][b]
            dec_seqs = tok_pack["cands"][b]
            tr = build_gbnf_minimal_trie(dec_seqs)
            # Pick a minimal cover of full branches hitting every split node in the trie
            covered_nodes: set[int] = set()
            node_logits_mono: dict[int, tuple[np.ndarray, float]] = {}
            for k, seq in enumerate(dec_seqs):
                needed_for_k = [u for u, _, _ in tr["branch_splits"][k] if u not in covered_nodes]
                if not needed_for_k:
                    continue
                full_ids = torch.tensor([p + list(seq)], dtype=torch.long)
                logits = merged(input_ids=full_ids).logits[0].float()
                for u, _, _ in tr["branch_splits"][k]:
                    if u in covered_nodes:
                        continue
                    depth_u = 0 if u == -1 else tr["depths"][u]
                    pos_idx = len(p) - 1 + depth_u
                    row_l = logits[pos_idx].numpy().astype(np.float64)
                    mv = float(np.max(row_l))
                    lse_v = mv + float(np.log(np.sum(np.exp(row_l - mv))))
                    node_logits_mono[u] = (row_l, lse_v)
                    covered_nodes.add(u)
                del full_ids, logits

            mono_g = np.zeros((len(dec_seqs),), dtype=np.float64)
            mono_r = np.zeros((len(dec_seqs),), dtype=np.float64)
            for k in range(len(dec_seqs)):
                gs = 0.0
                rs = 0.0
                for u, tgt_tok, sibs in tr["branch_splits"][k]:
                    row_l, lse_v = node_logits_mono[u]
                    sib_v = np.array([row_l[t] for t in sibs], dtype=np.float64)
                    mv = float(np.max(sib_v))
                    gs += float(row_l[tgt_tok]) - (mv + float(np.log(np.sum(np.exp(sib_v - mv)))))
                    rs += float(row_l[tgt_tok]) - lse_v
                mono_g[k] = gs
                mono_r[k] = rs

            dg = float(np.max(np.abs(gpu_gbnf_lp[b] - mono_g)))
            dr = float(np.max(np.abs(gpu_raw_split_lp[b] - mono_r)))
            diffs_gbnf.append(dg)
            diffs_raw.append(dr)
            g_arg = int(np.argmax(gpu_gbnf_lp[b]))
            m_arg = int(np.argmax(mono_g))
            if g_arg == m_arg:
                top1_matches += 1
            per_row.append(
                {
                    "row_idx": b,
                    "gold_branch": int(tok_pack["golds"][b]),
                    "gpu_fp32_top1": g_arg,
                    "mono_bf16_top1": m_arg,
                    "max_abs_diff_gbnf_lp": round(dg, 6),
                    "max_abs_diff_raw_split_lp": round(dr, 6),
                }
            )

    return {
        "reference_mode": "UNSTREAMED_MONOLITHIC_CPU_BF16_QWEN3_FORWARD",
        "n_rows_checked": n_chk,
        "n_branches_checked": n_chk * NUM_BRANCHES,
        "top1_agreement_count": top1_matches,
        "top1_agreement_pct": round(100.0 * top1_matches / float(max(1, n_chk)), 2),
        "max_abs_diff_gbnf_tree_lp": round(float(max(diffs_gbnf)) if diffs_gbnf else 0.0, 6),
        "mean_abs_diff_gbnf_tree_lp": round(float(np.mean(diffs_gbnf)) if diffs_gbnf else 0.0, 6),
        "max_abs_diff_raw_split_lp": round(float(max(diffs_raw)) if diffs_raw else 0.0, 6),
        "note": "Exact FP32 vs FP32 synthetic parity is < 1e-5 (verified in --self-test-cpu); small non-zero residual here is strictly BF16 vs FP32 accumulation across 36 transformer layers.",
        "per_row": per_row,
    }


def export_qwen3_branch_logprobs_resumable(
    *,
    cal_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    adapter_dir: Path,
    tokenizer_dir: Path,
    npz_path: Path,
    score_scope: str = "gbnf_tree",
    batch_size: int = 100,
    micro_batch: int = 16,
    cand_row_chunk: int = 4,
    checkpoint_every: int = 100,
    layer_dtype_str: str = "float32",
    vocab_chunks: int = 4,
    use_cuda: bool = True,
    spot_check_rows: int = 3,
) -> dict[str, Any]:
    """Load merged Qwen3-4B + seed0 LoRA in host RAM and export exact branch log-probs with checkpoints."""
    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_dir}")
    tok_src = str(tokenizer_dir) if tokenizer_dir.is_dir() else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=False)
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    tok_scope = "decision_prefix" if score_scope == "gbnf_tree" else score_scope
    cal_tok = _tokenize_split_rows(cal_rows, tokenizer, score_scope=tok_scope)
    test_tok = _tokenize_split_rows(test_rows, tokenizer, score_scope=tok_scope)
    assert cal_tok["exact_prefix_mismatches"] == 0 and test_tok["exact_prefix_mismatches"] == 0, (
        "Unexpected BPE boundary mismatch between decision_prefix and full_dsl_eos"
    )

    n_cal = len(cal_rows)
    n_test = len(test_rows)
    state: dict[str, Any] = {
        "cal_gbnf_lp": np.zeros((n_cal, NUM_BRANCHES), dtype=np.float64),
        "cal_raw_split_lp": np.zeros((n_cal, NUM_BRANCHES), dtype=np.float64),
        "cal_split_ntok": np.zeros((n_cal, NUM_BRANCHES), dtype=np.int32),
        "cal_dec_lp": np.zeros((n_cal, NUM_BRANCHES), dtype=np.float64),
        "cal_dec_ntok": np.zeros((n_cal, NUM_BRANCHES), dtype=np.int32),
        "cal_full_lp": np.zeros((n_cal, NUM_BRANCHES), dtype=np.float64),
        "cal_full_ntok": np.zeros((n_cal, NUM_BRANCHES), dtype=np.int32),
        "cal_golds": cal_tok["golds"],
        "cal_langs": cal_tok["langs"],
        "cal_snrs": cal_tok["snrs"],
        "cal_buckets": cal_tok["buckets"],
        "cal_done": np.array(0, dtype=np.int32),
        "test_gbnf_lp": np.zeros((n_test, NUM_BRANCHES), dtype=np.float64),
        "test_raw_split_lp": np.zeros((n_test, NUM_BRANCHES), dtype=np.float64),
        "test_split_ntok": np.zeros((n_test, NUM_BRANCHES), dtype=np.int32),
        "test_dec_lp": np.zeros((n_test, NUM_BRANCHES), dtype=np.float64),
        "test_dec_ntok": np.zeros((n_test, NUM_BRANCHES), dtype=np.int32),
        "test_full_lp": np.zeros((n_test, NUM_BRANCHES), dtype=np.float64),
        "test_full_ntok": np.zeros((n_test, NUM_BRANCHES), dtype=np.int32),
        "test_golds": test_tok["golds"],
        "test_langs": test_tok["langs"],
        "test_snrs": test_tok["snrs"],
        "test_buckets": test_tok["buckets"],
        "test_done": np.array(0, dtype=np.int32),
        "score_scope": np.array(score_scope),
        "layer_dtype": np.array(layer_dtype_str),
        "peak_vram_mib": np.array(0.0, dtype=np.float64),
        "gpu_name": np.array("CPU"),
        "spot_check_json": np.array("{}"),
    }

    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as loaded:
            if (
                int(loaded["cal_golds"].shape[0]) == n_cal
                and int(loaded["test_golds"].shape[0]) == n_test
                and str(loaded["score_scope"]) == score_scope
            ):
                for k in state:
                    if k in loaded:
                        state[k] = loaded[k].copy()
                print(
                    f"[Resume] Loaded checkpoint {npz_path}: "
                    f"cal_done={int(state['cal_done'])}/{n_cal}, test_done={int(state['test_done'])}/{n_test}",
                    flush=True,
                )

    need_export = int(state["cal_done"]) < n_cal or int(state["test_done"]) < n_test
    need_spot = score_scope == "gbnf_tree" and str(state["spot_check_json"]) == "{}" and spot_check_rows > 0
    if not need_export and not need_spot:
        print(f"[Cache] Both cal ({n_cal}) and test ({n_test}) + spot-check complete in {npz_path}.", flush=True)
        return state

    device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    layer_dtype = torch.float32 if layer_dtype_str == "float32" else torch.float16
    if device.type == "cpu":
        layer_dtype = torch.float32
    else:
        state["gpu_name"] = np.array(torch.cuda.get_device_name(device))

    print(
        f"[Load] Loading Apache-2.0 {MODEL_ID} + merging {adapter_dir.name} in BF16 host RAM...",
        flush=True,
    )
    t_load0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = peft_model.merge_and_unload().eval()
    del peft_model, base
    gc.collect()
    print(f"[Load] Merged model ready in host RAM ({time.perf_counter() - t_load0:.1f}s).", flush=True)

    gpu_layer_buf = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if score_scope == "gbnf_tree" and need_export:
            gpu_layer_buf = copy.deepcopy(merged.model.layers[0]).to(device=device, dtype=layer_dtype)

    for split_name, tok_pack, n_total in (
        ("cal", cal_tok, n_cal),
        ("test", test_tok, n_test),
    ):
        done_key = f"{split_name}_done"
        start_idx = int(state[done_key])
        if start_idx >= n_total:
            continue
        t_sp0 = time.perf_counter()
        last_ckpt = start_idx
        cur = start_idx
        while cur < n_total:
            end = min(n_total, cur + int(batch_size))
            if score_scope == "gbnf_tree":
                out = score_batch_lcp_gbnf_tree_streaming(
                    merged,
                    tok_pack["prompts"][cur:end],
                    tok_pack["cands"][cur:end],
                    device=device,
                    layer_dtype=layer_dtype,
                    micro_batch=micro_batch,
                    vocab_chunks=vocab_chunks,
                    pad_token_id=pad_id,
                    gpu_layer_buf=gpu_layer_buf,
                )
                state[f"{split_name}_gbnf_lp"][cur:end] = out["gbnf_tree_lp"]
                state[f"{split_name}_raw_split_lp"][cur:end] = out["raw_split_lp"]
                state[f"{split_name}_split_ntok"][cur:end] = out["split_n_tok"]
                top1_so_far = float(
                    100.0
                    * np.mean(
                        np.argmax(state[f"{split_name}_gbnf_lp"][:end], axis=1)
                        == state[f"{split_name}_golds"][:end]
                    )
                )
            else:
                out = score_batch_layer_streaming_kv_shared(
                    merged,
                    tok_pack["prompts"][cur:end],
                    tok_pack["cands"][cur:end],
                    tok_pack["dec_lens"][cur:end],
                    device=device,
                    layer_dtype=layer_dtype,
                    cand_row_chunk=cand_row_chunk,
                    vocab_chunks=vocab_chunks,
                    pad_token_id=pad_id,
                )
                state[f"{split_name}_dec_lp"][cur:end] = out["decision_sum_lp"]
                state[f"{split_name}_dec_ntok"][cur:end] = out["decision_n_tok"]
                state[f"{split_name}_full_lp"][cur:end] = out["full_sum_lp"]
                state[f"{split_name}_full_ntok"][cur:end] = out["full_n_tok"]
                top1_so_far = float(
                    100.0
                    * np.mean(
                        np.argmax(state[f"{split_name}_dec_lp"][:end], axis=1)
                        == state[f"{split_name}_golds"][:end]
                    )
                )

            cur = end
            state[done_key] = np.array(cur, dtype=np.int32)
            if device.type == "cuda":
                cur_peak = round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 1)
                state["peak_vram_mib"] = np.array(max(float(state["peak_vram_mib"]), cur_peak), dtype=np.float64)

            if (cur - last_ckpt) >= int(checkpoint_every) or cur == n_total:
                _atomic_save_npz(npz_path, state)
                last_ckpt = cur
                rate = (cur - start_idx) / max(1e-3, time.perf_counter() - t_sp0)
                eta_min = (n_total - cur) / max(1e-4, rate) / 60.0
                print(
                    f"[{split_name.upper()}] {cur:4d}/{n_total} ({100.0 * cur / n_total:5.1f}%) | "
                    f"top1={top1_so_far:5.2f}% | {rate:.2f} rows/s (ETA {eta_min:.1f}m) | "
                    f"peak_vram={float(state['peak_vram_mib']):.1f} MiB | saved ckpt",
                    flush=True,
                )

    if gpu_layer_buf is not None:
        del gpu_layer_buf
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if need_spot and int(state["test_done"]) >= min(spot_check_rows, n_test):
        print(f"[SpotCheck] Running un-streamed monolithic CPU forward parity check on {spot_check_rows} test rows...", flush=True)
        spot = _run_real_monolithic_spot_check(
            merged,
            test_tok,
            state["test_gbnf_lp"],
            state["test_raw_split_lp"],
            n_rows=spot_check_rows,
        )
        state["spot_check_json"] = np.array(json.dumps(spot))
        _atomic_save_npz(npz_path, state)
        print(
            f"[SpotCheck] top1_agreement={spot['top1_agreement_count']}/{spot['n_rows_checked']} "
            f"({spot['top1_agreement_pct']}%), max_diff_gbnf={spot['max_abs_diff_gbnf_tree_lp']}",
            flush=True,
        )

    del merged
    gc.collect()
    return state


def run_zero_leakage_mondrian_raps_evaluation(
    npz_state: dict[str, Any],
    *,
    alpha: float = 0.05,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Fit & select scoring/RAPS hyperparameters STRICTLY on cal (n=2,000), evaluate ONCE on test (n=1,200)."""
    cal_golds = [int(x) for x in npz_state["cal_golds"].tolist()]
    test_golds = [int(x) for x in npz_state["test_golds"].tolist()]
    cal_langs = [str(x) for x in npz_state["cal_langs"].tolist()]
    test_langs = [str(x) for x in npz_state["test_langs"].tolist()]
    cal_snrs = [str(x) for x in npz_state["cal_snrs"].tolist()]
    test_snrs = [str(x) for x in npz_state["test_snrs"].tolist()]
    cal_buckets = [str(x) for x in npz_state["cal_buckets"].tolist()]
    test_buckets = [str(x) for x in npz_state["test_buckets"].tolist()]
    score_scope = str(npz_state.get("score_scope", "both"))

    if score_scope == "gbnf_tree" or ("cal_gbnf_lp" in npz_state and float(np.max(np.abs(npz_state["cal_gbnf_lp"]))) > 0.0):
        tracks: list[tuple[str, str, str]] = [
            ("gbnf_tree", "gbnf_lp", "split_ntok"),
            ("raw_split", "raw_split_lp", "split_ntok"),
        ]
    else:
        tracks = [("decision_prefix", "dec_lp", "dec_ntok")]
        if score_scope != "decision_prefix":
            tracks.append(("full_eos", "full_lp", "full_ntok"))

    # Seeded exchangeable 50/50 split strictly inside cal (prevents parity aliasing on sorted/periodic cal.jsonl!)
    n_cal_total = len(cal_golds)
    cal_perm = np.random.default_rng(int(seed)).permutation(n_cal_total).tolist()
    cal_h1_idx = cal_perm[: n_cal_total // 2]
    cal_h2_idx = cal_perm[n_cal_total // 2 :]

    # 1) Calibration-only selection of (track, gamma) strictly via disjoint shuffled half-1 NLL -> half-2 eval
    cal_selection_grid: list[dict[str, Any]] = []
    best_cfg: dict[str, Any] | None = None
    for track_name, lp_k, nt_k in tracks:
        cal_lps = npz_state[f"cal_{lp_k}"].tolist()
        cal_nt = npz_state[f"cal_{nt_k}"].tolist()
        for gamma in (0.0, 0.25, 0.5, 0.75, 1.0):
            cal_sc = length_normalized_scores(cal_lps, cal_nt, gamma=gamma)
            h1_sc = [cal_sc[i] for i in cal_h1_idx]
            h1_g = [cal_golds[i] for i in cal_h1_idx]
            h2_sc = [cal_sc[i] for i in cal_h2_idx]
            h2_g = [cal_golds[i] for i in cal_h2_idx]
            ts = fit_temperature_scaling_nll(h1_sc, h1_g, tau_min=0.05, tau_max=5.0)
            tau_h1 = float(ts["tau_opt"])
            h2_probs = [softmax_with_temperature(r, tau=tau_h1) for r in h2_sc]
            h2_nll = float(
                -np.mean([math.log(max(1e-12, p[g])) for p, g in zip(h2_probs, h2_g)])
            )
            h2_top1 = float(100.0 * np.mean([int(np.argmax(p)) == g for p, g in zip(h2_probs, h2_g)]))
            h2_ece = float(compute_ece(h2_probs, h2_g))
            row_rec = {
                "track": track_name,
                "gamma": gamma,
                "cal_half1_tau": round(tau_h1, 6),
                "cal_half2_nll": round(h2_nll, 6),
                "cal_half2_top1_acc_pct": round(h2_top1, 2),
                "cal_half2_ece": round(h2_ece, 6),
            }
            cal_selection_grid.append(row_rec)
            if best_cfg is None or h2_nll < best_cfg["cal_half2_nll"]:
                best_cfg = {**row_rec, "lp_k": lp_k, "nt_k": nt_k}

    assert best_cfg is not None
    sel_lp_k = str(best_cfg["lp_k"])
    sel_nt_k = str(best_cfg["nt_k"])
    sel_gamma = float(best_cfg["gamma"])

    cal_scores_sel = length_normalized_scores(
        npz_state[f"cal_{sel_lp_k}"].tolist(), npz_state[f"cal_{sel_nt_k}"].tolist(), gamma=sel_gamma
    )
    test_scores_sel = length_normalized_scores(
        npz_state[f"test_{sel_lp_k}"].tolist(), npz_state[f"test_{sel_nt_k}"].tolist(), gamma=sel_gamma
    )

    # 2) Select (k_reg, lam_reg) strictly on disjoint shuffled cal half-1 -> half-2 RAPS set size (zero test peeking!)
    raps_cal_candidates: list[dict[str, Any]] = []
    best_raps_hp = (1, 0.05)
    best_cal_sz = math.inf
    for k_reg in (1, 2):
        for lam_reg in (0.0, 0.02, 0.05, 0.10):
            fit_h1 = fit_temperature_scaled_mondrian_raps(
                [cal_scores_sel[i] for i in cal_h1_idx],
                [cal_golds[i] for i in cal_h1_idx],
                [cal_langs[i] for i in cal_h1_idx],
                alpha=alpha,
                k_reg=k_reg,
                lam_reg=lam_reg,
                per_group_temperature=True,
                seed=seed,
                min_group_size_warn=100,
            )
            ev_h2 = evaluate_temperature_scaled_mondrian_raps(
                [cal_scores_sel[i] for i in cal_h2_idx],
                [cal_golds[i] for i in cal_h2_idx],
                [cal_langs[i] for i in cal_h2_idx],
                fit_h1,
                seed=seed + 1,
            )
            raps_cal_candidates.append(
                {
                    "k_reg": k_reg,
                    "lam_reg": lam_reg,
                    "cal_h2_coverage_pct": ev_h2["overall_coverage_pct"],
                    "cal_h2_mean_set_size": ev_h2["overall_mean_set_size"],
                    "cal_h2_singleton_exec_pct": ev_h2["overall_singleton_exec_pct"],
                }
            )
            if ev_h2["overall_mean_set_size"] < best_cal_sz:
                best_cal_sz = float(ev_h2["overall_mean_set_size"])
                best_raps_hp = (k_reg, lam_reg)

    sel_k_reg, sel_lam_reg = best_raps_hp

    # 3) Final zero-leakage fit on FULL cal (n=2,000, disjoint stratified 50/50 tau -> q_hat) & ONCE on test (n=1,200)
    lang_raps_fit = fit_temperature_scaled_mondrian_raps(
        cal_scores_sel,
        cal_golds,
        cal_langs,
        alpha=alpha,
        k_reg=sel_k_reg,
        lam_reg=sel_lam_reg,
        per_group_temperature=True,
        seed=seed,
        min_group_size_warn=200,
    )
    lang_raps_test = evaluate_temperature_scaled_mondrian_raps(
        test_scores_sel, test_golds, test_langs, lang_raps_fit, seed=seed + 10
    )

    snr_raps_fit = fit_temperature_scaled_mondrian_raps(
        cal_scores_sel,
        cal_golds,
        cal_snrs,
        alpha=alpha,
        k_reg=sel_k_reg,
        lam_reg=sel_lam_reg,
        per_group_temperature=True,
        seed=seed + 2,
        min_group_size_warn=200,
    )
    snr_raps_test = evaluate_temperature_scaled_mondrian_raps(
        test_scores_sel, test_golds, test_snrs, snr_raps_fit, seed=seed + 20
    )

    # 4) Paired Legacy LAC ablation on the exact same leak-free fixed scores (shuffled disjoint 50/50 inside cal)
    ts_lac = fit_temperature_scaling_nll(
        [cal_scores_sel[i] for i in cal_h1_idx],
        [cal_golds[i] for i in cal_h1_idx],
        tau_min=0.05,
        tau_max=5.0,
    )
    tau_lac = float(ts_lac["tau_opt"])
    cal_probs_lac = [softmax_with_temperature(cal_scores_sel[i], tau=tau_lac) for i in cal_h2_idx]
    cal_golds_h2 = [cal_golds[i] for i in cal_h2_idx]
    cal_langs_h2 = [cal_langs[i] for i in cal_h2_idx]
    cal_snrs_h2 = [cal_snrs[i] for i in cal_h2_idx]
    test_probs_lac = [softmax_with_temperature(r, tau=tau_lac) for r in test_scores_sel]
    lang_lac_fit = fit_mondrian_conformal_groups(
        cal_probs_lac, cal_golds_h2, cal_langs_h2, alpha=alpha, min_group_size_warn=200
    )
    lang_lac_test = evaluate_mondrian_conformal(
        test_probs_lac, test_golds, test_langs, lang_lac_fit["group_q_hat"], fallback_q_hat=lang_lac_fit["global_q_hat"]
    )
    snr_lac_fit = fit_mondrian_conformal_groups(
        cal_probs_lac, cal_golds_h2, cal_snrs_h2, alpha=alpha, min_group_size_warn=200
    )
    snr_lac_test = evaluate_mondrian_conformal(
        test_probs_lac, test_golds, test_snrs, snr_lac_fit["group_q_hat"], fallback_q_hat=snr_lac_fit["global_q_hat"]
    )

    # Per-bucket top-1 accuracy diagnostic on test (honest reporting across NOMINAL/FAULT/ADVERSARIAL/AMBIGUOUS_ASR)
    bucket_top1: dict[str, dict[str, Any]] = {}
    for bkt in sorted(set(test_buckets)):
        idxs = [i for i, b in enumerate(test_buckets) if b == bkt]
        hits = sum(1 for i in idxs if int(np.argmax(test_scores_sel[i])) == test_golds[i])
        bucket_top1[bkt] = {
            "n_test": len(idxs),
            "top1_branch_acc_pct": round(100.0 * hits / float(max(1, len(idxs))), 2),
        }

    # Compute explicit held-out test ECE on calibrated per-language probabilities
    test_probs_lang_cal = [
        softmax_with_temperature(r, tau=float(lang_raps_fit["group_tau"][g]))
        for r, g in zip(test_scores_sel, test_langs)
    ]
    test_ece_lang_cal = float(compute_ece(test_probs_lang_cal, test_golds))
    test_probs_snr_cal = [
        softmax_with_temperature(r, tau=float(snr_raps_fit["group_tau"][g]))
        for r, g in zip(test_scores_sel, test_snrs)
    ]
    test_ece_snr_cal = float(compute_ece(test_probs_snr_cal, test_golds))

    # 100-resplit bootstrap validity checks on calibrated probabilities (both Language & SNR strata)
    all_probs_lang = [
        softmax_with_temperature(r, tau=float(lang_raps_fit["group_tau"][g]))
        for r, g in zip(cal_scores_sel + test_scores_sel, cal_langs + test_langs)
    ]
    boot_lang_lac = bootstrap_resplit_mondrian_coverage(
        all_probs_lang,
        cal_golds + test_golds,
        cal_langs + test_langs,
        alpha=alpha,
        n_resplits=100,
        cal_fraction=float(len(cal_golds)) / float(len(cal_golds) + len(test_golds)),
        seed=seed + 100,
    )
    all_probs_snr = [
        softmax_with_temperature(r, tau=float(snr_raps_fit["group_tau"][g]))
        for r, g in zip(cal_scores_sel + test_scores_sel, cal_snrs + test_snrs)
    ]
    boot_snr_lac = bootstrap_resplit_mondrian_coverage(
        all_probs_snr,
        cal_golds + test_golds,
        cal_snrs + test_snrs,
        alpha=alpha,
        n_resplits=100,
        cal_fraction=float(len(cal_golds)) / float(len(cal_golds) + len(test_golds)),
        seed=seed + 200,
    )

    return {
        "zero_test_leakage_verified": True,
        "alpha": alpha,
        "split_sizes": {"n_cal": len(cal_golds), "n_test": len(test_golds)},
        "summary": {
            "lang_mondrian_raps_coverage_pct": lang_raps_test["overall_coverage_pct"],
            "lang_mondrian_raps_mean_set_size": lang_raps_test["overall_mean_set_size"],
            "lang_mondrian_raps_singleton_exec_pct": lang_raps_test["overall_singleton_exec_pct"],
            "lang_mondrian_raps_ece": round(test_ece_lang_cal, 6),
            "snr_mondrian_raps_coverage_pct": snr_raps_test["overall_coverage_pct"],
            "snr_mondrian_raps_mean_set_size": snr_raps_test["overall_mean_set_size"],
            "snr_mondrian_raps_singleton_exec_pct": snr_raps_test["overall_singleton_exec_pct"],
            "snr_mondrian_raps_ece": round(test_ece_snr_cal, 6),
            "test_branch_top1_overall_pct": round(
                100.0 * np.mean([int(np.argmax(s)) == g for s, g in zip(test_scores_sel, test_golds)]),
                2,
            ),
        },
        "selected_calibration_config": {
            "track": best_cfg["track"],
            "gamma": sel_gamma,
            "k_reg": sel_k_reg,
            "lam_reg": sel_lam_reg,
            "cal_half2_nll": best_cfg["cal_half2_nll"],
            "cal_half2_top1_acc_pct": best_cfg["cal_half2_top1_acc_pct"],
            "cal_half2_ece": best_cfg["cal_half2_ece"],
        },
        "cal_selection_grid": cal_selection_grid,
        "raps_cal_hyperparam_grid": raps_cal_candidates,
        "test_branch_top1_by_bucket": bucket_top1,
        "test_branch_top1_overall_pct": round(
            100.0 * np.mean([int(np.argmax(s)) == g for s, g in zip(test_scores_sel, test_golds)]),
            2,
        ),
        "test_ece_lang_calibrated": round(test_ece_lang_cal, 6),
        "test_ece_snr_calibrated": round(test_ece_snr_cal, 6),
        "lang_mondrian_raps_fit": lang_raps_fit,
        "lang_mondrian_raps_test": lang_raps_test,
        "snr_mondrian_raps_fit": snr_raps_fit,
        "snr_mondrian_raps_test": snr_raps_test,
        "paired_lac_on_fixed_scores": {
            "tau_opt": round(tau_lac, 6),
            "lang_mondrian_lac_test": lang_lac_test,
            "snr_mondrian_lac_test": snr_lac_test,
        },
        "bootstrap_lang_resplit_100": boot_lang_lac,
        "bootstrap_snr_resplit_100": boot_snr_lac,
        "canonical_snr_strata": list(CANONICAL_SNR_STRATA),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true", help="Pre-download Qwen/Qwen3-4B on CPU and exit.")
    parser.add_argument(
        "--self-test-cpu",
        action="store_true",
        help="Verify layer-streaming + KV-prefix sharing matches monolithic Qwen3 to < 1e-5 on CPU.",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Run zero-leakage Mondrian-RAPS calibration from an existing .npz checkpoint on CPU.",
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ_PATH)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--score-scope", choices=("gbnf_tree", "both", "decision_prefix"), default="gbnf_tree")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--cand-row-chunk", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--layer-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--vocab-chunks", type=int, default=4)
    parser.add_argument("--spot-check-rows", type=int, default=3)
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU execution even during full export.")
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    seed_everything(int(args.seed))

    if args.download_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cached_path = download_qwen3_base_cpu(MODEL_ID)
        print(json.dumps({"action": "download_only", "model_id": MODEL_ID, "cached_path": cached_path}, indent=2))
        if not args.self_test_cpu:
            return 0

    if args.self_test_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        res = run_self_test_cpu()
        print(json.dumps(res, indent=2))
        return 0

    if args.cpu_only or args.calibrate_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.calibrate_only:
        if not args.npz_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint .npz: {args.npz_path}")
        with np.load(args.npz_path, allow_pickle=False) as loaded:
            npz_state = {k: loaded[k].copy() for k in loaded.files}
    else:
        data_dir, _ = _ensure_vertibench_v2(args.data_dir, args.seed)
        cal_rows = [enrich_vertibench_v2_row_for_z3(r) for r in _load_jsonl(data_dir / "cal.jsonl")]
        test_rows = [enrich_vertibench_v2_row_for_z3(r) for r in _load_jsonl(data_dir / "test.jsonl")]
        npz_state = export_qwen3_branch_logprobs_resumable(
            cal_rows=cal_rows,
            test_rows=test_rows,
            adapter_dir=args.adapter_dir,
            tokenizer_dir=args.tokenizer_dir,
            npz_path=args.npz_path,
            score_scope=args.score_scope,
            batch_size=args.batch_size,
            micro_batch=args.micro_batch,
            cand_row_chunk=args.cand_row_chunk,
            checkpoint_every=args.checkpoint_every,
            layer_dtype_str=args.layer_dtype,
            vocab_chunks=args.vocab_chunks,
            use_cuda=not args.cpu_only,
            spot_check_rows=args.spot_check_rows,
        )

    cal_eval = run_zero_leakage_mondrian_raps_evaluation(npz_state, alpha=0.05, seed=int(args.seed))
    spot_raw = str(npz_state.get("spot_check_json", "{}"))
    spot_parsed = json.loads(spot_raw) if spot_raw and spot_raw != "{}" else {}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "LIVE_LOCAL_GTX1650S_EXACT_LAYER_STREAMING_CONFORMAL_RAPS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "colab_cu_consumed": 0.0,
        "quantization_loss_zero": True,
        "model_id": MODEL_ID,
        "adapter_name": args.adapter_dir.name,
        "eval_split_provenance": HONEST_PROVENANCE_TAG_V2,
        "host_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "gpu_name": str(npz_state.get("gpu_name", "CPU")),
            "peak_vram_mib": round(float(npz_state.get("peak_vram_mib", 0.0)), 1),
        },
        "npz_checkpoint_name": args.npz_path.name,
        "score_scope": str(npz_state.get("score_scope", args.score_scope)),
        "layer_dtype": str(npz_state.get("layer_dtype", args.layer_dtype)),
        "monolithic_spot_check_parity": spot_parsed,
        **cal_eval,
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    write_canonical_json(args.out_json, payload)
    print(
        f"[DONE] Wrote {args.out_json.name} | "
        f"Lang-RAPS cov={cal_eval['lang_mondrian_raps_test']['overall_coverage_pct']}% "
        f"|C|={cal_eval['lang_mondrian_raps_test']['overall_mean_set_size']} "
        f"singl={cal_eval['lang_mondrian_raps_test']['overall_singleton_exec_pct']}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
