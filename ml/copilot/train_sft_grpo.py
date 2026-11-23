"""Colab L4 (24 GB) / A100 BF16 LoRA SFT -> Logic-Verified Z3-GRPO (RLVR) Training Pipeline.

Strictly enforces:
- Apache-2.0 fine-tuning lock: Qwen/Qwen3-8B & Qwen/Qwen3-4B ONLY (asserts is_model_allowed).
- Full-BF16 LoRA r=32, alpha=64 across all 7 linear projections, enable_thinking=False.
- Right-sized L4 (24 GB) memory controls: `--batch-size 2`, `--no-vllm` (default safe for L4),
  optional `--vllm-gpu-util 0.30`, and gradient checkpointing (eliminating colocate vLLM OOM).
- Calibrated DR-6 Erratum #3 policy config:
    * Default token-level dr_grpo: epsilon=0.20, epsilon_high=0.28
    * Optional sequence-level GSPO (arXiv:2507.18071): epsilon=3e-4, epsilon_high=4e-4
    * Mismatched sequence + large clip bounds raises ValueError at startup!
- Non-zero KL beta=0.04 + 10% SFT bilingual replay to protect Turkish agglutinative morphology.
- PersistentZ3RewardPool(max_workers=8) using spawn context created ONCE before trainer.train().
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import random
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.vertibench_tr_en.validate_vertibench import is_model_allowed
from ml.copilot.grpo_z3_rewards import PersistentZ3RewardPool

QWEN3_EMPTY_THINK_SUFFIX: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class GrpoPolicyConfig:
    model_id: str = "Qwen/Qwen3-4B"
    loss_type: str = "dr_grpo"
    importance_sampling_level: str = "token"
    epsilon: float = 0.20
    epsilon_high: float = 0.28
    beta: float = 0.04
    entropy_coef: float = 0.005
    lora_r: int = 32
    lora_alpha: int = 64
    num_generations: int = 8
    max_completion_length: int = 64
    sft_replay_fraction: float = 0.10
    z3_workers: int = 8
    enable_thinking: bool = False
    batch_size: int = 2
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.30
    sft_epochs: int = 2
    max_grpo_steps: int = 120
    grpo_grad_accum: int = 2
    grpo_learning_rate: float = 5.0e-6
    grpo_temperature: float = 0.85
    save_merged_bf16: bool = False
    target_modules: tuple[str, ...] = field(default=LORA_TARGET_MODULES)

    def __post_init__(self) -> None:
        if not is_model_allowed(self.model_id, "train"):
            raise ValueError(
                f"Model {self.model_id!r} is banned from fine-tuning; "
                "only Apache-2.0 Qwen/Qwen3-8B and Qwen/Qwen3-4B are permitted."
            )
        if self.enable_thinking:
            raise ValueError("enable_thinking must remain False to prevent Qwen3 GBNF deadlock.")
        if self.beta <= 0.0:
            raise ValueError(
                f"KL beta must be > 0 (recommended 0.04) to preserve Turkish morphology, got {self.beta}"
            )
        if self.importance_sampling_level not in ("token", "sequence"):
            raise ValueError(
                f"importance_sampling_level must be 'token' or 'sequence', got {self.importance_sampling_level!r}"
            )
        if self.importance_sampling_level == "sequence":
            if self.epsilon > 1e-3 or self.epsilon_high > 1e-3:
                raise ValueError(
                    "GSPO sequence-level importance sampling requires calibrated tight clip bounds "
                    f"(<= 1e-3, e.g., 3e-4/4e-4 per arXiv:2507.18071), got {self.epsilon}/{self.epsilon_high}."
                )
        else:
            if not (0.0 < self.epsilon <= self.epsilon_high <= 0.5):
                raise ValueError(
                    f"Token-level clip bounds invalid: epsilon={self.epsilon}, epsilon_high={self.epsilon_high}"
                )
        if self.max_completion_length > 128:
            raise ValueError(
                f"Compact DSL must stay <= 128 tokens (default 64), got {self.max_completion_length}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grpo_grad_accum < 1:
            raise ValueError(f"grpo_grad_accum must be >= 1, got {self.grpo_grad_accum}")
        if not (0.1 <= self.grpo_temperature <= 2.0):
            raise ValueError(f"grpo_temperature must lie in [0.1, 2.0], got {self.grpo_temperature}")
        if not (0.10 <= self.vllm_gpu_memory_utilization <= 0.85):
            raise ValueError(
                f"vllm_gpu_memory_utilization must lie in [0.10, 0.85], got {self.vllm_gpu_memory_utilization}"
            )

    @property
    def sft_learning_rate(self) -> float:
        return 1.0e-4 if "8B" in self.model_id else 1.5e-4


def format_copilot_prompt(row: dict[str, Any]) -> str:
    """Format compact RTA state snapshot + user utterance ending with empty Qwen3 think block."""
    st = row["rta_snapshot"]
    own = st["ownship"]
    slz_summary = ",".join(
        f"{s['id']}:p={s['p_safe']:.2f},eta={int(round(s['eta_s']))},m={s['soc_margin_pct']:+.1f}"
        for s in st.get("ranked_slzs", [])[:4]
    )
    user_msg = (
        f"FAULT={st['fault_mode']};AGL={int(round(own['agl_m']))};"
        f"SOC={own['soc_pct']:.1f};V={own['v_term_volts']:.1f};"
        f"WIND={own['wind_mps']:.1f};SLZ=[{slz_summary}]\n"
        f"UTT={row['user_utterance']}"
    )
    return (
        "<|im_start|>system\n"
        "VertiRTA EASA AI L1/2A Contingency Compiler. Output strictly 2-line ASCII compact DSL.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}"
        f"{QWEN3_EMPTY_THINK_SUFFIX}"
    )


def build_training_plan_summary(
    cfg: GrpoPolicyConfig,
    train_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build deterministic training manifest for dry-run inspection or Colab L4/A100 execution."""
    grad_accum_sft = max(1, 32 // cfg.batch_size)
    return {
        "schema_version": "vertirta.sft_grpo_plan.v1",
        "model_id": cfg.model_id,
        "license": "apache-2.0",
        "dtype": "bfloat16",
        "target_gpu_tier": "L4_24GB" if "4B" in cfg.model_id else "A100_40GB",
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "target_modules": list(cfg.target_modules),
        "stage1_sft": {
            "epochs": cfg.sft_epochs,
            "learning_rate": cfg.sft_learning_rate,
            "per_device_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": grad_accum_sft,
            "effective_batch_size": cfg.batch_size * grad_accum_sft,
            "enable_thinking": cfg.enable_thinking,
        },
        "stage2_grpo": {
            "loss_type": cfg.loss_type,
            "importance_sampling_level": cfg.importance_sampling_level,
            "epsilon": cfg.epsilon,
            "epsilon_high": cfg.epsilon_high,
            "beta": cfg.beta,
            "entropy_coef": cfg.entropy_coef,
            "per_device_batch_size": cfg.batch_size,
            "num_generations": cfg.num_generations,
            "max_completion_length": cfg.max_completion_length,
            "max_steps": cfg.max_grpo_steps,
            "use_vllm": cfg.use_vllm,
            "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
            "sft_replay_fraction": cfg.sft_replay_fraction,
            "z3_reward_pool_start_method": "spawn",
            "z3_workers": cfg.z3_workers,
        },
        "train_jsonl": str(train_jsonl),
        "output_dir": str(output_dir),
    }


def summarize_reward_history(reward_history: list[float]) -> tuple[float, float]:
    """Compute mean reward over first-10 and last-10 GRPO batches safely."""
    first = reward_history[:10]
    last = reward_history[-10:]
    first_mean = sum(first) / float(len(first)) if first else 0.0
    last_mean = sum(last) / float(len(last)) if last else 0.0
    return round(first_mean, 4), round(last_mean, 4)


def sanitize_grpo_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Ensure gold_intent list/bool fields are never None after HF Arrow serialization."""
    out = dict(meta)
    gold = dict(out.get("gold_intent") or {})
    for list_key in ("valid_targets", "top2_sat_pads", "expected_unsat_labels"):
        val = gold.get(list_key)
        gold[list_key] = list(val) if val is not None else []
    for bool_key in ("requires_clarification", "requires_abstention", "allow_loiter"):
        gold[bool_key] = bool(gold.get(bool_key) or False)
    out["gold_intent"] = gold
    return out


def _filter_supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig_params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig_params}


def run_colab_sft_grpo(
    cfg: GrpoPolicyConfig,
    train_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute BF16 LoRA Stage-1 SFT -> Stage-2 Z3-GRPO on Colab L4 (24 GB) or A100."""
    import datasets  # type: ignore[import-not-found]
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in train_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(2026)
    rng.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, lora_cfg)

    sft_records = [
        {"text": format_copilot_prompt(r) + r["canonical_dsl"] + "<|im_end|>"} for r in rows
    ]
    sft_ds = datasets.Dataset.from_list(sft_records)
    sft_dir = output_dir / "stage1_sft"
    sft_lora_dir = output_dir / "stage1_sft_lora"
    grad_accum_sft = max(1, 32 // cfg.batch_size)

    sft_kwargs = _filter_supported_kwargs(
        SFTConfig,
        {
            "output_dir": str(sft_dir),
            "num_train_epochs": cfg.sft_epochs,
            "per_device_train_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": grad_accum_sft,
            "learning_rate": cfg.sft_learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": True,
            "gradient_checkpointing": True,
            "dataset_text_field": "text",
            "max_length": 512,
            "max_seq_length": 512,
            "logging_steps": 10,
            "save_strategy": "no",
            "report_to": "none",
        },
    )
    sft_trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": sft_ds,
        "args": SFTConfig(**sft_kwargs),
    }
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        sft_trainer_kwargs["processing_class"] = tokenizer
    else:
        sft_trainer_kwargs["tokenizer"] = tokenizer

    sft_trainer = SFTTrainer(**sft_trainer_kwargs)
    sft_result = sft_trainer.train()
    model.save_pretrained(str(sft_lora_dir))
    tokenizer.save_pretrained(str(sft_lora_dir))

    # Mix 10% SFT replay prompts into GRPO rollout schedule for Turkish fluency stability.
    # Store metadata as JSON string so HF Arrow never injects None into optional gold_intent keys.
    n_replay = max(1, int(round(len(rows) * cfg.sft_replay_fraction)))
    grpo_rows = rows + rng.sample(rows, min(n_replay, len(rows)))
    grpo_ds = datasets.Dataset.from_list(
        [
            {
                "prompt": format_copilot_prompt(r),
                "metadata_json": json.dumps(
                    sanitize_grpo_metadata(
                        {
                            "rta_snapshot": r["rta_snapshot"],
                            "bucket": r["bucket"],
                            "gold_intent": r["gold_intent"],
                            "expected_lang": r["lang"],
                        }
                    ),
                    ensure_ascii=False,
                ),
            }
            for r in grpo_rows
        ]
    )

    tokenizer.padding_side = "left"
    grpo_dir = output_dir / "stage2_grpo"
    grpo_lora_dir = output_dir / "stage2_grpo_lora"

    # TRL GRPO requires generation_batch_size divisible by num_generations.
    # On L4 24 GB, per_device_train_batch_size = max(cfg.num_generations, cfg.batch_size)
    # when TRL expects completions-per-step, or cfg.batch_size prompts depending on TRL version.
    grpo_raw_kwargs: dict[str, Any] = {
        "output_dir": str(grpo_dir),
        "learning_rate": cfg.grpo_learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.10,
        "max_grad_norm": 0.5,
        "per_device_train_batch_size": cfg.num_generations,
        "gradient_accumulation_steps": cfg.grpo_grad_accum,
        "num_generations": cfg.num_generations,
        "temperature": cfg.grpo_temperature,
        "max_completion_length": cfg.max_completion_length,
        "max_prompt_length": 384,
        "loss_type": cfg.loss_type,
        "importance_sampling_level": cfg.importance_sampling_level,
        "epsilon": cfg.epsilon,
        "epsilon_high": cfg.epsilon_high,
        "beta": cfg.beta,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_vllm": cfg.use_vllm,
        "mask_truncated_completions": True,
        "logging_steps": 5,
        "save_strategy": "no",
        "max_steps": cfg.max_grpo_steps,
        "report_to": "none",
    }
    if cfg.use_vllm:
        grpo_raw_kwargs["vllm_mode"] = "colocate"
        grpo_raw_kwargs["vllm_gpu_memory_utilization"] = cfg.vllm_gpu_memory_utilization

    grpo_args = GRPOConfig(**_filter_supported_kwargs(GRPOConfig, grpo_raw_kwargs))

    # Persistent spawn Z3 worker pool created ONCE outside trainer.train() (DR-6 Erratum #2)
    reward_history: list[float] = []
    with PersistentZ3RewardPool(max_workers=cfg.z3_workers) as reward_pool:

        def z3_rlvr_reward(
            completions: list[Any],
            metadata_json: list[str] | None = None,
            metadata: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            del kwargs
            texts = [
                c[0]["content"] if isinstance(c, list) and c and isinstance(c[0], dict) else str(c)
                for c in completions
            ]
            if metadata_json is not None:
                parsed_meta = [sanitize_grpo_metadata(json.loads(m)) for m in metadata_json]
            elif metadata is not None:
                parsed_meta = [sanitize_grpo_metadata(m) for m in metadata]
            else:
                raise RuntimeError("GRPO reward callback received neither metadata_json nor metadata")
            scores = reward_pool.score_batch(texts, parsed_meta)
            if scores:
                reward_history.append(sum(scores) / float(len(scores)))
            return scores

        grpo_trainer_kwargs: dict[str, Any] = {
            "model": model,
            "reward_funcs": [z3_rlvr_reward],
            "args": grpo_args,
            "train_dataset": grpo_ds,
        }
        if "processing_class" in inspect.signature(GRPOTrainer.__init__).parameters:
            grpo_trainer_kwargs["processing_class"] = tokenizer
        else:
            grpo_trainer_kwargs["tokenizer"] = tokenizer

        grpo_trainer = GRPOTrainer(**grpo_trainer_kwargs)
        grpo_result = grpo_trainer.train()

    model.save_pretrained(str(grpo_lora_dir))
    tokenizer.save_pretrained(str(grpo_lora_dir))

    if cfg.save_merged_bf16:
        merged = model.merge_and_unload()
        merged_dir = output_dir / "merged_bf16"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024.0**3), 3)
        if torch.cuda.is_available()
        else 0.0
    )
    first10_rew, last10_rew = summarize_reward_history(reward_history)
    return {
        "sft_train_loss": round(float(sft_result.training_loss), 5),
        "grpo_train_loss": round(float(grpo_result.training_loss), 5),
        "grpo_mean_reward_first10": first10_rew,
        "grpo_mean_reward_last10": last10_rew,
        "grpo_reward_batches": len(reward_history),
        "peak_vram_gb": peak_vram_gb,
        "sft_lora_dir": str(sft_lora_dir),
        "grpo_lora_dir": str(grpo_lora_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VertiRTA Colab L4 (24 GB) / A100 BF16 LoRA SFT -> Z3-GRPO trainer."
    )
    parser.add_argument(
        "--model",
        "--model-id",
        dest="model_id",
        default="Qwen/Qwen3-4B",
        help="Apache-2.0 model ID (Qwen/Qwen3-4B for L4 24 GB; Qwen/Qwen3-8B for A100).",
    )
    parser.add_argument("--train-jsonl", type=Path, default=Path("benchmarks/vertibench_tr_en/data/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3.1.0/copilot_ckpt"))
    parser.add_argument("--gspo", action="store_true", help="Use calibrated GSPO (3e-4 / 4e-4) instead of token dr_grpo.")
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size (default 2 for L4 24 GB).")
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO rollout group size G (default 8).")
    parser.add_argument("--sft-epochs", type=int, default=2, help="Stage-1 SFT epochs (default 2).")
    parser.add_argument("--max-grpo-steps", type=int, default=120, help="Stage-2 Z3-GRPO optimizer steps.")
    parser.add_argument("--grpo-grad-accum", type=int, default=2, help="GRPO gradient accumulation steps.")
    parser.add_argument("--grpo-lr", type=float, default=5.0e-6, help="GRPO learning rate.")
    parser.add_argument("--grpo-temperature", type=float, default=0.85, help="GRPO rollout sampling temperature.")
    parser.add_argument("--no-vllm", action="store_true", help="Disable colocate vLLM to prevent L4 24 GB OOM (default).")
    parser.add_argument("--use-vllm", action="store_true", help="Opt-in to colocate vLLM rollout engine.")
    parser.add_argument(
        "--vllm-gpu-util",
        type=float,
        default=0.30,
        help="vLLM GPU memory utilization fraction when --use-vllm is enabled (default 0.30).",
    )
    parser.add_argument("--save-merged-bf16", action="store_true", help="Also merge and save full BF16 weights.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan JSON without GPU.")
    args = parser.parse_args(argv)

    use_vllm = bool(args.use_vllm and not args.no_vllm)
    cfg = GrpoPolicyConfig(
        model_id=args.model_id,
        loss_type="dr_grpo",
        importance_sampling_level="sequence" if args.gspo else "token",
        epsilon=3e-4 if args.gspo else 0.20,
        epsilon_high=4e-4 if args.gspo else 0.28,
        beta=args.beta,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        sft_epochs=args.sft_epochs,
        max_grpo_steps=args.max_grpo_steps,
        grpo_grad_accum=args.grpo_grad_accum,
        grpo_learning_rate=args.grpo_lr,
        grpo_temperature=args.grpo_temperature,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        save_merged_bf16=args.save_merged_bf16,
    )
    summary = build_training_plan_summary(cfg, args.train_jsonl, args.output_dir)
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        metrics = run_colab_sft_grpo(cfg, args.train_jsonl, args.output_dir)
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
