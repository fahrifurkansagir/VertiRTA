"""Colab G4 96 GB Blackwell SFT -> Logic-Verified GRPO (RLVR) Training Pipeline.

Strictly enforces:
- Apache-2.0 fine-tuning lock: Qwen/Qwen3-8B & Qwen/Qwen3-4B ONLY (asserts is_model_allowed).
- Full-BF16 LoRA r=32, alpha=64 across all 7 linear projections, enable_thinking=False.
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
import json
from pathlib import Path
import random
from typing import Any

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
    """Build deterministic training manifest for dry-run inspection or Colab G4 execution."""
    return {
        "schema_version": "vertirta.sft_grpo_plan.v1",
        "model_id": cfg.model_id,
        "license": "apache-2.0",
        "dtype": "bfloat16",
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "target_modules": list(cfg.target_modules),
        "stage1_sft": {
            "epochs": 3,
            "learning_rate": cfg.sft_learning_rate,
            "effective_batch_size": 32,
            "enable_thinking": cfg.enable_thinking,
        },
        "stage2_grpo": {
            "loss_type": cfg.loss_type,
            "importance_sampling_level": cfg.importance_sampling_level,
            "epsilon": cfg.epsilon,
            "epsilon_high": cfg.epsilon_high,
            "beta": cfg.beta,
            "entropy_coef": cfg.entropy_coef,
            "num_generations": cfg.num_generations,
            "max_completion_length": cfg.max_completion_length,
            "sft_replay_fraction": cfg.sft_replay_fraction,
            "z3_reward_pool_start_method": "spawn",
            "z3_workers": cfg.z3_workers,
        },
        "train_jsonl": str(train_jsonl),
        "output_dir": str(output_dir),
    }


def run_colab_sft_grpo(cfg: GrpoPolicyConfig, train_jsonl: Path, output_dir: Path) -> None:
    """Execute full BF16 Stage-1 SFT -> Stage-2 Z3-GRPO on Colab G4 96 GB Blackwell."""
    import datasets  # type: ignore[import-not-found]
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    rows = [
        json.loads(line)
        for line in train_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(2026)
    rng.shuffle(rows)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=1024,
        load_in_4bit=False,  # Full BF16 fits cleanly in 96 GB Blackwell
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(cfg.target_modules),
    )

    sft_records = [
        {"text": format_copilot_prompt(r) + r["canonical_dsl"] + "<|im_end|>"} for r in rows
    ]
    sft_ds = datasets.Dataset.from_list(sft_records)
    sft_dir = output_dir / "stage1_sft"
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_ds,
        args=SFTConfig(
            output_dir=str(sft_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=cfg.sft_learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
        ),
    )
    sft_trainer.train()

    # Mix 10% SFT replay prompts into GRPO rollout schedule for Turkish fluency stability
    n_replay = max(1, int(round(len(rows) * cfg.sft_replay_fraction)))
    grpo_rows = rows + rng.sample(rows, min(n_replay, len(rows)))
    grpo_ds = datasets.Dataset.from_list(
        [
            {
                "prompt": format_copilot_prompt(r),
                "metadata": {
                    "rta_snapshot": r["rta_snapshot"],
                    "bucket": r["bucket"],
                    "gold_intent": r["gold_intent"],
                    "expected_lang": r["lang"],
                },
            }
            for r in grpo_rows
        ]
    )

    grpo_dir = output_dir / "stage2_grpo"
    grpo_args = GRPOConfig(
        output_dir=str(grpo_dir),
        learning_rate=5.0e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        max_grad_norm=0.5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        loss_type=cfg.loss_type,
        importance_sampling_level=cfg.importance_sampling_level,
        epsilon=cfg.epsilon,
        epsilon_high=cfg.epsilon_high,
        beta=cfg.beta,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        mask_truncated_completions=True,
        logging_steps=5,
        max_steps=450,
    )

    # Persistent spawn Z3 worker pool created ONCE outside trainer.train() (DR-6 Erratum #2)
    with PersistentZ3RewardPool(max_workers=cfg.z3_workers) as reward_pool:
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_pool],
            args=grpo_args,
            train_dataset=grpo_ds,
        )
        grpo_trainer.train()

    merged_dir = output_dir / "merged_bf16"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")


def main() -> int:
    parser = argparse.ArgumentParser(description="VertiRTA Colab G4 BF16 SFT -> Z3-GRPO trainer.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-jsonl", type=Path, default=Path("benchmarks/vertibench_tr_en/data/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2.0.0/copilot_ckpt"))
    parser.add_argument("--gspo", action="store_true", help="Use calibrated GSPO (3e-4 / 4e-4) instead of token dr_grpo.")
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan JSON without GPU.")
    args = parser.parse_args()

    cfg = GrpoPolicyConfig(
        model_id=args.model_id,
        loss_type="dr_grpo",
        importance_sampling_level="sequence" if args.gspo else "token",
        epsilon=3e-4 if args.gspo else 0.20,
        epsilon_high=4e-4 if args.gspo else 0.28,
        beta=args.beta,
    )
    summary = build_training_plan_summary(cfg, args.train_jsonl, args.output_dir)
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        run_colab_sft_grpo(cfg, args.train_jsonl, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
