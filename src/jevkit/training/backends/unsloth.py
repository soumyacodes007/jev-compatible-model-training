"""Unsloth LoRA backend (default): fastest and most memory-efficient on one GPU."""
# ruff: noqa: I001 -- Unsloth must be imported before TRL and Transformers.

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevkit.config import TrainConfig
from jevkit.registry import trainers
from jevkit.training.common import prepare_run, sft_config, write_manifest


@trainers.register("unsloth")
def train_unsloth(
    cfg: TrainConfig, data_dir: str | Path, output_root: str | Path, mode: str, overwrite: bool
) -> dict[str, Any]:
    import unsloth  # noqa: F401
    import torch
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    plan = prepare_run(cfg, data_dir, output_root, mode, overwrite)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        revision=cfg.model.revision,
        max_seq_length=cfg.model.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_16bit=not cfg.model.load_in_4bit,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.optim.seed,
        use_rslora=cfg.lora.use_rslora,
        loftq_config=None,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=plan.train_ds,
        eval_dataset=plan.dev_ds,
        args=sft_config(plan, dataset_num_proc=8, bf16=True, fp16=False),
    )
    result = trainer.train()
    eval_metrics = trainer.evaluate() if plan.dev_ds is not None else {}

    model.save_pretrained(plan.output / "adapter")
    tokenizer.save_pretrained(plan.output / "adapter")
    model.save_pretrained_merged(str(plan.output / "merged"), tokenizer, save_method="merged_16bit")
    return write_manifest(
        plan,
        result,
        eval_metrics,
        ["torch", "unsloth", "unsloth-zoo", "transformers", "trl", "datasets"],
    )
