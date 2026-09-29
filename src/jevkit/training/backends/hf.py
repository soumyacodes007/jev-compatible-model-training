"""Plain Hugging Face backend: transformers + peft + trl, no Unsloth.

Runs on any CUDA GPU, including older ones Unsloth does not support. With
``model.load_in_4bit: true`` it trains QLoRA via bitsandbytes, which fits a 4B
model on a 16 GB T4 (fp16 is used automatically where bf16 is unavailable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevkit.config import TrainConfig
from jevkit.registry import trainers
from jevkit.training.common import prepare_run, sft_config, write_manifest


@trainers.register("hf")
def train_hf(
    cfg: TrainConfig, data_dir: str | Path, output_root: str | Path, mode: str, overwrite: bool
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer

    plan = prepare_run(cfg, data_dir, output_root, mode, overwrite)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
    quant = None
    if cfg.model.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        dtype=dtype,
        quantization_config=quant,
        device_map="auto",
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            use_rslora=cfg.lora.use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    optim = (
        cfg.optim.optim if quant is None or "8bit" not in cfg.optim.optim else "paged_adamw_8bit"
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=plan.train_ds,
        eval_dataset=plan.dev_ds,
        args=sft_config(plan, bf16=bf16, fp16=not bf16, optim=optim),
    )
    result = trainer.train()
    eval_metrics = trainer.evaluate() if plan.dev_ds is not None else {}

    model.save_pretrained(plan.output / "adapter")
    tokenizer.save_pretrained(plan.output / "adapter")
    # Merge into full-precision base weights so vLLM can serve them directly.
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, dtype=dtype, device_map="cpu"
    )
    from peft import PeftModel

    merged = PeftModel.from_pretrained(base, plan.output / "adapter").merge_and_unload()
    merged.save_pretrained(plan.output / "merged", safe_serialization=True)
    tokenizer.save_pretrained(plan.output / "merged")
    return write_manifest(
        plan,
        result,
        eval_metrics,
        ["torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"],
        extra={"precision": "bf16" if bf16 else "fp16", "quantized_training": quant is not None},
    )
