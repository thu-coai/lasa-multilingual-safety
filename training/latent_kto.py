"""
Semantic-conditioned KTO training script for LASA.

This script implements the KTO-style alignment path used in the LASA
repository. It teaches the model to associate latent unsafe semantics with
the conditional refusal prompt used at inference time.

Expected data format:
- Paired preference format: prompt, chosen, rejected, and optional source.
- Unpaired KTO format: prompt, completion, label, and optional source.

The bundled release data uses paired preference format. The trainer converts
paired examples to completion/label records internally while preserving source,
which controls the training-time LASA unsafe semantic-label sampling ratio.

Example usage:
    python latent_kto.py \
        --dataset_path data/training/mixed_dpo_ultra_safety_converted.json \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --output_dir ./latent_kto_output \
        --safety_unsafe_ratio 0.5 \
        --ultrafeedback_unsafe_ratio 0.05 \
        --per_device_train_batch_size 4 \
        --learning_rate 5e-7 \
        --num_train_epochs 3 \
        --beta 0.1
"""

import os
import sys


def _add_trl_source_to_path() -> None:
    """Prefer an explicitly configured TRL source checkout when provided."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    candidates = [
        os.environ.get("TRL_SOURCE_DIR"),
        os.path.join(repo_root, "external", "trl"),
    ]

    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "trl")):
            sys.path.insert(0, candidate)
            return


_add_trl_source_to_path()

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from trl import KTOConfig
from latentkto_trainer import LatentKTOTrainer


@dataclass
class AdditionalArgs:
    """Additional arguments for LASA semantic-conditioned KTO training."""
    dataset_path: str = field(
        metadata={"help": "Path to a JSON dataset file in paired preference or unpaired KTO format"}
    )
    model_path: str = field(
        metadata={"help": "Path to the pretrained model or HuggingFace model name"}
    )
    safety_unsafe_ratio: float = field(
        default=0.5,
        metadata={"help": "Ratio of safety-source samples to label with LASA unsafe semantics (0.0-1.0)"}
    )
    ultrafeedback_unsafe_ratio: float = field(
        default=0.05,
        metadata={"help": "Ratio of ultrafeedback samples to label with LASA unsafe semantics (0.0-1.0)"}
    )
    ref_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to reference model. If None, a copy of the model is used."}
    )


def main():
    parser = HfArgumentParser((KTOConfig, AdditionalArgs))
    training_args, additional_args = parser.parse_args_into_dataclasses()
    
    print("="*80)
    print("LASA Semantic-Conditioned KTO Training")
    print("="*80)
    print("\nTraining Arguments:")
    print(training_args)
    print("\n" + "="*80)
    print("Additional Arguments:")
    print("="*80)
    print(f"  dataset_path: {additional_args.dataset_path}")
    print(f"  model_path: {additional_args.model_path}")
    print(f"  ref_model_path: {additional_args.ref_model_path}")
    print(f"  safety_unsafe_ratio: {additional_args.safety_unsafe_ratio}")
    print(f"  ultrafeedback_unsafe_ratio: {additional_args.ultrafeedback_unsafe_ratio}")
    print("="*80 + "\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(additional_args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(
        "json",
        data_files=additional_args.dataset_path,
    )["train"]
    
    # def add_chat_template_kwargs(example):
    #     example["chat_template_kwargs"] = {"enable_thinking": False}
    #     return example

    # if "Qwen3-8B" in additional_args.model_path:
    #     dataset = dataset.map(add_chat_template_kwargs)
    # else:
    #     pass
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Sample 0 keys: {list(dataset[0].keys())}")
    
    # Print dataset statistics
    if "source" in dataset[0]:
        safety_count = sum(1 for i in range(len(dataset)) if dataset[i].get("source") == "safety")
        ultra_count = sum(1 for i in range(len(dataset)) if dataset[i].get("source") == "ultrafeedback")
        other_count = len(dataset) - safety_count - ultra_count
        print(f"  - safety samples: {safety_count}")
        print(f"  - ultrafeedback samples: {ultra_count}")
        if other_count > 0:
            print(f"  - other samples: {other_count}")
    
    if "chosen" in dataset[0] and "rejected" in dataset[0]:
        print(f"  - paired preference examples: {len(dataset)}")
        print(f"  - unpaired training examples after conversion: {2 * len(dataset)}")
    
    if "label" in dataset[0]:
        desirable_count = sum(1 for i in range(len(dataset)) if dataset[i].get("label") is True)
        undesirable_count = len(dataset) - desirable_count
        print(f"  - desirable (label=True): {desirable_count}")
        print(f"  - undesirable (label=False): {undesirable_count}")
    
    # Determine reference model
    ref_model = additional_args.ref_model_path if additional_args.ref_model_path else None
    
    # Initialize trainer
    trainer = LatentKTOTrainer(
        model=additional_args.model_path,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        safety_unsafe_ratio=additional_args.safety_unsafe_ratio,
        ultrafeedback_unsafe_ratio=additional_args.ultrafeedback_unsafe_ratio,
        random_seed=training_args.seed,
    )
    
    print("\nStarting LASA semantic-conditioned KTO training...")
    trainer.train()
    print("\nTraining completed!")
    
    # Save the model
    trainer.save_model(training_args.output_dir)
    print(f"\nModel saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
