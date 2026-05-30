# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LatentKTOTrainer for LASA semantic-conditioned alignment.

This trainer:
1. Samples training-time unsafe semantic labels based on data source.
2. For unsafe samples, injects the LASA conditional refusal prompt.
3. Uses KTO loss for preference optimization.
"""

import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, concatenate_datasets
from torch import autocast
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.utils import is_peft_available

from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.kto_config import KTOConfig
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


# LASA conditional prompt injected when the latent semantic label is unsafe.
SAFETY_PREFIX = "Harmful query detected. I should refuse this request and provide a safe response in the user's language."


class LatentKTODataCollator:
    """
    Data collator for LASA KTO that handles normal and conditional-prompt versions.
    """
    
    def __init__(
        self,
        pad_token_id: int,
        label_pad_token_id: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Standard KTO fields
        batch = {}
        
        # Collect sequences that need padding
        prompt_input_ids = [torch.tensor(f["prompt_input_ids"]) for f in features]
        prompt_attention_mask = [torch.tensor(f["prompt_attention_mask"]) for f in features]
        completion_input_ids = [torch.tensor(f["completion_input_ids"]) for f in features]
        completion_attention_mask = [torch.tensor(f["completion_attention_mask"]) for f in features]
        completion_labels = [torch.tensor(f["completion_labels"]) for f in features]
        
        # Pad sequences
        batch["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        batch["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        batch["completion_input_ids"] = pad(completion_input_ids, padding_value=self.pad_token_id, padding_side="right")
        batch["completion_attention_mask"] = pad(completion_attention_mask, padding_value=0, padding_side="right")
        batch["completion_labels"] = pad(completion_labels, padding_value=self.label_pad_token_id, padding_side="right")
        
        # Handle with_prefix versions
        if "completion_input_ids_with_prefix" in features[0]:
            completion_input_ids_wp = [torch.tensor(f["completion_input_ids_with_prefix"]) for f in features]
            completion_attention_mask_wp = [torch.tensor(f["completion_attention_mask_with_prefix"]) for f in features]
            completion_labels_wp = [torch.tensor(f["completion_labels_with_prefix"]) for f in features]
            
            batch["completion_input_ids_with_prefix"] = pad(completion_input_ids_wp, padding_value=self.pad_token_id, padding_side="right")
            batch["completion_attention_mask_with_prefix"] = pad(completion_attention_mask_wp, padding_value=0, padding_side="right")
            batch["completion_labels_with_prefix"] = pad(completion_labels_wp, padding_value=self.label_pad_token_id, padding_side="right")
        
        # Handle KL dataset fields if present
        if "KL_completion_input_ids" in features[0]:
            kl_completion_input_ids = [torch.tensor(f["KL_completion_input_ids"]) for f in features]
            kl_completion_attention_mask = [torch.tensor(f["KL_completion_attention_mask"]) for f in features]
            kl_completion_labels = [torch.tensor(f["KL_completion_labels"]) for f in features]
            
            batch["KL_completion_input_ids"] = pad(kl_completion_input_ids, padding_value=self.pad_token_id, padding_side="right")
            batch["KL_completion_attention_mask"] = pad(kl_completion_attention_mask, padding_value=0, padding_side="right")
            batch["KL_completion_labels"] = pad(kl_completion_labels, padding_value=self.label_pad_token_id, padding_side="right")
            
            # KL with prefix
            if "KL_completion_input_ids_with_prefix" in features[0]:
                kl_completion_input_ids_wp = [torch.tensor(f["KL_completion_input_ids_with_prefix"]) for f in features]
                kl_completion_attention_mask_wp = [torch.tensor(f["KL_completion_attention_mask_with_prefix"]) for f in features]
                kl_completion_labels_wp = [torch.tensor(f["KL_completion_labels_with_prefix"]) for f in features]
                
                batch["KL_completion_input_ids_with_prefix"] = pad(kl_completion_input_ids_wp, padding_value=self.pad_token_id, padding_side="right")
                batch["KL_completion_attention_mask_with_prefix"] = pad(kl_completion_attention_mask_wp, padding_value=0, padding_side="right")
                batch["KL_completion_labels_with_prefix"] = pad(kl_completion_labels_wp, padding_value=self.label_pad_token_id, padding_side="right")
        
        # Non-tensor fields
        batch["label"] = [f["label"] for f in features]
        batch["is_unsafe"] = torch.tensor([f["is_unsafe"] for f in features], dtype=torch.long)
        
        if "prompt" in features[0]:
            batch["prompt"] = [f["prompt"] for f in features]
        if "completion" in features[0]:
            batch["completion"] = [f["completion"] for f in features]
        
        # Reference logps if precomputed
        if "reference_logps" in features[0]:
            batch["reference_logps"] = torch.tensor([f["reference_logps"] for f in features])
        if "reference_KL_logps" in features[0]:
            batch["reference_KL_logps"] = torch.tensor([f["reference_KL_logps"] for f in features])
        
        return batch


class LatentKTOTrainer(BaseTrainer):
    """
    LASA semantic-conditioned KTO trainer.
    
    This trainer samples unsafe semantic labels based on data source:
    - safety data: `safety_unsafe_ratio` of samples receive the conditional prompt
    - ultrafeedback data: `ultrafeedback_unsafe_ratio` of samples receive it
    
    Samples labeled as unsafe receive SAFETY_PREFIX before the completion.
    The model is then trained using KTO loss on preference pairs.
    """
    
    _tag_names = ["trl", "latent-kto"]
    _name = "LatentKTO"
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: KTOConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        data_collator: Optional[DataCollator] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        # Latent-specific parameters
        safety_unsafe_ratio: float = 0.5,
        ultrafeedback_unsafe_ratio: float = 0.05,
        random_seed: int = 42,
    ):
        """
        Args:
            model: The model to train.
            ref_model: The reference model for KTO loss computation.
            args: KTO configuration.
            train_dataset: The training dataset with preference pairs.
            eval_dataset: The evaluation dataset.
            processing_class: Tokenizer or processor.
            data_collator: Data collator for batching.
            safety_unsafe_ratio: Ratio of safety-source samples receiving LASA unsafe semantics (default: 0.5)
            ultrafeedback_unsafe_ratio: Ratio of ultrafeedback samples receiving LASA unsafe semantics (default: 0.05)
            random_seed: Random seed for reproducibility (default: 42)
        """
        if type(args) is TrainingArguments:
            raise ValueError("Please use `KTOConfig` instead of TrainingArguments.")
        
        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. Pass a copy or `None` for peft."
            )
        
        # Store latent-specific parameters
        self.safety_unsafe_ratio = safety_unsafe_ratio
        self.ultrafeedback_unsafe_ratio = ultrafeedback_unsafe_ratio
        self.random_seed = random_seed
        
        # Handle model initialization
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs but model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            dtype = model_init_kwargs.get("dtype")
            if dtype is not None:
                if isinstance(dtype, str) and dtype != "auto":
                    dtype = getattr(torch, dtype)
                if dtype != "auto" and not isinstance(dtype, torch.dtype):
                    raise ValueError(f"Invalid dtype: {dtype}")
                model_init_kwargs["dtype"] = dtype
        
        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError("You passed ref_model_init_kwargs but ref_model is already instantiated.")
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            dtype = ref_model_init_kwargs.get("dtype")
            if dtype is not None:
                if isinstance(dtype, str) and dtype != "auto":
                    dtype = getattr(torch, dtype)
                if dtype != "auto" and not isinstance(dtype, torch.dtype):
                    raise ValueError(f"Invalid dtype: {dtype}")
                ref_model_init_kwargs["dtype"] = dtype
        
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)
        
        self._peft_has_been_casted_to_bf16 = False
        
        # Handle PEFT
        if not is_peft_available() and peft_config is not None:
            raise ValueError("PEFT is not installed. Install with `pip install peft`.")
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True
        elif args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder
        
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        
        # Check if DeepSpeed ZeRO-3 is enabled (need to detect early before create_reference_model)
        self._is_deepspeed_zero3 = False
        if args.deepspeed is not None:
            import json
            if isinstance(args.deepspeed, str):
                with open(args.deepspeed, "r") as f:
                    ds_config = json.load(f)
            else:
                ds_config = args.deepspeed
            zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
            self._is_deepspeed_zero3 = (zero_stage == 3)
        
        # Store model path for deferred ref model loading with ZeRO-3
        self._ref_model_path = model.config._name_or_path
        self._ref_model_init_kwargs = ref_model_init_kwargs if ref_model_init_kwargs else model_init_kwargs
        
        # Setup reference model
        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            self.ref_model = None
        elif self._is_deepspeed_zero3:
            # DeepSpeed ZeRO-3 is not compatible with create_reference_model()
            # Defer loading until after super().__init__() when accelerator is available
            # This allows prepare_deepspeed to properly shard the model
            self.ref_model = None  # Will be loaded after super().__init__()
        else:
            self.ref_model = create_reference_model(model)
        
        if processing_class is None:
            raise ValueError("processing_class must be specified.")
        
        max_length = args.max_length or 1024
        max_prompt_length = args.max_prompt_length or 512
        
        # Setup data collator
        if data_collator is None:
            data_collator = LatentKTODataCollator(
                pad_token_id=processing_class.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
            )
            if args.remove_unused_columns:
                args.remove_unused_columns = False
        
        # Disable dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        
        # Store config
        self.loss_type = args.loss_type
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else processing_class.pad_token_id
        self.truncation_mode = args.truncation_mode
        self.processing_class = processing_class
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        
        # KL calculation
        self.calculate_KL = True
        if self.loss_type in ["apo_zero_unpaired"]:
            self.calculate_KL = False
        
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        # Store args for dataset preparation (before parent __init__ is called)
        self._dataset_num_proc = args.dataset_num_proc
        self._per_device_train_batch_size = args.per_device_train_batch_size
        
        # Suppress estimate_tokens warning
        model.warnings_issued["estimate_tokens"] = True
        
        # Process datasets
        with PartialState().main_process_first():
            train_dataset = self._prepare_dataset(train_dataset, "train")
            if eval_dataset is not None:
                eval_dataset = self._prepare_dataset(eval_dataset, "eval")
        
        # Print configuration
        print("\n" + "="*80)
        print("LASA LatentKTOTrainer Configuration:")
        print("="*80)
        print(f"  - safety_unsafe_ratio: {self.safety_unsafe_ratio}")
        print(f"  - ultrafeedback_unsafe_ratio: {self.ultrafeedback_unsafe_ratio}")
        print(f"  - random_seed: {self.random_seed}")
        print(f"  - beta: {self.beta}")
        print(f"  - loss_type: {self.loss_type}")
        print(f"  - safety_prefix: '{SAFETY_PREFIX}'")
        print("="*80 + "\n")
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        self.model_accepts_loss_kwargs = False
        
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
        
        if not hasattr(self, "accelerator"):
            raise AttributeError("Trainer does not have accelerator. Upgrade transformers.")
        
        # Handle DeepSpeed
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError("Cannot use precompute_ref_log_probs with DeepSpeed ZeRO-3.")
        
        # For ZeRO-3, load ref_model now that accelerator is available
        if self.ref_model is None and self._is_deepspeed_zero3 and not self.is_peft_model and not self.precompute_ref_log_probs:
            # Load reference model with ZeRO-3 initialization context
            import deepspeed
            from transformers.integrations.deepspeed import HfDeepSpeedConfig
            
            # Get deepspeed config from accelerator
            ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
            
            # Create HfDeepSpeedConfig to enable ZeRO-3 parameter partitioning during from_pretrained
            dschf = HfDeepSpeedConfig(ds_config)
            
            # Load the reference model - it will be automatically partitioned by ZeRO-3
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self._ref_model_path,
                **self._ref_model_init_kwargs,
            )
            
            if self.args.disable_dropout:
                disable_dropout_in_model(self.ref_model)
        
        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError("No reference model and model is not Peft. Try precompute_ref_log_probs=True.")
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
    
    def _unpair_preference_dataset_with_source(self, dataset: Dataset, num_proc: int = None, desc: str = None) -> Dataset:
        """
        Custom unpair function that preserves the 'source' column.
        
        Converts a paired preference dataset with 'chosen'/'rejected' columns to an unpaired
        dataset with 'completion'/'label' columns, while properly duplicating the 'source' column.
        """
        def _unpair_row_with_source(examples):
            batch_size = len(examples["prompt"])
            
            new_prompt = []
            new_completion = []
            new_label = []
            new_source = []
            
            for i in range(batch_size):
                # Chosen sample
                new_prompt.append(examples["prompt"][i])
                new_completion.append(examples["chosen"][i])
                new_label.append(True)
                if "source" in examples:
                    new_source.append(examples["source"][i])
                
                # Rejected sample
                new_prompt.append(examples["prompt"][i])
                new_completion.append(examples["rejected"][i])
                new_label.append(False)
                if "source" in examples:
                    new_source.append(examples["source"][i])
            
            result = {
                "prompt": new_prompt,
                "completion": new_completion,
                "label": new_label,
            }
            if "source" in examples:
                result["source"] = new_source
            
            return result
        
        # Determine columns to remove (all except prompt, chosen, rejected, source)
        columns_to_remove = [c for c in dataset.column_names if c not in ["prompt", "chosen", "rejected", "source"]]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Check if dataset needs unpairing
        if "chosen" in dataset.column_names and "rejected" in dataset.column_names:
            dataset = dataset.map(
                _unpair_row_with_source, 
                batched=True, 
                remove_columns=["chosen", "rejected"],
                num_proc=num_proc, 
                desc=desc
            )
        
        return dataset
    
    def _prepare_dataset(self, dataset: Dataset, split: str) -> Dataset:
        """
        Prepare dataset with:
        1. Tokenization with both normal and with_prefix versions
        2. Source-aware sampling of LASA unsafe semantic labels
        3. KL dataset for KTO loss
        """
        tokenizer = self.processing_class
        max_length = self.max_length
        max_prompt_length = self.max_prompt_length
        rng = random.Random(self.random_seed)
        safety_ratio = self.safety_unsafe_ratio
        ultrafeedback_ratio = self.ultrafeedback_unsafe_ratio
        dataset_num_proc = self._dataset_num_proc
        
        # Extract prompt if needed (handles chosen/rejected format)
        dataset = dataset.map(
            maybe_extract_prompt,
            num_proc=dataset_num_proc,
            desc=f"Extracting prompt from {split} dataset"
        )
        
        # Unpair the dataset (converts chosen/rejected to completion/label format)
        # Use custom function that preserves 'source' column
        dataset = self._unpair_preference_dataset_with_source(
            dataset, num_proc=dataset_num_proc, desc=f"Unpairing {split} dataset"
        )
        
        # Apply chat template if needed
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=dataset_num_proc,
            desc=f"Applying chat template to {split} dataset",
        )
        
        def tokenize_and_label(example, idx):
            """Tokenize and create both normal and with_prefix versions."""
            prompt = example["prompt"]
            completion = example["completion"]
            label = example.get("label", True)  # True = chosen, False = rejected
            source = example.get("source", "unknown")
            
            # Determine whether this sample gets the LASA conditional prompt.
            if source == "safety":
                is_unsafe = 1 if rng.random() < safety_ratio else 0
            elif source == "ultrafeedback":
                is_unsafe = 1 if rng.random() < ultrafeedback_ratio else 0
            else:
                is_unsafe = 0
            
            # Tokenize prompt
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be str but got {type(prompt)}")
            if not isinstance(completion, str):
                raise ValueError(f"completion should be str but got {type(completion)}")
            
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)
            prompt_input_ids = prompt_tokens["input_ids"]
            prompt_attention_mask = prompt_tokens["attention_mask"]
            
            # Tokenize completion
            prompt_and_completion = prompt + completion
            full_tokens = tokenizer(prompt_and_completion, add_special_tokens=False)
            full_input_ids = full_tokens["input_ids"]
            full_attention_mask = full_tokens["attention_mask"]
            
            answer_input_ids = full_input_ids[len(prompt_input_ids):]
            answer_attention_mask = full_attention_mask[len(prompt_input_ids):]
            
            # Handle token merging issues
            response_token_ids_start_idx = len(prompt_input_ids)
            if not np.array_equal(prompt_input_ids, full_input_ids[:response_token_ids_start_idx]):
                response_token_ids_start_idx -= 1
            
            prompt_input_ids = full_input_ids[:response_token_ids_start_idx]
            prompt_attention_mask = full_attention_mask[:response_token_ids_start_idx]
            answer_input_ids = full_input_ids[response_token_ids_start_idx:]
            answer_attention_mask = full_attention_mask[response_token_ids_start_idx:]
            
            # Build completion sequences
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
            
            completion_input_ids = list(prompt_input_ids) + list(answer_input_ids)
            completion_attention_mask = list(prompt_attention_mask) + list(answer_attention_mask)
            
            # Add BOS if needed
            if bos_token_id is not None:
                if len(prompt_input_ids) == 0 or bos_token_id != prompt_input_ids[0]:
                    prompt_input_ids = [bos_token_id] + list(prompt_input_ids)
                    prompt_attention_mask = [1] + list(prompt_attention_mask)
                    completion_input_ids = [bos_token_id] + completion_input_ids
                    completion_attention_mask = [1] + completion_attention_mask
            
            # Add EOS if needed
            if len(answer_input_ids) == 0 or eos_token_id != answer_input_ids[-1]:
                completion_input_ids = completion_input_ids + [eos_token_id]
                completion_attention_mask = completion_attention_mask + [1]
            
            # Create labels (mask prompt tokens)
            completion_labels = list(completion_input_ids)
            completion_labels[:len(prompt_input_ids)] = [self.label_pad_token_id] * len(prompt_input_ids)
            
            # Truncation
            if len(completion_input_ids) > max_length:
                # First truncate prompt
                excess = len(completion_input_ids) - max_length
                if self.truncation_mode == "keep_start":
                    prompt_input_ids = prompt_input_ids[:max(1, len(prompt_input_ids) - excess)]
                    prompt_attention_mask = prompt_attention_mask[:len(prompt_input_ids)]
                else:
                    prompt_input_ids = prompt_input_ids[excess:]
                    prompt_attention_mask = prompt_attention_mask[excess:]
                
                # Rebuild completion with truncated prompt
                completion_input_ids = list(prompt_input_ids) + list(answer_input_ids)
                completion_attention_mask = list(prompt_attention_mask) + list(answer_attention_mask)
                if len(answer_input_ids) == 0 or eos_token_id != answer_input_ids[-1]:
                    completion_input_ids = completion_input_ids + [eos_token_id]
                    completion_attention_mask = completion_attention_mask + [1]
                completion_labels = list(completion_input_ids)
                completion_labels[:len(prompt_input_ids)] = [self.label_pad_token_id] * len(prompt_input_ids)
                
                # If still too long, truncate answer
                if len(completion_input_ids) > max_length:
                    completion_input_ids = completion_input_ids[:max_length]
                    completion_attention_mask = completion_attention_mask[:max_length]
                    completion_labels = completion_labels[:max_length]
            
            # === WITH PREFIX VERSION ===
            prefix_tokens = tokenizer(SAFETY_PREFIX, add_special_tokens=False)["input_ids"]
            
            # Insert prefix after prompt
            prompt_len = len(prompt_input_ids)
            completion_input_ids_wp = (
                completion_input_ids[:prompt_len] + 
                prefix_tokens + 
                completion_input_ids[prompt_len:]
            )
            completion_attention_mask_wp = (
                completion_attention_mask[:prompt_len] + 
                [1] * len(prefix_tokens) + 
                completion_attention_mask[prompt_len:]
            )
            completion_labels_wp = (
                [self.label_pad_token_id] * (prompt_len + len(prefix_tokens)) + 
                completion_input_ids[prompt_len:]
            )
            
            # Truncate with_prefix version if needed
            if len(completion_input_ids_wp) > max_length:
                completion_input_ids_wp = completion_input_ids_wp[:max_length]
                completion_attention_mask_wp = completion_attention_mask_wp[:max_length]
                completion_labels_wp = completion_labels_wp[:max_length]
            
            return {
                "prompt": prompt,
                "completion": completion,
                "label": label,
                "is_unsafe": is_unsafe,
                "prompt_input_ids": prompt_input_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "completion_input_ids": completion_input_ids,
                "completion_attention_mask": completion_attention_mask,
                "completion_labels": completion_labels,
                "completion_input_ids_with_prefix": completion_input_ids_wp,
                "completion_attention_mask_with_prefix": completion_attention_mask_wp,
                "completion_labels_with_prefix": completion_labels_wp,
                "answer_input_ids": list(answer_input_ids),
                "answer_attention_mask": list(answer_attention_mask),
            }
        
        dataset = dataset.map(
            tokenize_and_label,
            with_indices=True,
            num_proc=dataset_num_proc,
            desc=f"Tokenizing {split} dataset with latent labels",
        )
        
        # Filter empty samples
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: len(x["completion_input_ids"]) > 0)
        if len(dataset) < original_len:
            print(f"[LASA LatentKTO] Filtered {original_len - len(dataset)} empty samples from {split}")
        
        # Create KL dataset if needed
        if self.calculate_KL:
            if self._per_device_train_batch_size <= 1:
                raise ValueError("Batch size must be > 1 for KTO KL term computation.")
            
            def get_kl_dataset(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
                """Create mismatched pairs for KL estimation."""
                batch["answer_input_ids"] = [batch["answer_input_ids"][-1]] + batch["answer_input_ids"][:-1]
                batch["answer_attention_mask"] = [batch["answer_attention_mask"][-1]] + batch["answer_attention_mask"][:-1]
                return batch
            
            kl_dataset = dataset.map(
                get_kl_dataset,
                batched=True,
                batch_size=self._per_device_train_batch_size,
                num_proc=dataset_num_proc,
                desc=f"Creating KL {split} dataset",
            )
            
            def process_kl_tokens(example):
                """Process KL tokens with prefix versions."""
                prompt_input_ids = example["prompt_input_ids"]
                answer_input_ids = example["answer_input_ids"]
                answer_attention_mask = example["answer_attention_mask"]
                is_unsafe = example["is_unsafe"]
                
                bos_token_id = tokenizer.bos_token_id
                eos_token_id = tokenizer.eos_token_id
                
                # Build KL completion
                kl_completion_input_ids = list(prompt_input_ids) + list(answer_input_ids)
                kl_completion_attention_mask = [1] * len(prompt_input_ids) + list(answer_attention_mask)
                
                # Add EOS if needed
                if len(answer_input_ids) == 0 or eos_token_id != answer_input_ids[-1]:
                    kl_completion_input_ids = kl_completion_input_ids + [eos_token_id]
                    kl_completion_attention_mask = kl_completion_attention_mask + [1]
                
                # Labels
                kl_completion_labels = list(kl_completion_input_ids)
                kl_completion_labels[:len(prompt_input_ids)] = [self.label_pad_token_id] * len(prompt_input_ids)
                
                # Truncate if needed
                if len(kl_completion_input_ids) > self.max_length:
                    kl_completion_input_ids = kl_completion_input_ids[:self.max_length]
                    kl_completion_attention_mask = kl_completion_attention_mask[:self.max_length]
                    kl_completion_labels = kl_completion_labels[:self.max_length]
                
                # With prefix version
                prefix_tokens = tokenizer(SAFETY_PREFIX, add_special_tokens=False)["input_ids"]
                prompt_len = len(prompt_input_ids)
                
                kl_completion_input_ids_wp = (
                    kl_completion_input_ids[:prompt_len] + 
                    prefix_tokens + 
                    kl_completion_input_ids[prompt_len:]
                )
                kl_completion_attention_mask_wp = (
                    kl_completion_attention_mask[:prompt_len] + 
                    [1] * len(prefix_tokens) + 
                    kl_completion_attention_mask[prompt_len:]
                )
                kl_completion_labels_wp = (
                    [self.label_pad_token_id] * (prompt_len + len(prefix_tokens)) + 
                    kl_completion_input_ids[prompt_len:]
                )
                
                if len(kl_completion_input_ids_wp) > self.max_length:
                    kl_completion_input_ids_wp = kl_completion_input_ids_wp[:self.max_length]
                    kl_completion_attention_mask_wp = kl_completion_attention_mask_wp[:self.max_length]
                    kl_completion_labels_wp = kl_completion_labels_wp[:self.max_length]
                
                return {
                    "KL_completion_input_ids": kl_completion_input_ids,
                    "KL_completion_attention_mask": kl_completion_attention_mask,
                    "KL_completion_labels": kl_completion_labels,
                    "KL_completion_input_ids_with_prefix": kl_completion_input_ids_wp,
                    "KL_completion_attention_mask_with_prefix": kl_completion_attention_mask_wp,
                    "KL_completion_labels_with_prefix": kl_completion_labels_wp,
                }
            
            kl_dataset = kl_dataset.map(
                process_kl_tokens,
                num_proc=dataset_num_proc,
                remove_columns=[c for c in kl_dataset.column_names if c in dataset.column_names],
                desc=f"Processing KL {split} tokens",
            )
            
            # Merge datasets
            dataset = concatenate_datasets([dataset, kl_dataset], axis=1)
        
        # Log statistics
        unsafe_count = sum(1 for i in range(len(dataset)) if dataset[i]["is_unsafe"] == 1)
        label_true_count = sum(1 for i in range(len(dataset)) if dataset[i]["label"] is True)
        print(f"\n[LASA LatentKTO {split} Dataset Statistics]")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Unsafe samples: {unsafe_count} ({100*unsafe_count/len(dataset):.1f}%)")
        print(f"  Desirable (label=True): {label_true_count} ({100*label_true_count/len(dataset):.1f}%)")
        print(f"  Undesirable (label=False): {len(dataset) - label_true_count}")
        
        return dataset
    
    @contextmanager
    def null_ref_context(self):
        """Context manager for null reference model (peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")
    
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute log probabilities of given labels under given logits."""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have same batch/seq dimensions.")
        
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            labels = labels.clone()
        
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        
        per_token_logps = selective_log_softmax(logits, labels)
        
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
    
    def _select_inputs_by_unsafe(self, batch: dict, prefix: str = "") -> tuple:
        """
        Select input_ids/attention_mask/labels based on the sampled unsafe semantic label.
        Unsafe samples use the conditional-prompt version.
        """
        is_unsafe = batch["is_unsafe"]
        batch_size = is_unsafe.size(0)
        device = is_unsafe.device
        
        input_ids_key = f"{prefix}completion_input_ids"
        attention_mask_key = f"{prefix}completion_attention_mask"
        labels_key = f"{prefix}completion_labels"
        
        input_ids = batch[input_ids_key]
        attention_mask = batch[attention_mask_key]
        labels = batch[labels_key]
        
        input_ids_wp = batch.get(f"{input_ids_key}_with_prefix")
        attention_mask_wp = batch.get(f"{attention_mask_key}_with_prefix")
        labels_wp = batch.get(f"{labels_key}_with_prefix")
        
        if input_ids_wp is None or is_unsafe.sum() == 0:
            return input_ids, attention_mask, labels
        
        # Build final tensors by selecting per sample
        final_input_ids_list = []
        final_attention_mask_list = []
        final_labels_list = []
        
        for i in range(batch_size):
            if is_unsafe[i] == 1:
                final_input_ids_list.append(input_ids_wp[i])
                final_attention_mask_list.append(attention_mask_wp[i])
                final_labels_list.append(labels_wp[i])
            else:
                final_input_ids_list.append(input_ids[i])
                final_attention_mask_list.append(attention_mask[i])
                final_labels_list.append(labels[i])
        
        # Re-pad to same length
        max_len = max(t.size(0) for t in final_input_ids_list)
        
        final_input_ids = torch.stack([
            F.pad(t, (0, max_len - t.size(0)), value=self.padding_value)
            for t in final_input_ids_list
        ])
        final_attention_mask = torch.stack([
            F.pad(t, (0, max_len - t.size(0)), value=0)
            for t in final_attention_mask_list
        ])
        final_labels = torch.stack([
            F.pad(t, (0, max_len - t.size(0)), value=self.label_pad_token_id)
            for t in final_labels_list
        ])
        
        return final_input_ids, final_attention_mask, final_labels
    
    def forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple:
        """Forward pass with latent selection."""
        # Select inputs according to LASA's sampled unsafe semantic label.
        completion_input_ids, completion_attention_mask, completion_labels = self._select_inputs_by_unsafe(batch, "")
        
        # Forward through model
        outputs = model(
            completion_input_ids,
            attention_mask=completion_attention_mask,
        )
        completion_logits = outputs.logits
        
        completion_logps = self.get_batch_logps(
            completion_logits,
            completion_labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        
        # Split by label (chosen/rejected)
        labels = torch.tensor(batch["label"], dtype=torch.bool, device=completion_logps.device)
        chosen_idx = labels.nonzero(as_tuple=True)[0]
        rejected_idx = (~labels).nonzero(as_tuple=True)[0]
        
        chosen_logps = completion_logps[chosen_idx]
        rejected_logps = completion_logps[rejected_idx]
        chosen_logits = completion_logits[chosen_idx]
        rejected_logits = completion_logits[rejected_idx]
        
        # KL logps
        KL_logps = None
        if self.calculate_KL and "KL_completion_input_ids" in batch:
            kl_input_ids, kl_attention_mask, kl_labels = self._select_inputs_by_unsafe(batch, "KL_")
            
            with torch.no_grad():
                kl_logits = model(kl_input_ids, attention_mask=kl_attention_mask).logits
            
            KL_logps = self.get_batch_logps(
                kl_logits,
                kl_labels,
                average_log_prob=False,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps
    
    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute KTO loss."""
        if self.calculate_KL:
            kl = (policy_KL_logps - reference_KL_logps).mean().detach()
            kl = self.accelerator.gather_for_metrics(kl).mean().clamp(min=0)
        else:
            kl = torch.zeros(1).to(policy_chosen_logps.device)
        
        # Chosen losses
        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            if self.loss_type == "kto":
                chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            elif self.loss_type == "apo_zero_unpaired":
                chosen_losses = 1 - F.sigmoid(self.beta * chosen_logratios)
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)
        
        # Rejected losses
        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            if self.loss_type == "kto":
                rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            elif self.loss_type == "apo_zero_unpaired":
                rejected_losses = F.sigmoid(self.beta * rejected_logratios)
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            rejected_losses = torch.Tensor([]).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.accelerator.device)
        
        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
            0,
        )
        
        return losses, chosen_rewards, rejected_rewards, kl
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
    ):
        """Compute KTO loss and metrics for batch."""
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        labels = torch.tensor(batch["label"], dtype=torch.bool)
        num_chosen = labels.sum().to(self.accelerator.device)
        num_rejected = (len(labels) - num_chosen).to(self.accelerator.device)
        
        # Log conditional-prompt stats.
        is_unsafe = batch["is_unsafe"]
        num_unsafe = (is_unsafe == 1).sum().item()
        
        # Forward pass
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logps,
        ) = self.forward(model, batch)
        
        # Reference model forward
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        reference_KL_logps,
                    ) = self.forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    reference_KL_logps,
                ) = self.forward(self.ref_model, batch)
        
        # Compute KTO loss
        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )
        
        metrics["kl"] = kl.item()
        metrics["num_unsafe"] = num_unsafe
        
        all_num_chosen = self.accelerator.gather_for_metrics(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather_for_metrics(num_rejected).sum().item()
        
        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = self.accelerator.gather_for_metrics(chosen_rewards.nansum()).nansum().item()
            metrics["logps/chosen_sum"] = self.accelerator.gather_for_metrics(policy_chosen_logps.nansum()).nansum().item()
            metrics["logits/chosen_sum"] = self.accelerator.gather_for_metrics(policy_chosen_logits.nansum()).nansum().item()
            metrics["count/chosen"] = all_num_chosen
        
        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = self.accelerator.gather_for_metrics(rejected_rewards.nansum()).nansum().item()
            metrics["logps/rejected_sum"] = self.accelerator.gather_for_metrics(policy_rejected_logps.nansum()).nansum().item()
            metrics["logits/rejected_sum"] = self.accelerator.gather_for_metrics(policy_rejected_logits.nansum()).nansum().item()
            metrics["count/rejected"] = all_num_rejected
        
        loss = losses.nanmean()
        return loss, metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)
        
        loss = loss.to(self.args.device)
        
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss
    
    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
    
    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if dataset is None:
            dataset = self.train_dataset
        if dataset is None or not has_length(dataset):
            return None
        return SequentialSampler(dataset)
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)
        
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")
        
        if prediction_loss_only:
            return (loss.detach(), None, None)
        
        logits_dict = {}
        if "logits/chosen_sum" in metrics:
            logits_dict["eval_logits/chosen"] = metrics["logits/chosen_sum"]
        if "logits/rejected_sum" in metrics:
            logits_dict["eval_logits/rejected"] = metrics["logits/rejected_sum"]
        
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
        
        return (loss.detach(), logits, labels)
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics including stored ones."""
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        
        # Accumulate average metrics
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._stored_metrics[train_eval]:
                count_sum = torch.Tensor(self._stored_metrics[train_eval][f"count/{split}"]).sum().item()
                for metric in ["rewards", "logps", "logits"]:
                    if f"{metric}/{split}_sum" in self._stored_metrics[train_eval]:
                        logs[f"{prefix}{metric}/{split}"] = (
                            torch.Tensor(self._stored_metrics[train_eval][f"{metric}/{split}_sum"]).sum().item()
                            / count_sum
                        )
                        del self._stored_metrics[train_eval][f"{metric}/{split}_sum"]
                del self._stored_metrics[train_eval][f"count/{split}"]
        
        # Reward margin
        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:
            logs[f"{prefix}rewards/margins"] = logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]
        
        # Add remaining averaged metrics
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[f"{prefix}{key}"] = torch.Tensor(metrics).mean().item()
        
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)
