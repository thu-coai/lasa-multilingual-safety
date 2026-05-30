#!/usr/bin/env python3
"""
Generate responses for LASA with optional SSI-conditioned prompt injection.

LASA inference flow:
    1. Extract the prompt representation at the semantic bottleneck layer.
    2. Use the Safety Semantic Interpreter (SSI) to predict harmful semantics.
    3. For harmful prompts, append the paper's conditional safety prompt.
    4. Generate responses with vLLM

Usage:
    # Normal generation
    python generate.py --base_model MODEL_PATH --input_file INPUT --output_file OUTPUT
    
    # With LASA/SSI conditioning
    python generate.py --base_model MODEL_PATH --input_file INPUT --output_file OUTPUT \
        --use_safety_prefix --classifier_dir CLASSIFIER_DIR
"""
import gc
import json
import argparse
import os
import sys
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from loguru import logger
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vllm_model import VLLMModel
from models.safety_classifier import load_safety_classifier


# LASA conditional generation prompt for harmful semantics detected by SSI.
SAFETY_PREFIX = "Harmful query detected. I should refuse this request and provide a safe response in the user's language."


def get_chat_template_kwargs(model_path: str) -> dict:
    """Get additional kwargs for apply_chat_template based on model type."""
    if model_path and "qwen3" in model_path.lower():
        return {"enable_thinking": True}
    return {}


def extract_hidden_states_and_predict_safety(
    hf_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    classifier,
    layer_idx: int,
    threshold: float,
    batch_size: int = 4,
    device: str = 'cuda:0',
) -> Tuple[List[bool], List[float]]:
    """
    Extract bottleneck hidden states and run SSI safety prediction.
    
    Args:
        hf_model: HuggingFace model for hidden state extraction
        tokenizer: Tokenizer
        prompts: List of prompts
        classifier: Safety Semantic Interpreter module
        layer_idx: Semantic bottleneck layer index for hidden state extraction
        threshold: Threshold for safety prediction
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        Tuple of (is_harmful_list, probability_list)
    """
    is_harmful_list = []
    probs_list = []
    
    num_layers = hf_model.config.num_hidden_layers
    target_layer = layer_idx if layer_idx >= 0 else layer_idx + num_layers
    
    if target_layer < 0 or target_layer >= num_layers:
        raise ValueError(f"layer_idx {layer_idx} out of range for {num_layers} layers")
    
    # hidden_states[0] = embedding, hidden_states[i+1] = layer i output
    hidden_state_idx = target_layer + 1
    
    logger.info(f"Extracting semantic bottleneck states from layer {target_layer} for SSI prediction...")
    
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Safety prediction", total=total_batches):
        batch_prompts = prompts[start_idx:start_idx + batch_size]
        
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(device)
        
        with torch.no_grad():
            outputs = hf_model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                output_hidden_states=True,
            )
            
            hidden_states = outputs.hidden_states[hidden_state_idx]
            
            # Get last valid token position for each sample
            seq_lengths = tokenized.attention_mask.sum(dim=1)
            last_positions = (seq_lengths - 1).clamp(min=0)
            
            batch_size_actual = hidden_states.size(0)
            hidden_vecs = hidden_states[
                torch.arange(batch_size_actual, device=device),
                last_positions
            ]
            
            predictions, probs = classifier.predict_safety(
                hidden_vecs.float(), threshold=threshold
            )
            
            is_harmful_list.extend(predictions.cpu().tolist())
            probs_list.extend(probs.cpu().tolist())
        
        del tokenized, outputs, hidden_states, hidden_vecs
        torch.cuda.empty_cache()
    
    num_harmful = sum(is_harmful_list)
    logger.info(f"Safety prediction: {num_harmful} harmful, {len(is_harmful_list) - num_harmful} benign")
    
    return [bool(x) for x in is_harmful_list], [float(p) for p in probs_list]


def prepare_prompts_with_safety_prefix(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    is_harmful_list: List[bool],
    model_path: str,
) -> List[str]:
    """
    Prepare prompts with the LASA conditional safety prompt for harmful semantics.
    """
    chat_kwargs = get_chat_template_kwargs(model_path)
    final_prompts = []
    
    for prompt, is_harmful in zip(prompts, is_harmful_list):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs
        )
        
        if is_harmful:
            formatted = formatted + SAFETY_PREFIX
        
        final_prompts.append(formatted)
    
    return final_prompts


def release_model(model, *others):
    """Release model and clean up GPU memory."""
    model.cpu()
    del model
    for obj in others:
        if obj is not None:
            if hasattr(obj, 'cpu'):
                obj.cpu()
            del obj
    
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


def load_data(input_file: str, prompt_key: str = 'prompt', limit: int = 0) -> Tuple[List[dict], List[str]]:
    """Load data from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit > 0:
        data = data[:limit]
    
    prompts = [item[prompt_key] for item in data]
    return data, prompts


def save_results(results: List[dict], output_file: str):
    """Save results to JSON file."""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def generate_with_safety_prefix(
    args,
    tokenizer: AutoTokenizer,
    data: List[dict],
    prompts: List[str],
) -> List[dict]:
    """
    Generate responses with LASA/SSI conditional prompt injection.
    
    Flow:
    1. Load HF model and predict harmful semantics with SSI
    2. Release HF model
    3. Load vLLM and generate
    """
    device = "cuda:0"
    
    # Step 1: Load SSI (config.json provides bottleneck layer_idx and threshold).
    logger.info("Step 1: Loading SSI and HF model for safety prediction...")
    
    classifier, layer_idx, threshold = load_safety_classifier(args.classifier_dir, device)
    
    logger.info(f"Layer index: {layer_idx}, Threshold: {threshold}")
    
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    hf_model.eval()
    
    # Predict safety
    is_harmful_list, safety_probs = extract_hidden_states_and_predict_safety(
        hf_model=hf_model,
        tokenizer=tokenizer,
        prompts=prompts,
        classifier=classifier,
        layer_idx=layer_idx,
        threshold=threshold,
        batch_size=args.latent_batch_size,
        device=device,
    )
    
    # Step 2: Release HF model
    logger.info("Step 2: Releasing HF model...")
    release_model(hf_model, classifier)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory after release: {allocated:.2f}GB")
    
    # Step 3: Prepare prompts
    logger.info("Step 3: Preparing prompts with LASA conditional safety prompt...")
    final_prompts = prepare_prompts_with_safety_prefix(
        tokenizer, prompts, is_harmful_list, args.base_model
    )
    
    # Log examples
    for i in range(min(3, len(final_prompts))):
        status = "HARMFUL" if is_harmful_list[i] else "benign"
        logger.info(f"  [{i}] {status}: {final_prompts[i][:150]}...")
    
    # Step 4: Load vLLM and generate
    logger.info("Step 4: Loading vLLM model...")
    llm_kwargs = {
        "model": args.base_model,
        "tokenizer": args.tokenizer_path,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        frequency_penalty=args.frequency_penalty,
    )
    
    logger.info("Generating responses...")
    outputs = llm.generate(final_prompts, generation_config)
    responses = [o.outputs[0].text for o in outputs]
    
    # Build results
    results = []
    for item, response, is_harmful, prob in zip(data, responses, is_harmful_list, safety_probs):
        result = dict(item)
        result["answer"] = response.strip()
        result["safety_prefix_used"] = is_harmful
        result["safety_score"] = prob
        results.append(result)
    
    return results


def generate_normal(
    args,
    tokenizer: AutoTokenizer,
    data: List[dict],
    prompts: List[str],
) -> List[dict]:
    """Generate responses without SSI conditioning."""
    logger.info(f"Loading vLLM model from {args.base_model}")
    
    llm_kwargs = {
        "model": args.base_model,
        "tokenizer": args.tokenizer_path,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        frequency_penalty=args.frequency_penalty,
    )
    
    chat_model = VLLMModel(
        model=llm,
        tokenizer=tokenizer,
        model_name=args.base_model,
        generation_config=generation_config
    )
    
    logger.info("Generating responses...")
    responses = chat_model.batch_chat(prompts)
    
    # Build results
    results = []
    for item, response in zip(data, responses):
        result = dict(item)
        result["answer"] = response.strip()
        results.append(result)
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses with optional LASA/SSI conditioning")
    
    # Model configuration
    parser.add_argument('--base_model', '--model_path', type=str, required=True,
                        dest='base_model', help='Path to the base model')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to tokenizer (defaults to base_model)')
    
    # Input/Output
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--prompt_key', type=str, default='prompt',
                        help='Key to extract prompt from input JSON items')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of samples (0 = all)')
    
    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.80)
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    
    # Safety prefix options
    parser.add_argument('--use_safety_prefix', action='store_true',
                        help='Enable LASA/SSI conditional prompt injection')
    parser.add_argument('--classifier_dir', type=str, default=None,
                        help='Path to classifier directory (containing checkpoint.pt and config.json)')
    parser.add_argument('--latent_batch_size', type=int, default=8,
                        help='Batch size for safety prediction')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default tokenizer path
    if not args.tokenizer_path:
        args.tokenizer_path = args.base_model
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, padding_side='left', trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    data, prompts = load_data(args.input_file, args.prompt_key, args.limit)
    logger.info(f"Loaded {len(data)} samples")
    
    # Generate
    if args.use_safety_prefix and args.classifier_dir:
        logger.info("Running with LASA/SSI conditional prompt injection")
        results = generate_with_safety_prefix(args, tokenizer, data, prompts)
    else:
        logger.info("Running normal generation")
        results = generate_normal(args, tokenizer, data, prompts)
    
    # Save results
    save_results(results, args.output_file)
    logger.info(f"Saved {len(results)} results to {args.output_file}")


if __name__ == "__main__":
    main()
