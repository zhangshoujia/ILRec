#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA inference script
Loads saved LoRA fine-tuned parameters and generates user/item embeddings.
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
from typing import List, Dict, Any

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoConfig
from llm.arguments import ModelArguments, DataTrainingArguments
from llm.data_processor.user_item_llama import UserItemDataProcessor
from llm.llama import UserItemPromptLlama, LlamaRSEmb
from llm.lora_cls import PeftModelForCLS
from llm.peft.tuners.lora import LoraConfig


def load_model_and_tokenizer(model_args: ModelArguments, model_path: str):
    """
    Load model and tokenizer.
    
    Args:
        model_args: model arguments
        model_path: path to the saved model
    
    Returns:
        model: loaded model
        tokenizer: tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    
    # Load config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Load base model
    if model_args.model_choice == "useritem_prompt":
        model = UserItemPromptLlama.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            pool_type=model_args.pool_type,
            tau=model_args.tau,
            user_prompt_length=model_args.user_prompt_length,
            item_prompt_length=model_args.item_prompt_length,
        )
    else:
        model = LlamaRSEmb.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            pool_type=model_args.pool_type,
            tau=model_args.tau,
        )
    
    # Load LoRA config (if needed)
    if hasattr(model_args, 'lora_rank') and model_args.lora_rank > 0:
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.trainable.split(","),
            lora_dropout=model_args.lora_dropout,
            task_type="SEQ_CLS",
        )
        model = PeftModelForCLS(model, peft_config)
        
        # Load saved LoRA weights
        adapter_path = os.path.join(model_path, "adapter_model")
        if os.path.exists(adapter_path):
            # Load weights saved by PEFT
            model.load_adapter(adapter_path)
            print(f"Loaded LoRA adapter from {adapter_path}")
        elif os.path.exists(os.path.join(model_path, "trainable_weights.bin")):
            # Load manually saved trainable weights
            trainable_weights = torch.load(os.path.join(model_path, "trainable_weights.bin"))
            model.load_state_dict(trainable_weights, strict=False)
            print(f"Loaded trainable weights from {model_path}")
        else:
            print(f"Warning: No saved weights found in {model_path}")
    
    # Eval mode and device
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, tokenizer


def process_data(data: List[Dict[str, Any]], tokenizer, data_args: DataTrainingArguments, 
                model_args: ModelArguments) -> List[Dict[str, Any]]:
    """
    Process raw data to model inputs.
    
    Args:
        data: raw data
        tokenizer: tokenizer
        data_args: data arguments
        model_args: model arguments
    
    Returns:
        processed data list
    """
    data_processor = UserItemDataProcessor(
        tokenizer=tokenizer,
        max_length=data_args.max_source_length,
        user_prompt_length=model_args.user_prompt_length,
        item_prompt_length=model_args.item_prompt_length,
    )
    
    processed_data = []
    for item in data:
        processed_item = data_processor.process_single_item(item)
        processed_data.append(processed_item)
    
    return processed_data


def generate_embeddings(model, processed_data: List[Dict[str, Any]]) -> tuple:
    """
    Generate embeddings.
    
    Args:
        model: model
        processed_data: processed input data
    
    Returns:
        user_embeddings: user embeddings
        item_embeddings: item embeddings
    """
    user_embeddings = []
    item_embeddings = []
    
    with torch.no_grad():
        for item in processed_data:
            # Prepare inputs
            user_input_ids = torch.tensor([item['user_input_ids']]).to(model.device)
            item_input_ids = torch.tensor([item['item_input_ids']]).to(model.device)
            user_attention_mask = torch.tensor([item['user_attention_mask']]).to(model.device)
            item_attention_mask = torch.tensor([item['item_attention_mask']]).to(model.device)
            
            # Forward pass
            outputs = model(
                user_input_ids=user_input_ids,
                item_input_ids=item_input_ids,
                user_attention_mask=user_attention_mask,
                item_attention_mask=item_attention_mask,
            )
            
            # Extract embeddings
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            batch_size = logits.shape[0] // 2
            user_emb = logits[:batch_size].cpu().numpy()
            item_emb = logits[batch_size:].cpu().numpy()
            
            user_embeddings.append(user_emb)
            item_embeddings.append(item_emb)
    
    # Concatenate all
    user_embeddings = np.concatenate(user_embeddings, axis=0)
    item_embeddings = np.concatenate(item_embeddings, axis=0)
    
    return user_embeddings, item_embeddings


def main():
    parser = argparse.ArgumentParser(description="LoRA inference script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model path")
    parser.add_argument("--model_choice", type=str, default="useritem_prompt", help="Model choice")
    parser.add_argument("--max_source_length", type=int, default=512, help="Max input length")
    parser.add_argument("--user_prompt_length", type=int, default=10, help="User soft prompt length")
    parser.add_argument("--item_prompt_length", type=int, default=10, help="Item soft prompt length")
    parser.add_argument("--pool_type", type=str, default="last", help="Pooling type")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--trainable", type=str, default="q_proj,v_proj", help="Trainable modules (comma-separated)")
    
    args = parser.parse_args()
    
    # Ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build argument objects
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        model_choice=args.model_choice,
        user_prompt_length=args.user_prompt_length,
        item_prompt_length=args.item_prompt_length,
        pool_type=args.pool_type,
        tau=args.tau,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        trainable=args.trainable,
    )
    
    data_args = DataTrainingArguments(
        max_source_length=args.max_source_length,
    )
    
    # Load model & tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args, args.model_path)
    
    # Load data
    print("Loading data...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process data
    print("Processing data...")
    processed_data = process_data(data, tokenizer, data_args, model_args)
    
    # Generate embeddings
    print("Generating embeddings...")
    user_embeddings, item_embeddings = generate_embeddings(model, processed_data)
    
    # Save embeddings
    print("Saving embeddings...")
    np.save(os.path.join(args.output_dir, "user_embeddings.npy"), user_embeddings)
    np.save(os.path.join(args.output_dir, "item_embeddings.npy"), item_embeddings)
    
    print(f"Embeddings saved to {args.output_dir}")
    print(f"User embeddings shape: {user_embeddings.shape}")
    print(f"Item embeddings shape: {item_embeddings.shape}")
    
    # Save simple stats
    stats = {
        "user_embeddings_shape": user_embeddings.shape,
        "item_embeddings_shape": item_embeddings.shape,
        "user_embeddings_mean": float(np.mean(user_embeddings)),
        "user_embeddings_std": float(np.std(user_embeddings)),
        "item_embeddings_mean": float(np.mean(item_embeddings)),
        "item_embeddings_std": float(np.std(item_embeddings)),
        "num_samples": len(data),
    }
    
    with open(os.path.join(args.output_dir, "embedding_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()