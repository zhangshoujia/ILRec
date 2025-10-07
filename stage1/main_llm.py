#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import logging
import json
import glob
import datetime
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import nltk
import evaluate
import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)

# Project-internal imports
from llm.arguments import ModelArguments, DataTrainingArguments
from llm.llama import LlamaRSEmb, UserItemPromptLlama
from llm.data_processor.llama import llama_train_mask, llama_eval_mask
from llm.data_processor.user_item_llama import UserItemTrainProcessor, UserItemEvalProcessor, UserItemDataCollator
from llm.data_processor.collator import PairwiseDataCollatorWithPadding, LongestSequenceMaskCollator
from llm.lora_cls import PeftModelForCLS
from llm.peft import LoraConfig
from llm.trainer_seq2seq import MedRecTrainer

logger = logging.getLogger(__name__)

# Constants
PREFIX_CHECKPOINT_DIR = "checkpoint"


class SavePeftModelCallback(TrainerCallback):
    """Callback that only saves lightweight parameters, with integrated early stopping."""
    
    def __init__(self, save_steps=100, patience=2):
        self.save_steps = save_steps
        self.last_save_step = 0
        
        # Early stopping
        self.patience = patience
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.best_checkpoint_path = None
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Check for early stopping after each evaluation."""
        if logs is None:
            return
            
        current_metric = logs.get("eval_loss", float('inf'))
        
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            
            # Save the best model
            self._save_best_model(args, state, model=model, **kwargs)
            # logger.warning(f"🏆 Better model found! eval_loss: {current_metric:.4f}")
            
        else:
            self.patience_counter += 1
            # logger.warning(f"📈 No improvement ({self.patience_counter}/{self.patience})")
            
            if self.patience_counter >= self.patience:
                # logger.warning(f"🛑 Early stopping triggered! Best eval_loss: {self.best_metric:.4f}")
                control.should_training_stop = True
    
    def _save_best_model(self, args, state, **kwargs):
        """Save lightweight parameters of the best model."""
        # Remove previous best
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            import shutil
            shutil.rmtree(self.best_checkpoint_path)
        
        # Save new best
        self.best_checkpoint_path = os.path.join(args.output_dir, "best_model")
        os.makedirs(self.best_checkpoint_path, exist_ok=True)
        
        self._save_lightweight_checkpoint_to_path(self.best_checkpoint_path, state, **kwargs)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Optionally save at step intervals."""
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            self._save_lightweight_checkpoint(args, state, **kwargs)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save final lightweight checkpoint at end of training."""
        self._save_lightweight_checkpoint(args, state, **kwargs)
    
    def _save_lightweight_checkpoint(self, args, state, **kwargs):
        """Save a lightweight checkpoint."""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._save_lightweight_checkpoint_to_path(checkpoint_dir, state, **kwargs)
    
    def _save_lightweight_checkpoint_to_path(self, checkpoint_path, state, **kwargs):
        """Save a lightweight checkpoint to a given path."""
        # Save trainer state
        state.save_to_json(os.path.join(checkpoint_path, "trainer_state.json"))
        
        # Save model config/adapter
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Save adapter config
                if hasattr(self.model, 'peft_config'):
                    self.model.peft_config.save_pretrained(checkpoint_path)
                
                # Save adapter weights
                if hasattr(self.model, 'save_pretrained'):
                    self.model.save_pretrained(checkpoint_path, is_main_process=True)
                    
            except Exception as e:
                logger.warning(f"Error saving model parameters: {e}")
        
        # Save tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(checkpoint_path)
            except Exception as e:
                logger.warning(f"Error saving tokenizer: {e}")
        
        logger.info(f"Lightweight checkpoint saved to: {checkpoint_path}")


class DetailedLoggingCallback(TrainerCallback):
    """Detailed logging during training."""
    
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        self.training_step_count = 0
        self.eval_step_count = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Log at the start of training."""
        if args.local_rank in [-1, 0]:
            logger.info("=" * 60)
            logger.info("🚀 Training started!")
            logger.info("=" * 60)
            logger.info("Training arguments:")
            logger.info(f"  - output_dir: {args.output_dir}")
            logger.info(f"  - per_device_train_batch_size: {args.per_device_train_batch_size}")
            logger.info(f"  - gradient_accumulation_steps: {args.gradient_accumulation_steps}")
            logger.info(f"  - learning_rate: {args.learning_rate}")
            logger.info(f"  - max_steps: {args.max_steps}")
            logger.info(f"  - num_train_epochs: {args.num_train_epochs}")
            logger.info(f"  - warmup_ratio: {args.warmup_ratio}")
            logger.info(f"  - weight_decay: {args.weight_decay}")
            logger.info(f"  - max_grad_norm: {args.max_grad_norm}")
            logger.info(f"  - logging_steps: {args.logging_steps}")
            logger.info(f"  - eval_steps: {args.eval_steps}")
            logger.info(f"  - save_steps: {args.save_steps}")
            logger.info("=" * 60)
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Log at the beginning of a step."""
        if args.local_rank in [-1, 0] and state.global_step % self.log_every_n_steps == 0:
            logger.info(f"📊 Train step {state.global_step} begins (epoch {state.epoch:.2f})")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log at the end of a step."""
        if args.local_rank in [-1, 0] and state.global_step % self.log_every_n_steps == 0:
            # Current LR
            if hasattr(state, 'learning_rate') and state.learning_rate is not None:
                lr = state.learning_rate
            else:
                lr = args.learning_rate
                
            logger.info(f"✅ Train step {state.global_step} finished (epoch {state.epoch:.2f})")
            logger.info(f"   current learning_rate: {lr:.2e}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log during evaluation."""
        if args.local_rank in [-1, 0]:
            logger.info("=" * 50)
            logger.info(f"🔍 Eval at step {state.global_step} (epoch {state.epoch:.2f})")
            if metrics:
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value:.6f}")
            logger.info("=" * 50)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle logs."""
        if args.local_rank in [-1, 0] and logs:
            # Training loss
            if 'loss' in logs:
                logger.info(f"📈 train_loss: {logs['loss']:.6f}")
            
            # Learning rate
            if 'learning_rate' in logs:
                logger.info(f"📚 learning_rate: {logs['learning_rate']:.2e}")
            
            # Other metrics
            for key, value in logs.items():
                if key not in ['loss', 'learning_rate'] and isinstance(value, (int, float)):
                    logger.info(f"📊 {key}: {value:.6f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log at the end of training."""
        if args.local_rank in [-1, 0]:
            logger.info("=" * 60)
            logger.info("🎉 Training finished!")
            logger.info(f"Total training steps: {state.global_step}")
            logger.info(f"Total training epochs: {state.epoch:.2f}")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info("=" * 60)


def parse_arguments():
    """Parse command line arguments."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    # Handle custom arguments not recognized by HfArgumentParser
    custom_args = {}
    
    # Handle early_stopping_patience
    if '--early_stopping_patience' in sys.argv:
        idx = sys.argv.index('--early_stopping_patience')
        custom_args['early_stopping_patience'] = int(sys.argv[idx+1])
        sys.argv = sys.argv[:idx] + sys.argv[idx+2:]
    
    # Handle save_steps
    if '--save_steps' in sys.argv:
        idx = sys.argv.index('--save_steps')
        custom_args['save_steps'] = int(sys.argv[idx+1])
        # Do not remove from sys.argv so HfArgumentParser can also see it
        # sys.argv = sys.argv[:idx] + sys.argv[idx+2:]
    
    # Handle save_only_model (remove since we use custom saving)
    if '--save_only_model' in sys.argv:
        idx = sys.argv.index('--save_only_model')
        # Remove the flag and its value
        sys.argv = sys.argv[:idx] + sys.argv[idx+2:]
        logger.warning("Removed --save_only_model; using custom lightweight saving logic.")
    
    # Parse
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Apply training args optimization
    _optimize_training_args(training_args)
    
    return model_args, data_args, training_args, custom_args


def _optimize_training_args(training_args):
    """Optimize training args: fully disable automatic saving and rely on custom callback."""
    # Disable wandb to avoid account selection prompts
    training_args.report_to = ["tensorboard"]
    
    # Keep custom fields
    training_args.remove_unused_columns = False
    
    # 🔥 Key: completely disable Trainer's saving behavior
    training_args.save_strategy = "no"
    training_args.save_steps = None
    training_args.save_total_limit = None
    
    # Disable load_best_model_at_end (no autosave)
    training_args.load_best_model_at_end = False
    
    # Avoid saving large optimizer files, etc.
    training_args.save_safetensors = False
    training_args.save_only_model = True
    
    # logger.warning("✅ Disabled auto-saving; using custom early-stop + lightweight saving.")


def setup_logging(training_args):
    """Configure logging."""
    # Create log directory
    log_dir = os.path.join(training_args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Log formats
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    
    # Root logger config
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),          # console
            logging.FileHandler(log_file, encoding='utf-8')  # file
        ]
    )
    
    # Transformers verbosity
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Only in main process to avoid multi-GPU duplication
    if training_args.local_rank in [-1, 0]:
        logger.info(f"Log file: {log_file}")
        logger.info(f"Output directory: {training_args.output_dir}")
        logger.info(f"Process rank: {training_args.local_rank}")
        logger.info(f"Device: {training_args.device}")
        logger.info(f"#GPUs: {training_args.n_gpu}")
        logger.info(f"Distributed training: {bool(training_args.local_rank != -1)}")
        logger.info(f"FP16: {training_args.fp16}")
        logger.info(f"BF16: {training_args.bf16}")
        logger.info("=" * 50)
        logger.info("Training args:")
        for arg, value in vars(training_args).items():
            if not arg.startswith('_'):
                logger.info(f"  {arg}: {value}")
        logger.info("=" * 50)


def load_datasets(data_args, training_args):
    """Load and preprocess datasets — supports pre-tokenized .pt files."""
    # Check for pre-tokenized files
    train_file = data_args.train_file
    validation_file = data_args.validation_file
    
    # Pre-tokenized counterparts
    if train_file and train_file.endswith('.json'):
        tokenized_train_file = train_file.replace('.json', '_tokenized.pt')
        if os.path.exists(tokenized_train_file):
            logger.warning(f"🚀 Found pre-tokenized train file: {tokenized_train_file}")
            train_file = tokenized_train_file
    
    if validation_file and validation_file.endswith('.json'):
        tokenized_validation_file = validation_file.replace('.json', '_tokenized.pt')
        if os.path.exists(tokenized_validation_file):
            logger.warning(f"🚀 Found pre-tokenized validation file: {tokenized_validation_file}")
            validation_file = tokenized_validation_file
    
    # If .pt, load directly
    if train_file and train_file.endswith('.pt'):
        logger.warning(f"📦 Loading tokenized train data directly: {train_file}")
        train_dataset = torch.load(train_file)
        logger.warning(f"✅ Train data loaded, #samples: {len(train_dataset)}")
    else:
        # Raw JSON loading path
        data_files = {"train": train_file}
        extension = train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=getattr(data_args, 'cache_dir', None))
        train_dataset = raw_datasets["train"]
    
    if validation_file and validation_file.endswith('.pt'):
        logger.warning(f"📦 Loading tokenized validation data directly: {validation_file}")
        validation_dataset = torch.load(validation_file)
        logger.warning(f"✅ Validation data loaded, #samples: {len(validation_dataset)}")
    else:
        # Raw JSON loading path
        data_files = {"validation": validation_file}
        extension = validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=getattr(data_args, 'cache_dir', None))
        validation_dataset = raw_datasets["validation"]
    
    # Build a minimal dataset-like wrapper
    class TokenizedDatasets:
        def __init__(self, train_data=None, validation_data=None):
            self.data = {}
            if train_data is not None:
                self.data["train"] = train_data
            if validation_data is not None:
                self.data["validation"] = validation_data
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __contains__(self, key):
            return key in self.data
    
    return TokenizedDatasets(train_dataset if 'train_dataset' in locals() else None,
                           validation_dataset if 'validation_dataset' in locals() else None)


def load_model_and_tokenizer(model_args, data_args, training_args):
    """Load model and tokenizer."""
    # Config
    config = AutoConfig.from_pretrained(
        getattr(model_args, 'config_name', None) or model_args.model_name_or_path,
        cache_dir=getattr(model_args, 'cache_dir', None),
        revision=getattr(model_args, 'model_revision', None),
        token=getattr(model_args, 'use_auth_token', None),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(model_args, 'tokenizer_name', None) or model_args.model_name_or_path,
        cache_dir=getattr(model_args, 'cache_dir', None),
        use_fast=getattr(model_args, 'use_fast_tokenizer', True),
        revision=getattr(model_args, 'model_revision', None),
        token=getattr(model_args, 'use_auth_token', None),
    )

    # Tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # Model choice
    if model_args.model_choice == "useritem_prompt":
        model = UserItemPromptLlama.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=getattr(model_args, 'cache_dir', None),
            revision=getattr(model_args, 'model_revision', None),
            token=getattr(model_args, 'use_auth_token', None),
            pool_type=model_args.pool_type,
            tau=model_args.tau,
            user_prompt_length=model_args.user_prompt_length,
            item_prompt_length=model_args.item_prompt_length,
            use_global_batch_loss=getattr(model_args, 'use_global_batch_loss', False),
        )
        
        # 🔍 Inspect critical prompt params before PEFT wrapping
        # logger.warning("=== Parameter check before PEFT wrapping ===")
        # for name, param in model.named_parameters():
        #     if any(key in name for key in ["user_prompts", "item_prompts", "adapter", "final_projection", "tau"]):
        #         logger.warning(f"🎯 Found before PEFT: {name} - {param.shape}")
        
        # Set trainable params (before PEFT wrapping)
        _setup_useritem_prompt_training(model)

    # LoRA (if any)
    if hasattr(model_args, 'lora_rank') and model_args.lora_rank > 0:
        # logger.warning("=== Begin PEFT wrapping ===")
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.trainable.split(","),
            lora_dropout=model_args.lora_dropout,
            task_type="SEQ_CLS",
        )
        model = PeftModelForCLS(model, peft_config)

        _setup_useritem_prompt_training(model)
        
        # Attach LoRA semantic anchor & prompt-diversity configs
        model.enable_lora_anchor = getattr(model_args, "enable_lora_anchor", False)
        model.lora_anchor_weight = getattr(model_args, "lora_anchor_weight", 0.0)
        model.enable_prompt_diversity = getattr(model_args, "enable_prompt_diversity", False)
        model.prompt_div_weight = getattr(model_args, "prompt_div_weight", 0.0)
        
        # Attach contrastive anchor configs
        model.enable_contrastive_anchor = getattr(model_args, "enable_contrastive_anchor", False)
        model.contrastive_anchor_weight = getattr(model_args, "contrastive_anchor_weight", 0.0)
        model.contrastive_temperature = getattr(model_args, "contrastive_temperature", 1.0)
        
        # logger.warning("=== Parameter check after PEFT wrapping ===")
        # for name, param in model.named_parameters():
        #     if any(key in name for key in ["user_prompts", "item_prompts", "adapter", "final_projection", "tau"]):
        #         logger.warning(f"🎯 Found after PEFT: {name} - requires_grad={param.requires_grad} - {param.shape}")

    return model, tokenizer, config


def _setup_useritem_prompt_training(model):
    """Set trainable params for the UserItemPrompt model."""
    # logger.warning("=== Setting trainable params for UserItemPrompt ===")
    
    # Check if wrapped by PEFT
    if hasattr(model, 'base_model'):
        # logger.warning("⚠️ Model already wrapped by PEFT; checking base_model parameters.")
        base_model = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
    else:
        base_model = model
    
    # Print all parameter names for debugging (optional)
    # logger.warning("=== All parameter names after PEFT wrapping ===")
    # for name, param in model.named_parameters():
    #     if any(key in name for key in ["prompt", "adapter", "final_projection", "tau"]):
    #         logger.warning(f"🔍 Key param: {name} - {param.shape}")
    
    trainable_count = 0
    for name, param in model.named_parameters():
        # Normalize name for checks
        clean_name = name.replace("base_model.model.", "").replace("base_model.", "")
        if any(key in clean_name for key in [
            "user_prompts", "user_prompt_weights", 
            "item_prompts", "item_prompt_weights", 
            "adapter", "final_projection", "tau"
        ]):
            param.requires_grad = True
            # logger.warning(f"✅ Set trainable: {name} -> {clean_name} - {param.shape}")
            trainable_count += 1
        elif "lora" in name:
            # LoRA params are trainable already
            # logger.warning(f"🔧 LoRA param: {name} - {param.shape}")
            trainable_count += 1
    
    logger.warning(f"📊 Total trainable parameter groups set: {trainable_count}")


def setup_data_processing(model_args, data_args, tokenizer, training_args):
    """Set up data processors and collators."""
    if model_args.model_choice == "useritem_prompt":
        if training_args.do_train:
            preprocess_func = UserItemTrainProcessor(data_args, model_args, tokenizer)
            data_collator = UserItemDataCollator(tokenizer, padding=True, max_length=data_args.max_source_length)
        else:
            preprocess_func = UserItemEvalProcessor(data_args, model_args, tokenizer)
            data_collator = UserItemDataCollator(tokenizer, padding=True, max_length=data_args.max_source_length)
    else:
        if training_args.do_train:
            preprocess_func = llama_train_mask(data_args, model_args, tokenizer)
            data_collator = PairwiseDataCollatorWithPadding(tokenizer)
        else:
            preprocess_func = llama_eval_mask(data_args, model_args, tokenizer)
            data_collator = LongestSequenceMaskCollator(tokenizer)
    
    return preprocess_func, data_collator


def process_datasets(raw_datasets, preprocess_func, data_args, training_args):
    """Process datasets — supports pre-tokenized data."""
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        
        # If already tokenized, skip preprocessing
        if hasattr(train_dataset, '__getitem__') and len(train_dataset) > 0:
            sample = train_dataset[0]
            if isinstance(sample, dict) and 'user_input_ids' in sample and 'item_input_ids' in sample:
                logger.warning("🎯 Detected pre-tokenized train data; skipping preprocessing.")
            else:
                logger.warning("📝 Tokenizing train data on-the-fly.")
                if getattr(data_args, 'max_train_samples', None) is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        preprocess_func,
                        batched=False,
                        num_proc=1,
                        load_from_cache_file=False,
                        desc="Running tokenizer on train dataset",
                    )
        else:
            # For list-like data, limit sample size if requested
            if getattr(data_args, 'max_train_samples', None) is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset[:max_train_samples]

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        
        # If already tokenized, skip preprocessing
        if hasattr(eval_dataset, '__getitem__') and len(eval_dataset) > 0:
            sample = eval_dataset[0]
            if isinstance(sample, dict) and 'user_input_ids' in sample and 'item_input_ids' in sample:
                logger.warning("🎯 Detected pre-tokenized validation data; skipping preprocessing.")
            else:
                logger.warning("📝 Tokenizing validation data on-the-fly.")
                if getattr(data_args, 'max_eval_samples', None) is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        preprocess_func,
                        batched=False,
                        num_proc=getattr(data_args, 'preprocessing_num_workers', None),
                        load_from_cache_file=not getattr(data_args, 'overwrite_cache', False),
                        desc="Running tokenizer on validation dataset",
                    )
        else:
            if getattr(data_args, 'max_eval_samples', None) is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset[:max_eval_samples]
        
        # Show first sample (if exists)
        if eval_dataset and len(eval_dataset) > 0:
            logger.info(f"eval_dataset[0]: {eval_dataset[0]}")

    return train_dataset, eval_dataset


def setup_callbacks(training_args, early_stopping_patience, save_steps):
    """Set up training callbacks."""
    callbacks = []
    
    # Add SavePeftModelCallback with early stopping
    if training_args.do_train:
        patience = early_stopping_patience if early_stopping_patience else 2
        savesteps = save_steps if save_steps else 100
        callbacks.append(SavePeftModelCallback(save_steps=savesteps, patience=patience))
        logger.warning(f"✅ Custom early stopping enabled, patience: {patience}")
    
    return callbacks


def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    logger.warning("Metrics calculation is disabled to avoid evaluate.load issue.")
    return {}


def cleanup_unnecessary_files(output_dir):
    """Remove unnecessary large files."""
    # Delete full model files
    for pattern in ["pytorch_model*.bin", "model*.safetensors", "*.pt"]:
        for file_path in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.exists(file_path):
                os.remove(file_path)
                # logger.info(f"🗑️  Removed unnecessary file: {file_path}")
    
    # Also clean within checkpoint dirs
    for checkpoint_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if os.path.isdir(checkpoint_dir):
            for pattern in ["pytorch_model*.bin", "model*.safetensors", "*.pt"]:
                for file_path in glob.glob(os.path.join(checkpoint_dir, pattern)):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        # logger.info(f"🗑️  Removed large file from checkpoint: {file_path}")


def verify_saved_parameters(output_dir):
    """Verify that required parameter files exist and look reasonable."""
    logger.info("=== Verifying saved parameter files ===")
    
    required_files = [
        "adapter_config.json",          # LoRA config
        "adapter_model.bin",            # LoRA parameters
        "prompt_params.bin",            # soft prompt parameters
        "model_adapter_params.bin",     # model adapter parameters
        "training_config.json",         # training config
    ]
    
    file_sizes = {}
    for file_name in required_files:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_sizes[file_name] = size_mb
            logger.info(f"✅ {file_name}: {size_mb:.2f} MB")
        else:
            logger.warning(f"❌ Missing file: {file_name}")
    
    total_size = sum(file_sizes.values())
    logger.info(f"📊 Total parameter file size: {total_size:.2f} MB")
    
    # Check for other large files that may be unnecessary
    large_files = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith(('.bin', '.safetensors', '.pt')) and file_name not in required_files:
            file_path = os.path.join(output_dir, file_name)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > 100:  # larger than 100MB
                large_files.append((file_name, size_mb))
    
    if large_files:
        logger.warning("⚠️  Found potentially unnecessary large files:")
        for file_name, size_mb in large_files:
            logger.warning(f"   {file_name}: {size_mb:.2f} MB")
    
    logger.info("=== Parameter file verification done ===")
    return total_size < 500  # consider success if total < 500MB


def train():
    """Main training function."""
    # 1) Parse args
    model_args, data_args, training_args, custom_args = parse_arguments()
    
    # 2) Setup logging
    setup_logging(training_args)
    
    # 3) Seed
    set_seed(training_args.seed)
    
    # 4) Check output dir
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 5) Load datasets
    if training_args.local_rank in [-1, 0]:
        logger.info("Loading datasets...")
    raw_datasets = load_datasets(data_args, training_args)
    
    # 6) Load model & tokenizer
    if training_args.local_rank in [-1, 0]:
        logger.info("Loading model and tokenizer...")
    model, tokenizer, config = load_model_and_tokenizer(model_args, data_args, training_args)
    
    # 7) Setup data processing
    if training_args.local_rank in [-1, 0]:
        logger.info("Setting up data processing...")
    preprocess_func, data_collator = setup_data_processing(model_args, data_args, tokenizer, training_args)
    
    # 8) Process datasets
    if training_args.local_rank in [-1, 0]:
        logger.info("Processing datasets...")
    train_dataset, eval_dataset = process_datasets(raw_datasets, preprocess_func, data_args, training_args)
    
    # Dataset stats
    if training_args.local_rank in [-1, 0]:
        logger.info(f"Train dataset size: {len(train_dataset) if train_dataset else 0}")
        logger.info(f"Validation dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    # 9) Callbacks
    early_stopping_patience = custom_args.get('early_stopping_patience')
    save_steps = custom_args.get('save_steps')
    callbacks = setup_callbacks(training_args, early_stopping_patience, save_steps)
    
    # 10) Trainer
    if training_args.local_rank in [-1, 0]:
        logger.info("Initializing Trainer...")
    trainer = MedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=callbacks,
    )

    # 11) Train
    if training_args.do_train:
        if training_args.local_rank in [-1, 0]:
            logger.info("=" * 50)
            logger.info("Starting training...")
            logger.info("=" * 50)
            
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            if training_args.local_rank in [-1, 0]:
                logger.info(f"Resuming from checkpoint: {checkpoint}")
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            if training_args.local_rank in [-1, 0]:
                logger.info(f"Resuming from last checkpoint: {checkpoint}")
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # After training, save only lightweight parameters
        if training_args.local_rank in [-1, 0]:
            logger.info("=" * 50)
            logger.info("Training finished, saving final lightweight parameters")
            logger.info("=" * 50)
        
        # Custom save
        trainer.save_model()
        
        # Cleanup large full-model files if any
        cleanup_unnecessary_files(training_args.output_dir)
        
        # Verify saved params
        if verify_saved_parameters(training_args.output_dir):
            if training_args.local_rank in [-1, 0]:
                logger.info("🎉 Lightweight parameter save succeeded!")
        else:
            if training_args.local_rank in [-1, 0]:
                logger.warning("⚠️  Parameter saving might have issues; please check file sizes.")
        
        # Log training metrics
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Detailed training metrics
        if training_args.local_rank in [-1, 0]:
            logger.info("=" * 50)
            logger.info("Training metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 50)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 12) Eval
    if training_args.do_eval:
        if training_args.local_rank in [-1, 0]:
            logger.info("=" * 50)
            logger.info("Starting evaluation...")
            logger.info("=" * 50)
            
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # Detailed eval metrics
        if training_args.local_rank in [-1, 0]:
            logger.info("=" * 50)
            logger.info("Evaluation metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 50)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training and evaluation completed.")
    return {}


def _mp_fn(index):
    """For xla_spawn (TPUs)"""
    train()


if __name__ == "__main__":
    train()