# here put the import lib
import torch
import random
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


def dropout_feature(text: str, dropout_ratio: float) -> tuple:
    """
    Apply feature dropout to text and produce two versions.
    Args:
        text: original text
        dropout_ratio: dropout ratio
    Returns:
        chosen: the retained/original text
        rejected: the text after dropping words
    """
    words = text.split()
    if len(words) <= 1:
        return text, text
    
    # Randomly select words to drop
    num_to_drop = max(1, int(len(words) * dropout_ratio))
    indices_to_drop = random.sample(range(len(words)), num_to_drop)
    
    # Create two versions
    chosen_words = words.copy()
    rejected_words = words.copy()
    
    # Delete the selected words in the rejected version
    for idx in sorted(indices_to_drop, reverse=True):
        del rejected_words[idx]
    
    chosen = " ".join(chosen_words)
    rejected = " ".join(rejected_words)
    
    return chosen, rejected


class UserItemTrainProcessor:
    """
    Training data processor for user–item interaction pairs.
    Suitable for the ICLL contrastive loss.
    """
    
    def __init__(self, data_args, model_args, tokenizer: PreTrainedTokenizer):
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.user_text_column = "user_text"
        self.item_text_column = "item_text"
        self.label_column = "label"
    
    def __call__(self, example):
        max_seq_length = self.data_args.max_source_length
        user_text = example.get(self.user_text_column, "")
        item_text = example.get(self.item_text_column, "")
        label = example[self.label_column] if self.label_column in example else 1
            
        if user_text and item_text:
            user_ids = self.tokenizer.encode(text=user_text, add_special_tokens=False)
            item_ids = self.tokenizer.encode(text=item_text, add_special_tokens=False)
            if len(user_ids) > max_seq_length - 1:
                user_ids = user_ids[:max_seq_length - 1]
            if len(item_ids) > max_seq_length - 1:
                item_ids = item_ids[:max_seq_length - 1]
            user_input_ids = self.tokenizer.build_inputs_with_special_tokens(user_ids)
            item_input_ids = self.tokenizer.build_inputs_with_special_tokens(item_ids)
            user_mask = [1] * len(user_input_ids)
            item_mask = [1] * len(item_input_ids)
            return {
                "user_input_ids": user_input_ids,
                "item_input_ids": item_input_ids,
                "user_attention_mask": user_mask,
                "item_attention_mask": item_mask,
                "labels": label
            }
        else:
            return {}



class UserItemEvalProcessor:
    """
    Evaluation data processor for user–item interaction pairs.
    """
    
    def __init__(self, data_args, model_args, tokenizer: PreTrainedTokenizer):
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.user_text_column = "user_text"
        self.item_text_column = "item_text"
        self.label_column = "label"
    
    def __call__(self, example):
        """
        Process a single evaluation example.
        Args:
            example: a dict containing user text, item text, and label (single sample)
        Returns:
            dict containing user/item input_ids and attention_mask, etc.
        """
        max_seq_length = self.data_args.max_source_length
        user_text = example.get(self.user_text_column, "")
        item_text = example.get(self.item_text_column, "")
        label = example[self.label_column] if self.label_column in example else 1
        if user_text and item_text:
            user_ids = self.tokenizer.encode(text=user_text, add_special_tokens=False)
            item_ids = self.tokenizer.encode(text=item_text, add_special_tokens=False)
            if len(user_ids) > max_seq_length - 1:
                user_ids = user_ids[:max_seq_length - 1]
            if len(item_ids) > max_seq_length - 1:
                item_ids = item_ids[:max_seq_length - 1]
            user_input_ids = self.tokenizer.build_inputs_with_special_tokens(user_ids)
            item_input_ids = self.tokenizer.build_inputs_with_special_tokens(item_ids)
            user_mask = [True] * len(user_input_ids)
            item_mask = [True] * len(item_input_ids)
            return {
                "user_input_ids": user_input_ids,
                "item_input_ids": item_input_ids,
                "user_attention_mask": user_mask,
                "item_attention_mask": item_mask,
                "labels": label
            }
        else:
            return {}


class UserItemDataCollator:
    """
    Data collator for user–item interaction pairs.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        Args:
            features: a list of user–item pairs
        """
        batch = {}
        
        # Check if features are empty or contain invalid data
        if not features:
            raise ValueError("Features list is empty")
        
        # Check that the first feature contains required keys
        first_feature = features[0]
        # print(f"first_feature: {first_feature}")
        required_keys = ["user_input_ids", "item_input_ids", "user_attention_mask", "item_attention_mask", "labels"]
        missing_keys = [key for key in required_keys if key not in first_feature]
        if missing_keys:
            raise ValueError(f"Missing required keys in features: {missing_keys}. Available keys: {list(first_feature.keys())}")
        
        # Handle inputs for users and items separately
        user_input_ids = [f["user_input_ids"] for f in features]
        item_input_ids = [f["item_input_ids"] for f in features]
        user_attention_mask = [f["user_attention_mask"] for f in features]
        item_attention_mask = [f["item_attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Padding
        if self.padding:
            user_input_ids = self.tokenizer.pad(
                {"input_ids": user_input_ids},
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            item_input_ids = self.tokenizer.pad(
                {"input_ids": item_input_ids},
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Manually pad attention masks
            max_user_len = max(len(mask) for mask in user_attention_mask)
            max_item_len = max(len(mask) for mask in item_attention_mask)
            
            padded_user_masks = []
            padded_item_masks = []
            
            for mask in user_attention_mask:
                padded_mask = mask + [0] * (max_user_len - len(mask))
                padded_user_masks.append(padded_mask)
            for mask in item_attention_mask:
                padded_mask = mask + [0] * (max_item_len - len(mask))
                padded_item_masks.append(padded_mask)
            
            batch["user_input_ids"] = user_input_ids["input_ids"]
            batch["item_input_ids"] = item_input_ids["input_ids"]
            batch["user_attention_mask"] = torch.tensor(padded_user_masks)
            batch["item_attention_mask"] = torch.tensor(padded_item_masks)
        else:
            batch["user_input_ids"] = torch.tensor(user_input_ids)
            batch["item_input_ids"] = torch.tensor(item_input_ids)
            batch["user_attention_mask"] = torch.tensor(user_attention_mask)
            batch["item_attention_mask"] = torch.tensor(item_attention_mask)
        
        batch["labels"] = torch.tensor(labels)
        
        # Remap parameter names to match the model's forward method
        # The model expects user_input_ids and item_input_ids, not input_ids
        batch["user_input_ids"] = batch["user_input_ids"]
        batch["item_input_ids"] = batch["item_input_ids"]
        batch["user_attention_mask"] = batch["user_attention_mask"]
        batch["item_attention_mask"] = batch["item_attention_mask"]
        
        return batch 