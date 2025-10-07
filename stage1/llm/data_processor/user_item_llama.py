# here put the import lib
import torch
import random
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


def dropout_feature(text: str, dropout_ratio: float) -> tuple:
    """
    对文本进行特征dropout，生成两个版本
    Args:
        text: 原始文本
        dropout_ratio: dropout比例
    Returns:
        chosen: 保留的文本
        rejected: dropout后的文本
    """
    words = text.split()
    if len(words) <= 1:
        return text, text
    
    # 随机选择要dropout的单词
    num_to_drop = max(1, int(len(words) * dropout_ratio))
    indices_to_drop = random.sample(range(len(words)), num_to_drop)
    
    # 创建两个版本
    chosen_words = words.copy()
    rejected_words = words.copy()
    
    # 在rejected版本中删除选中的单词
    for idx in sorted(indices_to_drop, reverse=True):
        del rejected_words[idx]
    
    chosen = " ".join(chosen_words)
    rejected = " ".join(rejected_words)
    
    return chosen, rejected


class UserItemTrainProcessor:
    """
    用户-物品交互对的训练数据处理器
    适用于ICLL对比损失函数
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
    用户-物品交互对的评估数据处理器
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
        处理评估数据
        Args:
            example: 包含用户文本、物品文本和标签的字典（单条）
        Returns:
            dict，包含user/item的input_ids和attention_mask等
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
    用户-物品交互对的数据整理器
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理batch数据
        Args:
            features: 包含用户-物品对的列表
        """
        batch = {}
        
        # 检查features是否为空或包含无效数据
        if not features:
            raise ValueError("Features list is empty")
        
        # 检查第一个feature是否包含必要的键
        first_feature = features[0]
        # print(f"first_feature: {first_feature}")
        required_keys = ["user_input_ids", "item_input_ids", "user_attention_mask", "item_attention_mask", "labels"]
        missing_keys = [key for key in required_keys if key not in first_feature]
        if missing_keys:
            raise ValueError(f"Missing required keys in features: {missing_keys}. Available keys: {list(first_feature.keys())}")
        
        # 分别处理用户和物品的输入
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
            
            # 手动处理attention mask的padding
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
        
        # 重新映射参数名以匹配模型的forward方法
        # 模型期望 user_input_ids 和 item_input_ids，而不是 input_ids
        batch["user_input_ids"] = batch["user_input_ids"]
        batch["item_input_ids"] = batch["item_input_ids"]
        batch["user_attention_mask"] = batch["user_attention_mask"]
        batch["item_attention_mask"] = batch["item_attention_mask"]
        
        return batch 