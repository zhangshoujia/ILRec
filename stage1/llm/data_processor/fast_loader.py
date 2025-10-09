"""
Fast data loader: directly load preprocessed tokenized data
"""
import os
import torch
import pickle
from datasets import Dataset
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class FastDatasetLoader:
    """Fast dataset loader"""
    
    @staticmethod
    def load_preprocessed_dataset(file_path, format='auto'):
        """
        Load a preprocessed dataset
        
        Args:
            file_path: path to the original JSON file
            format: 'pt', 'pkl', or 'auto'
        """
        base_path = file_path.replace('.json', '')
        
        # Try formats in priority order
        formats_to_try = ['pt', 'pkl'] if format == 'auto' else [format]
        
        for fmt in formats_to_try:
            tokenized_path = f"{base_path}_tokenized.{fmt}"
            
            if os.path.exists(tokenized_path):
                logger.info(f"📦 Loading preprocessed data from {tokenized_path}")
                
                if fmt == 'pt':
                    data = torch.load(tokenized_path, map_location='cpu')
                elif fmt == 'pkl':
                    with open(tokenized_path, 'rb') as f:
                        data = pickle.load(f)
                
                # Convert to HuggingFace Dataset
                return Dataset.from_list(data)
        
        # If no preprocessed file is found, return None
        logger.warning(f"⚠️  No preprocessed data found for {file_path}")
        return None
    
    @staticmethod
    def get_dataset_stats(file_path):
        """Get dataset statistics"""
        stats_path = file_path.replace('.json', '_stats.json')
        if os.path.exists(stats_path):
            import json
            with open(stats_path, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def create_fast_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        """Create an optimized DataLoader"""
        
        def collate_fn(batch):
            """Custom collate function to handle variable-length sequences"""
            # Get max length
            max_user_len = max(len(item['user_input_ids']) for item in batch)
            max_item_len = max(len(item['item_input_ids']) for item in batch)
            
            # Batch padding
            user_input_ids = []
            user_attention_mask = []
            item_input_ids = []
            item_attention_mask = []
            labels = []
            
            for item in batch:
                # Pad user sequences
                user_ids = item['user_input_ids'] + [0] * (max_user_len - len(item['user_input_ids']))
                user_mask = item['user_attention_mask'] + [0] * (max_user_len - len(item['user_attention_mask']))
                
                # Pad item sequences
                item_ids = item['item_input_ids'] + [0] * (max_item_len - len(item['item_input_ids']))
                item_mask = item['item_attention_mask'] + [0] * (max_item_len - len(item['item_attention_mask']))
                
                user_input_ids.append(user_ids)
                user_attention_mask.append(user_mask)
                item_input_ids.append(item_ids)
                item_attention_mask.append(item_mask)
                labels.append(item['labels'])
            
            return {
                'user_input_ids': torch.tensor(user_input_ids, dtype=torch.long),
                'user_attention_mask': torch.tensor(user_attention_mask, dtype=torch.long),
                'item_input_ids': torch.tensor(item_input_ids, dtype=torch.long),
                'item_attention_mask': torch.tensor(item_attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,  # Avoid memory issues
            persistent_workers=False
        )