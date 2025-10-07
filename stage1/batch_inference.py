import os
import json
import torch
import torch.distributed as dist
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from llm.llama import UserItemPromptLlama
from llm.lora_cls import PeftModelForCLS
import time
import pickle

def init_distributed():
    """Initialize distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def load_json_file(file_path):
    """Load a JSON file – supports JSONL (one JSON object per line)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: skip invalid JSON line: {e}")
                    continue
    return data

def save_embeddings(embeddings, output_file, rank=0):
    """Save embeddings to file."""
    if rank == 0:  # only main process saves
        try:
            # ensure output dir exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # save as pickle
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"✅ Saved embeddings to {output_file}")
            print(f"📊 Shape: {embeddings.shape}")
            print(f"📁 File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
            
            # also save as .npy (compat with bash scripts)
            npy_file = output_file.replace('.pkl', '.npy')
            np.save(npy_file, embeddings)
            print(f"✅ Also saved NumPy format: {npy_file}")
            
            # verify files exist
            if os.path.exists(output_file) and os.path.exists(npy_file):
                print(f"✅ File verification passed: {output_file} and {npy_file}")
            else:
                print(f"❌ File saving failed")
                
        except Exception as e:
            print(f"❌ Error saving files: {e}")
            import traceback
            traceback.print_exc()

def load_model_with_all_params(model_path, output_dir, user_prompt_length, item_prompt_length, 
                              pool_type, tau, torch_dtype=torch.float16, device_map=None, rank=0):
    """
    Load model and all trained parameters (LoRA + soft prompts + adapter).
    """
    if rank == 0:
        print("=== Start loading model and all trained parameters ===")
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "user_prompt_length": user_prompt_length,
        "item_prompt_length": item_prompt_length,
        "pool_type": pool_type,
        "tau": tau,
    }
    
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    
    # 1) base model
    print(f"📦 Loading base model: {model_path}")
    model = UserItemPromptLlama.from_pretrained(model_path, **model_kwargs)
    
    # 2) load soft prompts and adapter params (before PEFT wrapping)
    current_state_dict = model.state_dict()
    loaded_params = set()
    
    # soft prompt params
    prompt_params_path = os.path.join(output_dir, "prompt_params.bin")
    if os.path.exists(prompt_params_path):
        prompt_state_dict = torch.load(prompt_params_path, map_location="cpu")
        
        # clean keys, remove PEFT prefix
        cleaned_prompt_dict = {}
        for key, value in prompt_state_dict.items():
            # strip "base_model.model." prefix
            clean_key = key.replace("base_model.model.", "")
            cleaned_prompt_dict[clean_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_prompt_dict, strict=False)
        loaded_params.update(cleaned_prompt_dict.keys())
        print(f"✅ Loaded soft prompt params ({len(cleaned_prompt_dict)} keys)")
        
        # only report important missing params
        important_missing = [k for k in missing_keys if any(x in k for x in ["user_prompts", "item_prompts", "user_prompt_weights", "item_prompt_weights"])]
        if important_missing:
            print(f"   ⚠️  Important missing keys: {important_missing}")
        else:
            print(f"   ✅ All soft prompt params loaded successfully")
        
        # show loaded keys
        print(f"   Loaded keys:")
        for key in cleaned_prompt_dict.keys():
            print(f"     ✅ {key}: {cleaned_prompt_dict[key].shape}")
    else:
        print("⚠️  Soft prompt parameter file not found")
    
    # adapter params
    adapter_params_path = os.path.join(output_dir, "model_adapter_params.bin")
    if os.path.exists(adapter_params_path):
        adapter_state_dict = torch.load(adapter_params_path, map_location="cpu")
        
        # clean keys, remove PEFT prefix
        cleaned_adapter_dict = {}
        for key, value in adapter_state_dict.items():
            clean_key = key.replace("base_model.model.", "")
            cleaned_adapter_dict[clean_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_adapter_dict, strict=False)
        loaded_params.update(cleaned_adapter_dict.keys())
        print(f"✅ Loaded adapter params ({len(cleaned_adapter_dict)} keys)")
        
        # only report important missing params
        important_missing = [k for k in missing_keys if any(x in k for x in ["adapter", "final_projection", "tau"])]
        if important_missing:
            print(f"   ⚠️  Important missing keys: {important_missing}")
        else:
            print(f"   ✅ All adapter params loaded successfully")
        
        # show loaded keys
        print(f"   Loaded keys:")
        for key in cleaned_adapter_dict.keys():
            print(f"     ✅ {key}: {cleaned_adapter_dict[key].shape}")
    else:
        print("⚠️  Adapter parameter file not found")
    
    # 3) finally load LoRA params (this wraps the model)
    lora_config_path = os.path.join(output_dir, "adapter_config.json")
    lora_model_path = os.path.join(output_dir, "adapter_model.bin")
    if os.path.exists(lora_config_path) and os.path.exists(lora_model_path):
        model = PeftModelForCLS.from_pretrained(model, output_dir)
        print(f"✅ Loaded LoRA parameters from: {output_dir}")
    else:
        print("⚠️  LoRA parameter files not found")
    
    print("=== Model loading complete ===")
    return model


def load_preprocessed_data(input_file, mode):
    """
    Try loading pre-tokenized data.
    
    Args:
        input_file: original input path
        mode: 'user' or 'item'
    
    Returns:
        (data, is_preprocessed): the data and a boolean flag
    """
    base_path = input_file.replace('.json', '')
    
    # try formats by priority
    formats_to_try = ['pt', 'pkl']
    
    for fmt in formats_to_try:
        tokenized_path = f"{base_path}_tokenized.{fmt}"
        
        if os.path.exists(tokenized_path):
            print(f"📦 Found preprocessed data: {tokenized_path}")
            
            try:
                start_time = time.time()
                
                if fmt == 'pt':
                    data = torch.load(tokenized_path, map_location='cpu')
                elif fmt == 'pkl':
                    with open(tokenized_path, 'rb') as f:
                        data = pickle.load(f)
                
                load_time = time.time() - start_time
                print(f"✅ Loaded preprocessed data ({load_time:.2f}s)")
                print(f"📊 Loaded {len(data)} preprocessed samples")
                
                # validate format
                if validate_preprocessed_data(data, mode):
                    return data, True
                else:
                    print(f"⚠️  Preprocessed data format check failed; falling back to on-the-fly tokenization")
                    break
                    
            except Exception as e:
                print(f"⚠️  Failed to load preprocessed data: {e}")
                continue
    
    # not found / failed
    print(f"📄 No valid preprocessed data, using on-the-fly tokenization")
    return None, False


def validate_preprocessed_data(data, mode):
    """
    Validate the preprocessed data format.
    
    Args:
        data: preprocessed data list
        mode: 'user' or 'item'
    
    Returns:
        bool: whether format is valid
    """
    if not data or not isinstance(data, list):
        print("❌ Invalid data: not a list")
        return False
    
    # check first sample
    sample = data[0]
    
    # new single-entity format
    if 'input_ids' in sample and 'attention_mask' in sample and 'entity_type' in sample:
        print(f"✅ Detected single-entity preprocessed format")
        print(f"   Entity type: {sample['entity_type']}")
        print(f"   Sequence length: {len(sample['input_ids'])}")
        return True
    
    # old user-item pair format
    required_fields = ['user_input_ids', 'user_attention_mask', 'item_input_ids', 'item_attention_mask']
    
    for field in required_fields:
        if field not in sample:
            print(f"❌ Missing required field: {field}")
            return False
        
        if not isinstance(sample[field], (list, torch.Tensor)):
            print(f"❌ Invalid field type: {field}")
            return False
    
    print(f"✅ Preprocessed format validated (user-item pair format)")
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   User seq length: {len(sample['user_input_ids'])}")
    print(f"   Item seq length: {len(sample['item_input_ids'])}")
    
    return True


def extract_embeddings_from_preprocessed(data, mode):
    """
    Extract items needed for embedding inference from preprocessed data.
    Args:
        data: list of preprocessed records
        mode: 'user' or 'item'
    Returns:
        list of dict tensors for model input
    """
    extracted_data = []
    
    for item in data:
        # single-entity format
        if 'entity_type' in item and 'input_ids' in item:
            # construct a pair with a placeholder counterpart
            if mode == 'user':
                # real user + placeholder item
                user_input_ids = torch.tensor(item['input_ids'])
                user_attention_mask = torch.tensor(item['attention_mask'])
                item_input_ids = torch.zeros_like(user_input_ids)
                item_attention_mask = torch.zeros_like(user_attention_mask)
            else:  # mode == 'item'
                # placeholder user + real item
                item_input_ids = torch.tensor(item['input_ids'])
                item_attention_mask = torch.tensor(item['attention_mask'])
                user_input_ids = torch.zeros_like(item_input_ids)
                user_attention_mask = torch.zeros_like(item_attention_mask)
                
            extracted_data.append({
                'user_input_ids': user_input_ids,
                'user_attention_mask': user_attention_mask,
                'item_input_ids': item_input_ids,
                'item_attention_mask': item_attention_mask,
            })
        else:
            # old pair format: copy through
            user_input_ids = torch.tensor(item.get('user_input_ids', []))
            user_attention_mask = torch.tensor(item.get('user_attention_mask', []))
            item_input_ids = torch.tensor(item.get('item_input_ids', []))
            item_attention_mask = torch.tensor(item.get('item_attention_mask', []))
            
            extracted_data.append({
                'user_input_ids': user_input_ids,
                'user_attention_mask': user_attention_mask,
                'item_input_ids': item_input_ids,
                'item_attention_mask': item_attention_mask,
            })
    
    return extracted_data


def load_data(file_path, mode):
    """
    Load data file (prefer preprocessed data).
    Args:
        file_path: path to data file
        mode: 'user' or 'item'
    Returns:
        (data, is_preprocessed): list and a boolean flag
    """
    print(f"🔍 Checking for preprocessed data...")
    
    # try preprocessed first
    preprocessed_data, is_preprocessed = load_preprocessed_data(file_path, mode)
    
    if is_preprocessed:
        return preprocessed_data, True
    
    # fallback to original JSON lines
    print(f"📄 Loading raw JSON data and tokenizing on the fly...")
    start_time = time.time()
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    # single entity text selection
                    if mode == 'user':
                        text = data.get('user_text', data.get('text', ''))
                    else:  # mode == 'item'
                        text = data.get('item_text', data.get('text', ''))
                    
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    print(f"Skip invalid JSON line: {line[:100]}...")
                    continue
    
    load_time = time.time() - start_time
    print(f"📊 Loaded raw data ({load_time:.2f}s)")
    print(f"Loaded {len(texts)} {mode} texts")
    
    return texts, False


def process_texts_for_inference(texts, mode, tokenizer, max_length, user_prompt_length, item_prompt_length):
    """
    Prepare text data for inference.
    Args:
        texts: list of strings
        mode: 'user' or 'item'
        tokenizer: tokenizer
        max_length: max sequence length
        user_prompt_length: reserved user prompt length
        item_prompt_length: reserved item prompt length
    Returns:
        list of dicts with 'user_text' and 'item_text'
    """
    processed_data = []
    
    for text in texts:
        if mode == 'user':
            # user mode: real user text + placeholder item
            item = {
                'user_text': text,
                'item_text': '[PLACEHOLDER]',
                'label': 1
            }
        else:  # mode == 'item'
            # item mode: placeholder user + real item text
            item = {
                'user_text': '[PLACEHOLDER]',
                'item_text': text,
                'label': 1
            }
        
        processed_data.append(item)
    
    return processed_data


def process_single_item(item, tokenizer, max_length, user_prompt_length, item_prompt_length):
    """
    Tokenize a single record.
    Args:
        item: dict with user_text and item_text
        tokenizer: tokenizer
        max_length: max sequence length
        user_prompt_length: reserved tokens for user soft prompt
        item_prompt_length: reserved tokens for item soft prompt
    Returns:
        dict of tensors for model input
    """
    user_text = item.get('user_text', '')
    item_text = item.get('item_text', '')
    
    # leave space for soft prompts
    effective_max_length = max_length - user_prompt_length - item_prompt_length
    
    # user
    user_encoding = tokenizer(
        user_text,
        max_length=effective_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # item
    item_encoding = tokenizer(
        item_text,
        max_length=effective_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'user_input_ids': user_encoding['input_ids'].squeeze(0),
        'user_attention_mask': user_encoding['attention_mask'].squeeze(0),
        'item_input_ids': item_encoding['input_ids'].squeeze(0),
        'item_attention_mask': item_encoding['attention_mask'].squeeze(0),
    }

def batch_inference(args):
    """Batch inference."""
    # device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # model + trained params
    print("Loading model and trained parameters...")
    model = load_model_with_all_params(
        model_path=args.model_path,
        output_dir=args.output_dir,
        user_prompt_length=args.user_prompt_length,
        item_prompt_length=args.item_prompt_length,
        pool_type=args.pool_type,
        tau=args.tau,
        torch_dtype=torch.float16,
        device_map=None  # no auto sharding
    )
    
    # ensure on target device
    model = model.to(device)
    model.eval()
    
    # verify parameter loading
    print("\n=== Parameter verification ===")
    prompt_params = []
    adapter_params = []
    lora_params = []
    
    for name, param in model.named_parameters():
        if any(key in name for key in ["user_prompts", "item_prompts", "user_prompt_weights", "item_prompt_weights"]):
            prompt_params.append((name, param.shape, param.device))
        elif any(key in name for key in ["adapter", "final_projection", "tau"]):
            adapter_params.append((name, param.shape, param.device))
        elif "lora" in name.lower():
            lora_params.append((name, param.shape, param.device))
    
    print(f"📊 Soft prompt params ({len(prompt_params)}):")
    for name, shape, device in prompt_params:
        print(f"   ✅ {name}: {shape} on {device}")
    
    print(f"📊 Adapter params ({len(adapter_params)}):")
    for name, shape, device in adapter_params:
        print(f"   ✅ {name}: {shape} on {device}")
    
    print(f"📊 LoRA params ({len(lora_params)}):")
    for name, shape, device in lora_params[:3]:  # show first 3
        print(f"   ✅ {name}: {shape} on {device}")
    if len(lora_params) > 3:
        print(f"   ... {len(lora_params) - 3} more LoRA parameters")
    
    # check critical keys
    required_params = ["user_prompts", "item_prompts", "adapter", "final_projection"]
    missing_params = []
    for req_param in required_params:
        found = any(req_param in name for name, _, _ in prompt_params + adapter_params)
        if not found:
            missing_params.append(req_param)
    
    if missing_params:
        print(f"❌ Missing critical params: {missing_params}")
    else:
        print("✅ All critical params are loaded")
    
    # quick forward test
    print("\n=== Quick forward test ===")
    test_batch_size = 2
    test_seq_len = 20
    
    test_user_ids = torch.randint(0, 1000, (test_batch_size, test_seq_len)).to(device)
    test_item_ids = torch.randint(0, 1000, (test_batch_size, test_seq_len)).to(device)
    test_user_mask = torch.ones_like(test_user_ids).to(device)
    test_item_mask = torch.ones_like(test_item_ids).to(device)
    
    try:
        with torch.no_grad():
            test_output = model(
                user_input_ids=test_user_ids,
                user_attention_mask=test_user_mask,
                item_input_ids=test_item_ids,
                item_attention_mask=test_item_mask,
            )
            
            if hasattr(test_output, 'logits'):
                test_emb = test_output.logits
            else:
                test_emb = test_output
            
            print(f"✅ Forward pass succeeded")
            print(f"   Output shape: {test_emb.shape}")
            print(f"   Output device: {test_emb.device}")
            print(f"   Output range: [{test_emb.min().item():.4f}, {test_emb.max().item():.4f}]")
            print(f"   Output mean: {test_emb.mean().item():.4f}")
            
            # sanity: not all zeros or constant
            if test_emb.std().item() < 1e-6:
                print("⚠️  Very small output variance; parameters may not be loaded correctly")
            else:
                print("✅ Output looks reasonable")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== Parameter verification done ===\n")
    
    # verify devices
    print("Verifying parameter device placement...")
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)
    print(f"Model parameters are on devices: {devices}")
    
    if len(devices) > 1:
        print("⚠️  Detected parameters across multiple devices, forcing to target device...")
        model = model.to(device)
        
        # re-check
        devices = set()
        for name, param in model.named_parameters():
            devices.add(param.device)
        print(f"After move, parameter devices: {devices}")

    
    # load data (optimized path)
    print(f"Loading {args.mode} data...")
    data, is_preprocessed = load_data(args.input_file, args.mode)
    
    if not data:
        print("❌ No valid data found")
        return
    
    # choose processing path
    if is_preprocessed:
        print("🚀 Using preprocessed data; skipping tokenization...")
        
        # extract model inputs
        processed_data = extract_embeddings_from_preprocessed(data, args.mode)
        
        # stats
        print(f"📊 Preprocessed data stats:")
        print(f"   #Samples: {len(processed_data)}")
        if processed_data:
            sample = processed_data[0]
            print(f"   User seq length: {len(sample['user_input_ids'])}")
            print(f"   Item seq length: {len(sample['item_input_ids'])}")
        
    else:
        print("⏳ Using raw data, performing on-the-fly tokenization...")
        start_time = time.time()
        
        # prepare pairs
        processed_data_raw = process_texts_for_inference(
            data, args.mode, tokenizer, args.max_length, 
            args.user_prompt_length, args.item_prompt_length
        )
        
        # tokenize each sample
        processed_data = []
        for item in processed_data_raw:
            processed_item = process_single_item(
                item, tokenizer, args.max_length, 
                args.user_prompt_length, args.item_prompt_length
            )
            processed_data.append(processed_item)
        
        tokenization_time = time.time() - start_time
        print(f"⏱️  On-the-fly tokenization done ({tokenization_time:.2f}s)")
    
    # batching
    print("Start inference...")
    all_embeddings = []
    
    total_batches = (len(processed_data) + args.batch_size - 1) // args.batch_size
    inference_start_time = time.time()
    
    for i in tqdm(range(0, len(processed_data), args.batch_size), 
                  desc="Inference", total=total_batches):
        batch_data = processed_data[i:i + args.batch_size]
        
        # prepare tensors on the correct device
        user_input_ids = torch.stack([item['user_input_ids'] for item in batch_data]).to(device)
        user_attention_mask = torch.stack([item['user_attention_mask'] for item in batch_data]).to(device)
        item_input_ids = torch.stack([item['item_input_ids'] for item in batch_data]).to(device)
        item_attention_mask = torch.stack([item['item_attention_mask'] for item in batch_data]).to(device)
        
        # forward
        with torch.no_grad():
            outputs = model(
                user_input_ids=user_input_ids,
                user_attention_mask=user_attention_mask,
                item_input_ids=item_input_ids,
                item_attention_mask=item_attention_mask,
            )
            
            # extract embeddings
            if hasattr(outputs, 'logits'):
                emb = outputs.logits
            else:
                emb = outputs
            
            # model returns concatenated [user_emb, item_emb] by batch
            batch_size = len(batch_data)
            if emb.shape[0] == 2 * batch_size:
                user_emb = emb[:batch_size]   # first half: users
                item_emb = emb[batch_size:]   # second half: items
                
                # pick the side according to mode
                if args.mode == 'user':
                    emb = user_emb
                else:
                    emb = item_emb
            elif emb.shape[0] == batch_size:
                # already the correct size
                pass
            else:
                print(f"⚠️  Unexpected output shape: {emb.shape}, expected: ({batch_size}, embedding_dim)")
                # fallback to first batch_size rows
                emb = emb[:batch_size]
            
            all_embeddings.append(emb.cpu().numpy())
    
    inference_time = time.time() - inference_start_time
    
    # merge all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # save
    output_file = os.path.join(args.output_dir, f"{args.mode}_embeddings.pkl")
    save_embeddings(final_embeddings, output_file)
    
    # perf stats
    print(f"\n=== Performance summary ===")
    print(f"📊 Data source: {'preprocessed' if is_preprocessed else 'on-the-fly tokenization'}")
    print(f"⏱️  Inference time: {inference_time:.2f}s")
    print(f"🚀 Throughput: {len(processed_data)/inference_time:.1f} samples/s")
    
    print(f"\n✅ {args.mode} embedding inference complete")
    print(f"Output shape: {final_embeddings.shape}")
    print(f"Saved to: {output_file}")
    print(f"Value range: [{final_embeddings.min():.4f}, {final_embeddings.max():.4f}]")

def main():
    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--mode", type=str, choices=['user', 'item'], required=True, 
                        help="Inference mode: 'user' or 'item'")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--user_prompt_length", type=int, default=10, help="Reserved tokens for user soft prompt")
    parser.add_argument("--item_prompt_length", type=int, default=10, help="Reserved tokens for item soft prompt")
    parser.add_argument("--pool_type", type=str, default="avg", help="Pooling type")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed launch (auto passed)")
    parser.add_argument("--load_all_params", action="store_true", help="Load all trained params (if available)")
    
    args = parser.parse_args()
    batch_inference(args)

if __name__ == "__main__":
    main()