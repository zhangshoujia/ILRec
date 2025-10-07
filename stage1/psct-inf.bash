#!/bin/bash
# --- Inference configuration ---
# Model and data paths
MODEL_NAME="resources/llama-7b"

OUTPUT_DIR="./saved_models/amazon_all_train_example" # Model and results output directory

USER_FILE="./data/amazon/amazon_all_train_json_samples/unique_users.json"  # user data file
ITEM_FILE="./data/amazon/amazon_all_train_json_samples/unique_items.json"  # item data file

# Inference hyperparameters
BATCH_SIZE=16
MAX_LENGTH=512
USER_PROMPT_LENGTH=64
ITEM_PROMPT_LENGTH=64
POOL_TYPE="avg"
TAU=1.0

# GPU config
GPU_IDS="6"  # Single-GPU inference is more stable

MASTER_PORT=$(shuf -n 1 -i 10000-65535)  # random port (not used here but kept for completeness)

# --- Start inference ---
echo "==================================="
echo "     Starting inference with all trained parameters..."
echo "==================================="
echo "Using GPU(s): ${GPU_IDS}"
echo "Model path: ${MODEL_NAME}"
echo "Trained parameters path: ${OUTPUT_DIR}"

# Check that trained parameter files exist
echo "Checking trained parameter files..."
if [ ! -f "${OUTPUT_DIR}/adapter_config.json" ]; then
    echo "❌ LoRA config file not found: ${OUTPUT_DIR}/adapter_config.json"
    exit 1
fi

if [ ! -f "${OUTPUT_DIR}/prompt_params.bin" ]; then
    echo "❌ Soft prompt parameters file not found: ${OUTPUT_DIR}/prompt_params.bin"
    exit 1
fi

if [ ! -f "${OUTPUT_DIR}/model_adapter_params.bin" ]; then
    echo "❌ Model adapter parameters file not found: ${OUTPUT_DIR}/model_adapter_params.bin"
    exit 1
fi

echo "✅ All trained parameter files are present."

# Inference for user embeddings
echo "Running user embedding inference..."
CUDA_VISIBLE_DEVICES=${GPU_IDS} python batch_inference.py \
    --input_file ${USER_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_NAME} \
    --mode user \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --user_prompt_length ${USER_PROMPT_LENGTH} \
    --item_prompt_length ${ITEM_PROMPT_LENGTH} \
    --pool_type ${POOL_TYPE} \
    --tau ${TAU}

# Inference for item embeddings
echo "Running item embedding inference..."
CUDA_VISIBLE_DEVICES=${GPU_IDS} python batch_inference.py \
    --input_file ${ITEM_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_NAME} \
    --mode item \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --user_prompt_length ${USER_PROMPT_LENGTH} \
    --item_prompt_length ${ITEM_PROMPT_LENGTH} \
    --pool_type ${POOL_TYPE} \
    --tau ${TAU}

echo "Inference completed! Embeddings saved to ${OUTPUT_DIR}"
echo "User embeddings: ${OUTPUT_DIR}/ft_usr_emb.npy"
echo "Item embeddings: ${OUTPUT_DIR}/ft_itm_emb.npy"