#!/bin/bash

# ==========================================
# GFMRec Training Script - Dataset-based Config System
# ==========================================

# Default config
MODEL_NAME="resources/llama-7b"
EMBEDDING_DIM=1536  # Change this to set embedding dimension (e.g., 1024, 1536, 2048)

# Additional configurable params
LORA_ANCHOR_WEIGHT=0.005
PROMPT_DIV_WEIGHT=0.005
CONTRASTIVE_ANCHOR_WEIGHT=0.005
ANCHOR_CONTRASTIVE_TEMPERATURE=1.0
INTER_CONTRASTIVE_TEMPERATURE=1.0

# Check CLI arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name> [options]"
    echo ""
    echo "Supported datasets:"
    echo "  book_1k     - book 1K sample dataset"
    echo "  book        - book full dataset"
    echo "  yelp          - Yelp full dataset"
    echo "  steam         - Steam full dataset"
    echo "  toy           - Toy full dataset"
    echo ""
    echo "Optional arguments:"
    echo "  --lora_anchor_weight <val>        - LoRA anchor weight (default: $LORA_ANCHOR_WEIGHT)"
    echo "  --prompt_div_weight <val>         - Prompt diversity weight (default: $PROMPT_DIV_WEIGHT)"
    echo "  --contrastive_anchor_weight <val> - Contrastive anchor weight (default: $CONTRASTIVE_ANCHOR_WEIGHT)"
    echo "  --contrastive_temperature <val>   - Contrastive temperature (default: $ANCHOR_CONTRASTIVE_TEMPERATURE)"
    echo ""
    echo "Example: $0 book --lora_anchor_weight 0.01 --prompt_div_weight 0.01"
    exit 1
fi

DATASET_NAME=$1
shift  # remove first arg (dataset name)

# Parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_anchor_weight)
            LORA_ANCHOR_WEIGHT="$2"
            shift 2
            ;;
        --prompt_div_weight)
            PROMPT_DIV_WEIGHT="$2"
            shift 2
            ;;
        --contrastive_anchor_weight)
            CONTRASTIVE_ANCHOR_WEIGHT="$2"
            shift 2
            ;;
        --contrastive_temperature)
            ANCHOR_CONTRASTIVE_TEMPERATURE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help to see usage."
            exit 1
            ;;
    esac
done

# Dataset-specific config
configure_dataset() {
    local dataset=$1
    
    case $dataset in
        "book")
            echo "Configuring book full dataset..."
            TRAIN_FILE="./data/RLMRecdata/book_all_train_json_samples/train_useritem_data.json"
            VALID_FILE="./data/RLMRecdata/book_all_train_json_samples/valid_useritem_data.json"
            OUTPUT_DIR="./saved_models/${EMBEDDING_DIM}/book_all_train_example"
            
            LEARNING_RATE=5e-6
            BATCH_SIZE=2
            NUM_EPOCHS=3
            MAX_STEPS=15000
            EVAL_STEPS=2000
            SAVE_STEPS=2000
            LOGGING_STEPS=50
            EARLY_STOPPING_PATIENCE=3
            
            NUM_GPUS=4
            GPU_IDS="2,4,5,6"
            ;;
        *)
            echo "Error: unsupported dataset '$dataset'"
            echo "Supported: book, yelp, steam, toy"
            exit 1
            ;;
    esac
}

# Common config
configure_common() {
    # DeepSpeed config
    DEEPSPEED_CONFIG="configs/ds_config_stable.json"
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    
    # Model params
    MAX_SOURCE_LENGTH=512
    MAX_TARGET_LENGTH=512
    PROMPT_LENGTH=10
    
    # CUDA env
    export CUDA_VISIBLE_DEVICES=${GPU_IDS}
    export TOKENIZERS_PARALLELISM=false
}

# Show config
show_config() {
    echo ""
    echo "==================================="
    echo "        Training Configuration"
    echo "==================================="
    echo "Dataset: $DATASET_NAME"
    echo "Model: $MODEL_NAME"
    echo "Embedding dim: $EMBEDDING_DIM"
    echo "Train file: $TRAIN_FILE"
    echo "Valid file: $VALID_FILE"
    echo "Output dir: $OUTPUT_DIR"
    echo "Learning rate: $LEARNING_RATE"
    echo "Batch size: $BATCH_SIZE"
    echo "Epochs: $NUM_EPOCHS"
    echo "Max steps: $MAX_STEPS"
    echo "Eval steps: $EVAL_STEPS"
    echo "Save steps: $SAVE_STEPS"
    echo "Num GPUs: $NUM_GPUS"
    echo "GPU IDs: $GPU_IDS"
    echo ""
    echo "Advanced params:"
    echo "LoRA anchor weight: $LORA_ANCHOR_WEIGHT"
    echo "Prompt diversity weight: $PROMPT_DIV_WEIGHT"
    echo "Contrastive anchor weight: $CONTRASTIVE_ANCHOR_WEIGHT"
    echo "Contrastive temperature: $ANCHOR_CONTRASTIVE_TEMPERATURE"
    echo "==================================="
    echo ""
}

# Run training
run_training() {
    echo "Starting training..."
    echo "DeepSpeed config: $DEEPSPEED_CONFIG"
    echo "Master port: $MASTER_PORT"
    echo ""
    
    deepspeed --include localhost:${GPU_IDS} --master_port $MASTER_PORT main_llm.py \
        --model_choice useritem_prompt \
        --do_train \
        --do_eval \
        --bf16 \
        --train_file ${TRAIN_FILE} \
        --validation_file ${VALID_FILE} \
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --max_source_length ${MAX_SOURCE_LENGTH} \
        --max_target_length ${MAX_TARGET_LENGTH} \
        --user_prompt_length ${PROMPT_LENGTH} \
        --item_prompt_length ${PROMPT_LENGTH} \
        --pool_type avg \
        --tau ${INTER_CONTRASTIVE_TEMPERATURE} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --max_grad_norm 0.5 \
        --warmup_ratio 0.1 \
        --logging_steps ${LOGGING_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --evaluation_strategy steps \
        --save_strategy steps \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better false \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --overwrite_output_dir \
        --lr_scheduler_type cosine \
        --seed 42 \
        --max_steps ${MAX_STEPS} \
        --deepspeed ${DEEPSPEED_CONFIG} \
        --use_global_batch_loss \
        --enable_lora_anchor true \
        --lora_anchor_weight ${LORA_ANCHOR_WEIGHT} \
        --enable_prompt_diversity true \
        --prompt_div_weight ${PROMPT_DIV_WEIGHT} \
        --enable_contrastive_anchor true \
        --contrastive_anchor_weight ${CONTRASTIVE_ANCHOR_WEIGHT} \
        --contrastive_temperature ${ANCHOR_CONTRASTIVE_TEMPERATURE} \
        --save_safetensors false
}

# Main
main() {
    echo "==================================="
    echo "       GFMRec Training Launcher"
    echo "==================================="
    echo "Dataset: $DATASET_NAME"
    echo "Embedding dim: $EMBEDDING_DIM"
    echo ""
    
    # Dataset config
    configure_dataset $DATASET_NAME
    
    # Common config
    configure_common
    
    # Show
    show_config
    
    # Confirm
    read -p "Start training? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training canceled."
        exit 0
    fi
    
    # Run
    run_training
}

# Run main
main