# ILRec: Interaction-Aware LLM-Driven Semantic Representation Generation and Alignment for Recommendation

> **Anonymous code release (double-blind).**  
> This repository implements **ILRec**, a two-stage framework that couples the semantic generation ability of Large Language Models (LLMs) with the higher-order relational modeling of Graph Neural Networks (GNNs) to improve recommendation.

- **ILRec.**  
  1) **Interaction‑Aware Contrastive Fine‑Tuning (ICFT):** fine‑tune an LLM to generate interaction‑aware user/item semantic embeddings that pull together observed pairs and push apart non‑interacted pairs.  
  2) **Multi‑Level Representation Alignment (MLRA):** align **semantic** and **collaborative** representations at both **node** and **interaction** levels with contrastive objectives.
- **Outcome.** On four public datasets, ILRec consistently boosts GNN recommenders. Ablations verify each component.

---

## Repository Layout

```
.
├─ stage1/                                # Stage-1: LLM fine-tuning + semantic inference (ICFT)
│  ├─ configs/
│  │   └─ ds_config_stable.json           # DeepSpeed config
│  ├─ data/
│  │   ├─ amazon/ yelp/ steam/ toy/       # JSON/PKL for tuning & inference
│  ├─ figs/
│  ├─ llm/
│  │   ├─ data_processor/                 # tokenization & processors
│  │   ├─ peft/                           # LoRA utilities
│  │   ├─ arguments.py, llama.py, lora_cls.py, trainer_seq2seq.py
│  ├─ resources/
│  │   └─ llama-7b/                       # base LLM weights (place locally)
│  ├─ saved_models/                       # fine-tuned adapters + prompt params
│  ├─ trainers/, utils/
│  ├─ main_llm.py                         # training entry (ICFT)
│  ├─ batch_inference.py                  # batched embedding inference (user/item)
│  ├─ psct-tun.bash                       # script: fine-tune LLMs (ICFT)
│  └─ psct-inf.bash                       # script: infer semantics for all users/items
│
├─ stage2/                                # Stage-2: GNN training + multi-level alignment (MLRA)
│  ├─ data/                               # matrices + (fine-tuned) semantics
│  ├─ encoder/
│  │   ├─ config/modelconf/               # YAMLs (lightgcn[_pro], gccf[_pro], …)
│  │   ├─ data_utils/, models/, trainer/  # SSLRec-style pipeline
│  │   └─ train_encoder.py                # training entry (MLRA)
│
├─ LICENSE
└─ README.md                              # (this file)
```

---

## 1. Environment

We recommend one conda env for both stages (adjust CUDA to your setup).

```bash
conda create -y -n ilrec python=3.9
conda activate ilrec

# PyTorch (pick the wheel that matches your CUDA; example: cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# LLM stack
pip install transformers datasets accelerate deepspeed peft

# Utilities
pip install numpy scipy scikit-learn tqdm pyyaml

# Stage-2 sparse ops (PyG): choose the wheel that matches your torch & CUDA
# See https://data.pyg.org/whl/ for the correct URLs for your versions.
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

> Tip: ICFT with LoRA is far lighter than full fine‑tuning, but GPUs with ≥24 GB help.

---

## 2. Data

We evaluate on **Book**, **Yelp**, **Steam**, and a **Toy** demo.  
Stage‑2 expects sparse matrices and text resources:

```
stage2/data/{amazon|yelp|steam|toy}/
  ├─ trn_mat.pkl          # training interactions (scipy sparse)
  ├─ val_mat.pkl          # validation
  ├─ tst_mat.pkl          # test
  ├─ usr_prf.pkl          # (optional) user profile texts
  ├─ itm_prf.pkl          # (optional) item profile texts
  ├─ ft_usr_emb.pkl       # ← produced by Stage-1 (ILRec semantics)
  └─ ft_itm_emb.pkl       # ← produced by Stage-1 (ILRec semantics)
```

If you precompute a semantic interaction graph, place it as:
```
stage2/data/<dataset>/semantic_graph_top<K>_mat.pkl
```
It will be auto‑loaded based on the YAML’s `semantic_topk`.

---

## 3. Stage‑1 — Interaction‑Aware Contrastive Fine‑Tuning (ICFT)

**Goal:** adapt a base LLM (e.g., LLaMA‑7B) to output **interaction‑aware** user/item embeddings.

### 3.1 Prepare resources

- Put base weights under:
  ```
  stage1/resources/llama-7b/
  ```
- Provide JSONL files under `stage1/data/<dataset>/`, e.g.:
  - `train_useritem_data.json`, `valid_useritem_data.json` (pairs for ICFT)
  - `unique_users.json`, `unique_items.json` (for full inference)

### 3.2 Fine‑tune (LoRA + soft prompts + adapters)

```bash
cd stage1
bash psct-tun.bash amazon \
  --lora_anchor_weight 0.005 \
  --prompt_div_weight 0.005 \
  --contrastive_anchor_weight 0.005 \
  --contrastive_temperature 1.0
```

Artifacts (lightweight only) are saved to:

```
stage1/saved_models/<EMB_DIM>/<run>/
  ├─ adapter_config.json
  ├─ adapter_model.bin
  ├─ prompt_params.bin
  ├─ model_adapter_params.bin
  └─ training_config.json
```

> The ICFT loss pulls together observed (user, item) pairs and separates negatives; optional **LoRA‑anchor** and **prompt‑diversity** regularizers are available (see bash flags). The DeepSpeed config is `stage1/configs/ds_config_stable.json`.

### 3.3 Inference (generate semantics for all users/items)

```bash
cd stage1
bash psct-inf.bash
```

Outputs:
```
stage1/saved_models/<run>/user_embeddings.{pkl,npy}
stage1/saved_models/<run>/item_embeddings.{pkl,npy}
```

Copy (or symlink) into **Stage‑2** using the expected filenames:

```bash
cp stage1/saved_models/<run>/user_embeddings.pkl stage2/data/amazon/ft_usr_emb.pkl
cp stage1/saved_models/<run>/item_embeddings.pkl stage2/data/amazon/ft_itm_emb.pkl
```

> If you see errors like “missing `adapter_config.json` / `prompt_params.bin`”, re‑check that `psct-tun.bash` finished successfully and that `psct-inf.bash` points to the same `OUTPUT_DIR`.

---

## 4. Stage‑2 — Multi‑Level Representation Alignment (MLRA)

**Goal:** train a GNN recommender and align **semantic** (from Stage‑1) with **collaborative** representations at both **node** and **interaction** levels.

### 4.1 Choose a model

YAMLs live in `stage2/encoder/config/modelconf/`:

- **With MLRA**: `lightgcn_pro.yml`, `gccf_pro.yml`
- **Baselines**: `lightgcn.yml`, `gccf.yml`

Supported datasets: `amazon`, `yelp`, `steam`, `toy`.

### 4.2 Train

```bash
cd stage2
python encoder/train_encoder.py \
  --model lightgcn_pro \
  --dataset amazon \
  --cuda 0
```

The trainer will:

- Read `{trn,val,tst}_mat.pkl`
- Load `ft_usr_emb.pkl` / `ft_itm_emb.pkl`
- (Optionally) load `semantic_graph_top<K>_mat.pkl`
- Optimize a mixture of: BPR, supervised contrastive terms, cross‑view (node + interaction) alignment losses, and regularization (see YAML).

#### Quick hyper‑parameter overrides

You can override YAML keys at runtime:

```bash
export RLMREC_TUNING_OVERRIDES='{
  "sup_cl_weight": 0.1,
  "cv_sup_cl_weight": 0.1,
  "emb_cl_weight": 0.05,
  "contrast_temperature": 0.25,
  "sup_contrast_temperature": 0.25,
  "semantic_topk": 10
}'
python encoder/train_encoder.py --model lightgcn_pro --dataset amazon --cuda 0
```

---

## 5. Minimal Quickstart

```bash
# 0) Create env
conda create -y -n ilrec python=3.9 && conda activate ilrec
# (install deps as in Section 1)

# 1) Prepare data under stage2/data/<dataset>/ and stage1/data/<dataset>/

# 2) Stage-1: tune + infer
cd stage1
bash psct-tun.bash amazon
bash psct-inf.bash
cp saved_models/<run>/user_embeddings.pkl ../stage2/data/amazon/ft_usr_emb.pkl
cp saved_models/<run>/item_embeddings.pkl ../stage2/data/amazon/ft_itm_emb.pkl

# 3) Stage-2: train recommender
cd ../stage2
python encoder/train_encoder.py --model lightgcn_pro --dataset amazon --cuda 0
```

---

## 6. What’s Inside (by stage)

### Stage‑1 (ICFT)

- **Backbone:** LLaMA‑style encoder (see `stage1/llm/llama.py`)
- **Trainables:** LoRA adapters (`stage1/llm/lora_cls.py`), soft prompts, lightweight projection heads
- **Losses:** interaction‑aware contrastive; optional LoRA‑anchor regularization; prompt diversity regularization
- **Trainer:** custom `MedRecTrainer` / `trainer_seq2seq.py` with lightweight checkpointing (adapters + prompts only)
- **Scripts:** `psct-tun.bash` (training), `psct-inf.bash` (inference)

### Stage‑2 (MLRA)

- **Backbones:** `lightgcn_pro.py`, `gccf_pro.py` (dual‑graph propagation with semantic graph)
- **Losses:** BPR, supervised contrastive, cross‑view (node-/interaction‑level) alignment, embedding‑level InfoNCE, regularization
- **Configs:** YAMLs per model/dataset in `encoder/config/modelconf/`
- **Entrypoint:** `encoder/train_encoder.py`

---

## 7. Troubleshooting

- **`psct-inf.bash` complains about missing files.**  
  Ensure `psct-tun.bash` finished and all of the following exist in the same `OUTPUT_DIR`:
  `adapter_config.json`, `adapter_model.bin`, `prompt_params.bin`, `model_adapter_params.bin`.
- **OOM during ICFT.**  
  Reduce batch size, enable gradient checkpointing (via HF/DS config), or switch to fewer GPUs. LoRA rank/dropout can also be lowered.
- **PyG sparse wheels fail to install.**  
  Make sure the wheel versions exactly match your installed `torch` and CUDA. See the PyG index URL in Section 1.
- **Semantic graph not found.**  
  The code will still run (dynamic construction) but for full reproduction provide `semantic_graph_top<K>_mat.pkl`.

---

## 8. Acknowledgements

Stage‑2 training follows an SSLRec‑style pipeline; several dataset/file conventions follow prior public work on LLM‑enhanced recommendation (e.g., RLMRec). We thank the maintainers and the providers of the public datasets (Yelp, Amazon, Steam).
