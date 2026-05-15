# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ILRec is a two-stage recommender system research project that combines LLM-based semantic embedding generation with graph-based collaborative filtering.

## Running the Pipeline

### Stage 1 — ICFT (LLM Fine-tuning & Embedding Inference)

```bash
cd stage1
bash icft-tun.bash   # fine-tune LLM with LoRA + soft prompts via DeepSpeed
bash icft-inf.bash   # run inference to produce ft_usr_emb.pkl and ft_itm_emb.pkl
```

Environment: Python 3.9–3.10, separate conda env `ilrec-stage1`. See `stage1/requirements1.txt`.

### Stage 2 — TriRA (Graph-based Recommender Training)

```bash
cd stage2
python encoder/train_encoder.py --model lightgcn_pro --dataset book --cuda 0
```

Available models: `lightgcn_pro`, `gccf_pro`, `cogcl_pro`, `nescl_pro`  
Available datasets: `book`, `yelp`, `steam`, `toy`

Environment: Python 3.9, PyTorch 1.13.1+cu116, separate conda env `ilrec-stage2`. See `stage2/requirements2.txt`.

## Architecture

### Data Flow

```
Stage 1 (ICFT)
  LLM (LLaMA + LoRA + soft prompts)
  → ft_usr_emb.pkl, ft_itm_emb.pkl
        ↓
Stage 2 (TriRA)
  GNN model + semantic graph alignment
  ← trn/val/tst_mat.pkl (CF interaction matrices)
  ← ft_usr_emb.pkl, ft_itm_emb.pkl
  ← semantic_graph_top<K>_mat.pkl
```

### Stage 1 Key Files

- `stage1/main_llm.py` — training entry point (launched via DeepSpeed in `icft-tun.bash`)
- `stage1/batch_inference.py` — embedding extraction entry point
- `stage1/llm/llama.py` — model definition with LoRA + soft prompt integration
- `stage1/llm/data_processor/` — dataset loading and prompt construction
- `stage1/trainers/` — training loop logic

### Stage 2 Key Files

- `stage2/encoder/train_encoder.py` — training entry point; wires together data handler, model, and trainer
- `stage2/encoder/config/configurator.py` — central config loader; reads YAML model configs, loads Stage-1 embeddings and semantic graph from `./data/<dataset>/`
- `stage2/encoder/models/bulid_model.py` — dynamic model import via `models.<data_type>.<model_name>`
- `stage2/encoder/models/general_cf/lightgcn_pro.py` — reference model implementation; combines BPR loss with contrastive alignment between CF and semantic embeddings
- `stage2/encoder/config/modelconf/<model>.yml` — per-model hyperparameters including `semantic_topk` (controls which semantic graph file is loaded)

### Stage 2 Data Layout

All data files live under `stage2/data/<dataset>/`:
- `trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl` — sparse user-item interaction matrices
- `ft_usr_emb.pkl`, `ft_itm_emb.pkl` — semantic embeddings from Stage 1
- `semantic_graph_top<K>_mat.pkl` — precomputed semantic similarity graph; K is set by `semantic_topk` in the model YAML
