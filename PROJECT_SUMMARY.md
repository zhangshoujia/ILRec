# ILRec 项目总结

## 项目概述

ILRec（Interaction-Aware LLM-Driven Semantic Representation for Recommendation）是一个**两阶段推荐系统**，将大语言模型（LLM）的语义理解能力与图神经网络（GNN）的协同过滤能力结合起来。

---

## 两阶段架构

### Stage 1 — ICFT（LLM 微调 + 语义嵌入生成）
- 基于 **LLaMA** 模型，使用 LoRA + 软提示（soft prompt）进行微调
- 输入：用户/物品的文本描述 + 交互历史
- 输出：`ft_usr_emb.pkl` 和 `ft_itm_emb.pkl`（用户和物品的语义向量）
- 训练工具：DeepSpeed 多卡训练，PEFT/LoRA 高效微调

### Stage 2 — TriRA（语义对齐推荐模型）
- 基于 **PyTorch Geometric** 的 GNN 推荐模型（LightGCN、GCCF 等变体）
- 同时利用：
  - 交互图（协同过滤信号）
  - 语义图（基于 Stage 1 嵌入的相似度图）
  - LLM 语义嵌入（直接对齐）
- 损失函数：BPR + 多个对比/对齐损失

---

## 数据流

```
用户/物品文本 + 交互数据
        ↓
  Stage 1: LLM 微调 & 推理
        ↓
  语义嵌入 (pkl/npy)
        ↓
  Stage 2: GNN 对齐训练
        ↓
  推荐结果 (Recall/NDCG)
```

---

## 技术栈

| 阶段 | 核心依赖 |
|------|---------|
| Stage 1 | PyTorch 2.7, Transformers 4.30, DeepSpeed, PEFT |
| Stage 2 | PyTorch 1.13, PyTorch Geometric, scipy, scikit-learn |

支持数据集：Book、Yelp、Steam、Toy

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `stage1/icft-tun.bash` | Stage 1 训练启动脚本 |
| `stage1/main_llm.py` | LLM 微调主程序 |
| `stage1/batch_inference.py` | 语义嵌入推理脚本 |
| `stage2/encoder/train_encoder.py` | Stage 2 训练入口 |
| `stage2/encoder/config/configurator.py` | 配置加载（含嵌入路径） |
| `stage2/encoder/models/general_cf/lightgcn_pro.py` | LightGCN 对齐模型实现 |
