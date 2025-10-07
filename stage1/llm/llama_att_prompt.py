# here put the import lib
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class UserItemPromptLlama(LlamaPreTrainedModel):
    """
    用户-物品交互对的软提示学习模型
    支持用户和物品的软提示token、权重学习和adapter
    """
    main_input_name = "user_input_ids"  # 告诉Trainer主要的输入参数名
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 1) 先拿到 hidden_size，后续所有层都要用到
        self.hidden_size = config.hidden_size

        # 2) 基础 Llama 模型
        self.model = LlamaModel(config)

        # 3) 配置参数
        self.pool_type = kwargs.pop("pool_type", "avg")
        tau_init = kwargs.pop("tau", 1.0)
        self.tau = nn.Parameter(torch.tensor([tau_init], dtype=torch.float32))

        # 4) 软提示长度 & 最终投影维度
        self.user_prompt_length = kwargs.pop("user_prompt_length", 10)
        self.item_prompt_length = kwargs.pop("item_prompt_length", 10)
        self.emb_dim = kwargs.pop("emb_dim", 1536)  # 如果你期望 1024，就把默认值改为 1024

        # 5) 注意力相关层（现在 self.hidden_size 已就绪，可以安全使用）
        self.user_prompt_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_prompt_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 用 [user_pool ; item_pool] 作为查询，分别生成 user/item 的 query
        self.user_query = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.item_query = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.attn_scale = self.hidden_size ** 0.5  # sqrt(d)

        # 6) 软提示参数（可训练）
        self.user_prompts = nn.Parameter(
            torch.randn(self.user_prompt_length, self.hidden_size) * 0.02
        )
        self.user_prompt_weights = nn.Parameter(
            torch.ones(self.user_prompt_length) / self.user_prompt_length
        )

        self.item_prompts = nn.Parameter(
            torch.randn(self.item_prompt_length, self.hidden_size) * 0.02
        )
        self.item_prompt_weights = nn.Parameter(
            torch.ones(self.item_prompt_length) / self.item_prompt_length
        )

        # 7) Adapter 层（可选）
        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )

        # 8) 最终投影层（把 hidden_size -> emb_dim）
        self.final_projection = nn.Linear(self.hidden_size, self.emb_dim)

        # 9) 其它
        self.use_global_batch_loss = kwargs.pop("use_global_batch_loss", False)

        # 10) 初始化（LlamaPreTrainedModel 会处理线性层的权重初始化）
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def _mean_pool(self, embs, mask):
        # embs: [B, L, d], mask: [B, L] 或 None
        if mask is None:
            return embs.mean(dim=1)  # [B, d]
        m = mask.unsqueeze(-1).float()  # [B, L, 1]
        s = (embs * m).sum(dim=1)       # [B, d]
        l = m.sum(dim=1).clamp(min=1.0) # [B, 1]
        return s / l

    def _attend_prompts(self, prompts,  # [P, d]
                        query,          # [B, d]
                        key_proj,       # nn.Linear(d->d) for K
                        prior_weights=None):
        """
        用 query 对 prompt 做点积注意力，得到每个样本对 P 个 prompt 的权重，然后按权重缩放每个 prompt token。
        返回: [B, P, d]，适合与原始嵌入在序列维度拼接。
        """
        # K: [P, d]
        K = key_proj(prompts)                          

        # 注意力 logits: [B, P]
        # (q @ K^T) / sqrt(d)
        attn_logits = torch.matmul(query, K.t()) / self.attn_scale

        # 可选：把原有的可训练 prior 作为偏置（让模型在“注意力”基础上仍可保留先验偏好）
        if prior_weights is not None:
            # prior: [P] -> log_softmax 后加到 logits 上
            prior_bias = torch.log_softmax(prior_weights, dim=0)  # [P]
            attn_logits = attn_logits + prior_bias.unsqueeze(0)   # [B, P]

        # 权重: [B, P]
        attn = torch.softmax(attn_logits, dim=-1)

        # 按权重缩放每个 prompt：先扩展到 [B, P, 1]，逐 token 乘以原始 prompts（而不是 K）
        weighted_prompts = attn.unsqueeze(-1) * prompts.unsqueeze(0)  # [B, P, d]
        return weighted_prompts
    
    def _process_user_item_pair(self, user_input_ids, item_input_ids,
                                user_attention_mask=None, item_attention_mask=None):
        """
        处理用户-物品对：先取 user/item 的句向量（mean pool），
        再以它们为 query 对各自的 prompt 做注意力加权，最后与原始 token 嵌入拼接。
        """
        batch_size = user_input_ids.shape[0]

        # 1) 原始 token 嵌入
        user_embeddings = self.model.embed_tokens(user_input_ids)  # [B, Lu, d]
        item_embeddings = self.model.embed_tokens(item_input_ids)  # [B, Li, d]

        # 2) 句向量（可换成最后一层 hidden state 再池化，这里先用 embedding 的 mean pool）
        u_pool = self._mean_pool(user_embeddings, user_attention_mask)  # [B, d]
        i_pool = self._mean_pool(item_embeddings, item_attention_mask)  # [B, d]

        # 3) 生成查询（query）
        ui = torch.cat([u_pool, i_pool], dim=-1)  # [B, 2d]
        iu = torch.cat([i_pool, u_pool], dim=-1)  # [B, 2d]
        q_u = self.user_query(ui)                 # [B, d] ——用于“挑选/加权 user prompts”
        q_i = self.item_query(iu)                 # [B, d] ——用于“挑选/加权 item prompts”

        # 4) 基于注意力为每个样本生成加权后的 prompt（保留每个 token 的独立性）
        user_weighted_prompts = self._attend_prompts(
            self.user_prompts, q_u, self.user_prompt_key, prior_weights=self.user_prompt_weights
        )  # [B, Pu, d]

        item_weighted_prompts = self._attend_prompts(
            self.item_prompts, q_i, self.item_prompt_key, prior_weights=self.item_prompt_weights
        )  # [B, Pi, d]

        # 5) 与原始嵌入拼接
        user_enhanced_embeddings = torch.cat([user_weighted_prompts, user_embeddings], dim=1)  # [B, Pu+Lu, d]
        item_enhanced_embeddings = torch.cat([item_weighted_prompts, item_embeddings], dim=1)  # [B, Pi+Li, d]

        # 6) 同步扩展 attention_mask（前置的 prompt 位置全部置 1）
        if user_attention_mask is None:
            user_attention_mask = torch.ones(user_input_ids.size(0), user_input_ids.size(1),
                                            dtype=torch.long, device=user_input_ids.device)
        if item_attention_mask is None:
            item_attention_mask = torch.ones(item_input_ids.size(0), item_input_ids.size(1),
                                            dtype=torch.long, device=item_input_ids.device)

        Pu = user_weighted_prompts.size(1)
        Pi = item_weighted_prompts.size(1)
        user_prompt_mask = torch.ones(batch_size, Pu, dtype=user_attention_mask.dtype,
                                    device=user_attention_mask.device)
        item_prompt_mask = torch.ones(batch_size, Pi, dtype=item_attention_mask.dtype,
                                    device=item_attention_mask.device)

        user_enhanced_mask = torch.cat([user_prompt_mask, user_attention_mask], dim=1)  # [B, Pu+Lu]
        item_enhanced_mask = torch.cat([item_prompt_mask, item_attention_mask], dim=1)  # [B, Pi+Li]

        return (user_enhanced_embeddings, item_enhanced_embeddings,
                user_enhanced_mask, item_enhanced_mask)
    
    def forward(
        self,
        user_input_ids: torch.LongTensor = None,
        item_input_ids: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,  # 添加标准参数支持
        attention_mask: Optional[torch.Tensor] = None,
        user_attention_mask: Optional[torch.Tensor] = None,  # 添加用户注意力掩码
        item_attention_mask: Optional[torch.Tensor] = None,  # 添加物品注意力掩码
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        前向传播
        Args:
            user_input_ids: 用户文本的token ids
            item_input_ids: 物品文本的token ids
            input_ids: 标准参数名（兼容性）
            attention_mask: 注意力掩码
            user_attention_mask: 用户注意力掩码
            item_attention_mask: 物品注意力掩码
            labels: 标签（用于计算损失）
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 处理标准参数名的情况
        if input_ids is not None and user_input_ids is None and item_input_ids is None:
            raise ValueError("请使用user_input_ids和item_input_ids参数，而不是input_ids")
        
        # 确保必要的参数存在
        if user_input_ids is None or item_input_ids is None:
            raise ValueError("user_input_ids和item_input_ids都是必需的参数")
        
        # 处理注意力掩码
        if user_attention_mask is None:
            user_attention_mask = attention_mask
        if item_attention_mask is None:
            item_attention_mask = attention_mask
        
        # 处理用户-物品对，添加软提示
        user_enhanced_embeddings, item_enhanced_embeddings, user_enhanced_mask, item_enhanced_mask = \
            self._process_user_item_pair(user_input_ids, item_input_ids, user_attention_mask, item_attention_mask)
        
        # 通过Llama模型处理用户嵌入
        user_transformer_outputs = self.model(
            inputs_embeds=user_enhanced_embeddings,
            attention_mask=user_enhanced_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        user_hidden_states = user_transformer_outputs[0]
        
        # 通过Llama模型处理物品嵌入
        item_transformer_outputs = self.model(
            inputs_embeds=item_enhanced_embeddings,
            attention_mask=item_enhanced_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        item_hidden_states = item_transformer_outputs[0]
        
        # 池化得到用户和物品的表示（adapter/投影之前）
        user_pooled = self._get_pool_emb(user_hidden_states, user_enhanced_mask)
        item_pooled = self._get_pool_emb(item_hidden_states, item_enhanced_mask)

        # 保存一份“前 adapter/投影”的拼接表示，供锚定方案A使用
        pre_adapter_pooled = torch.cat([user_pooled, item_pooled], dim=0)
        
        # 在 fp16 训练时，池化操作可能把张量变回 float32，导致与 adapter(fp16) 权重 dtype 不匹配
        target_dtype = self.adapter[0].weight.dtype  # adapter 权重的 dtype（fp16 或 fp32）
        if user_pooled.dtype != target_dtype:
            user_pooled = user_pooled.to(target_dtype)
        if item_pooled.dtype != target_dtype:
            item_pooled = item_pooled.to(target_dtype)
        
        # 通过 adapter
        user_embeddings = self.adapter(user_pooled)
        item_embeddings = self.adapter(item_pooled)
        
        # 通过最终投影层，将4096维投影到1024维
        user_embeddings = self.final_projection(user_embeddings)
        item_embeddings = self.final_projection(item_embeddings)

        # 输出embedding大小信息，验证batch_size
        # print("=== Embedding Size Debug ===")
        # print(f"user_embeddings.size: {user_embeddings.size()}")
        # print(f"item_embeddings.size: {item_embeddings.size()}")
        # print(f"Batch size (from user_embeddings): {user_embeddings.size(0)}")
        # print(f"Hidden size: {user_embeddings.size(1)}")
        # print(f"Total samples processed: {user_embeddings.size(0) + item_embeddings.size(0)}")
        # print(f"Expected per_device_batch_size: 2")
        # print(f"Expected total_batch_size: 4 (2 GPUs × 2 per_device)")
        # print("===========================")
        
        # 计算损失
        loss = None
        if labels is not None:
            # 从模型配置中获取是否使用全局batch损失
            use_global_batch = getattr(self, 'use_global_batch_loss', False)
            
            if use_global_batch:
                # 使用全局batch损失函数
                loss_fct = GlobalBatchContrastiveLoss(self.tau)
            else:
                # 使用原始本地batch损失函数
                loss_fct = UserItemContrastiveLoss(self.tau)
            
            loss = loss_fct(user_embeddings, item_embeddings)
            
            # 输出损失计算信息
            # print("=== Loss Computation Debug ===")
            # print(f"Loss value: {loss.item()}")
            # print(f"Loss shape: {loss.shape}")
            # print(f"Device: {loss.device}")
            # print("=============================")
        
        if not return_dict:
            output = (user_embeddings, item_embeddings) + user_transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=torch.cat([user_embeddings, item_embeddings], dim=0),  # 为了兼容性（最终表征）
            past_key_values=user_transformer_outputs.past_key_values,
            hidden_states=pre_adapter_pooled,  # 方案A：在此携带“前 adapter/投影”的拼接池化表示
            attentions=user_transformer_outputs.attentions,
        )
    
    def _get_pool_emb(self, hidden_states, attention_mask):
        """池化得到最终表示"""
        if self.pool_type == "last":
            # 取最后一个token
            pooled_emb = hidden_states[:, -1, :]
        elif self.pool_type == "avg":
            # 平均池化
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                pooled_emb = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
            else:
                pooled_emb = torch.mean(hidden_states, dim=1)
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")
        
        return pooled_emb


class UserItemContrastiveLoss(nn.Module):
    """
    用户-物品交互对的ICLL对比损失
    实现公式: L_icll = -sum_{<u,i> in B_r} log(exp(sim(e_u, e_i)/tau) / sum_{v in B} exp(sim(e_u + e_i, e_v)/tau))
    """
    
    def __init__(self, tau=1.0):
        super().__init__()
        self.temperature = tau
    
    def forward(self, user_embeddings, item_embeddings):
        """
        计算用户-物品对的ICLL对比损失
        Args:
            user_embeddings: (batch_size, hidden_size) - 用户嵌入
            item_embeddings: (batch_size, hidden_size) - 物品嵌入
        """
        batch_size = user_embeddings.shape[0]
        
        # 计算用户-物品相似度矩阵 (batch_size, batch_size)
        # sim(e_u, e_i) = e_u^T * e_i
        user_item_sim = (user_embeddings @ item_embeddings.T) / self.temperature
        
        # 计算用户+物品与所有实体的相似度
        # 将所有用户和物品嵌入拼接: [user_embeddings, item_embeddings]
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # (2*batch_size, hidden_size)
        
        # 计算用户+物品与所有实体的相似度
        # (e_u + e_i)^T * e_v for all v in B
        user_item_sum = user_embeddings + item_embeddings  # (batch_size, hidden_size)
        user_item_all_sim = (user_item_sum @ all_embeddings.T) / self.temperature  # (batch_size, 2*batch_size)
        
        # 计算损失
        # 对于每个交互对 <u,i>，正例是 sim(e_u, e_i)，负例是 sim(e_u + e_i, e_v) for all v in B
        loss = 0.0
        for i in range(batch_size):
            # 正例: sim(e_u, e_i)
            positive_score = user_item_sim[i, i]  # 对角线元素
            
            # 负例: sim(e_u + e_i, e_v) for all v in B
            negative_scores = user_item_all_sim[i, :]  # (2*batch_size,)
            
            # 计算log概率
            log_prob = positive_score - torch.logsumexp(negative_scores, dim=0)
            loss -= log_prob
        
        return loss / batch_size


class GlobalBatchContrastiveLoss(nn.Module):
    """
    全局batch对比损失函数
    收集所有GPU的数据，进行全局batch对比学习
    """
    
    def __init__(self, tau=1.0):
        super().__init__()
        self.temperature = tau
    
    def forward(self, user_embeddings, item_embeddings):
        """
        计算全局batch对比损失
        Args:
            user_embeddings: (batch_size, hidden_size) - 用户嵌入
            item_embeddings: (batch_size, hidden_size) - 物品嵌入
        """
        import torch.distributed as dist
        import torch.distributed.nn.functional as dist_nn_f
        
        # 收集所有GPU的embeddings（使用autograd-safe的all_gather）
        global_user_embeddings = dist_nn_f.all_gather(user_embeddings)  # 返回tuple
        global_item_embeddings = dist_nn_f.all_gather(item_embeddings)  # 返回tuple
        
        # 拼接tuple中的张量
        global_user_embeddings = torch.cat(global_user_embeddings, dim=0)  # (world_size * batch_size, hidden_size)
        global_item_embeddings = torch.cat(global_item_embeddings, dim=0)  # (world_size * batch_size, hidden_size)
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_batch_size = user_embeddings.shape[0]
        global_batch_size = global_user_embeddings.shape[0]
        
        # print(f"=== Global Batch Debug ===")
        # print(f"World size: {world_size}")
        # print(f"Local batch size: {local_batch_size}")
        # print(f"Global batch size: {global_batch_size}")
        # print(f"Global user embeddings shape: {global_user_embeddings.shape}")
        # print(f"Global item embeddings shape: {global_item_embeddings.shape}")
        # print("==========================")

        # 相比于第一版，多加了一个归一化操作
        global_user_embeddings = F.normalize(global_user_embeddings, dim=-1)
        global_item_embeddings = F.normalize(global_item_embeddings, dim=-1)
        # 使用全局batch计算损失，对原始版本进行了修改
        # 正例分数: e_u ⊙ e_i (逐元素相乘后求和)
        pos_score = (global_user_embeddings * global_item_embeddings).sum(dim=-1)  # (global_batch_size,)
        pos_score = torch.exp(pos_score / self.temperature)
        
        # 负例分数: e_u^T × e_v + e_i^T × e_v
        node_emb = torch.cat([global_user_embeddings, global_item_embeddings], dim=0)  # (2*global_batch_size, hidden_size)
        ttl_u_score = torch.matmul(global_user_embeddings, node_emb.transpose(0, 1))  # (global_batch_size, 2*global_batch_size)
        ttl_i_score = torch.matmul(global_item_embeddings, node_emb.transpose(0, 1))   # (global_batch_size, 2*global_batch_size)
        ttl_score = torch.exp((ttl_u_score + ttl_i_score) / self.temperature).sum(dim=1)  # (global_batch_size,)
        
        # 计算对比损失
        loss = -torch.log(pos_score / ttl_score).mean()
        
        return loss / global_batch_size
        
        # 使用全局batch计算损失
        # user_item_sim = (global_user_embeddings @ global_item_embeddings.T) / self.temperature
        # all_embeddings = torch.cat([global_user_embeddings, global_item_embeddings], dim=0)
        # user_item_sum = global_user_embeddings + global_item_embeddings
        # user_item_all_sim = (user_item_sum @ all_embeddings.T) / self.temperature
        
        # # 计算全局损失
        # loss = 0.0
        # for i in range(global_batch_size):
        #     positive_score = user_item_sim[i, i]
        #     negative_scores = user_item_all_sim[i, :]
        #     log_prob = positive_score - torch.logsumexp(negative_scores, dim=0)
        #     loss -= log_prob
        
        # return loss / global_batch_size


# 保留原有的LlamaRSEmb类以保持兼容性
class LlamaRSEmb(LlamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.model = LlamaModel(config)
        self.pool_type = kwargs.pop("pool_type")
        self.tau = kwargs.pop("tau")
        self.tau = nn.Parameter(torch.FloatTensor([3]))

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.model.embed_tokens


    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # get the output from LLM (bs, seq_len, hidden_size)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(input_ids.device)
            else:
                sequence_lengths = -1

        # transformer_outputs.hidden_states = hidden_states[torch.arange(batch_size, device=logits.device), sequence_lengths]   # output the hidden_states for feature-based KD
        transformer_outputs.hidden_states = self._get_pool_emb(hidden_states, sequence_lengths, attention_mask)

        # get pooled logits
        pooled_logits = transformer_outputs.hidden_states

        # split the pair of outputs
        batch_size = input_ids.size(0) // 2
        chosen_logits, rejected_logits = pooled_logits.split(batch_size, dim=0)

        loss = None
        if labels is not None:

            loss_fct = Contrastive_Loss(self.tau)
            loss = loss_fct(chosen_logits, rejected_logits)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    

    def _get_pool_emb(self, hidden_states, sequence_lengths, pooled_mask):
        """get the logits according to pool type"""
        if self.pool_type == "last":    # take out the last token as LLM embedding
            pooled_emb = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), 
                                       sequence_lengths]
        # average pooling all tokens as LLM embedding or average pooling attribute tokens as LLM embedding
        elif self.pool_type == "avg":   
            pooled_emb = torch.sum(hidden_states * pooled_mask.unsqueeze(-1), dim=1) / torch.sum(pooled_mask, dim=1).unsqueeze(-1) #sequence_lengths.unsqueeze(-1)

        return pooled_emb



class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        






