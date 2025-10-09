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
    Soft-prompt learning model for user–item interaction pairs.
    Supports user/item soft-prompt tokens, learnable weights, and an adapter.
    """
    main_input_name = "user_input_ids"  # Tell Trainer the primary input argument name

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 1) Obtain hidden_size first; subsequent layers depend on it
        self.hidden_size = config.hidden_size

        # 2) Base Llama model
        self.model = LlamaModel(config)

        # 3) Configurable arguments
        self.pool_type = kwargs.pop("pool_type", "avg")
        tau_init = kwargs.pop("tau", 1.0)
        self.tau = nn.Parameter(torch.tensor([tau_init], dtype=torch.float32))

        # 4) Soft-prompt lengths & final projection dimension
        self.user_prompt_length = kwargs.pop("user_prompt_length", 10)
        self.item_prompt_length = kwargs.pop("item_prompt_length", 10)
        self.emb_dim = kwargs.pop("emb_dim", 1536)  # If you prefer 1024, change the default to 1024

        # 5) Attention-related layers (safe to use now that self.hidden_size is ready)
        self.user_prompt_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_prompt_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Use [user_pool ; item_pool] as the query to form user/item queries respectively
        self.user_query = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.item_query = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.attn_scale = self.hidden_size ** 0.5  # sqrt(d)

        # 6) Soft-prompt parameters (trainable)
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

        # 7) Adapter layer (optional)
        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )

        # 8) Final projection (hidden_size -> emb_dim)
        self.final_projection = nn.Linear(self.hidden_size, self.emb_dim)

        # 9) Others
        self.use_global_batch_loss = kwargs.pop("use_global_batch_loss", False)

        # 10) Initialize (LlamaPreTrainedModel handles linear layer weight init)
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def _mean_pool(self, embs, mask):
        # embs: [B, L, d], mask: [B, L] or None
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
        Use query to perform dot-product attention over prompts, obtaining per-sample
        weights over P prompt tokens; then scale each prompt token by its weight.
        Returns: [B, P, d], which can be concatenated with the original embeddings along the sequence dimension.
        """
        # K: [P, d]
        K = key_proj(prompts)                          

        # Attention logits: [B, P]
        # (q @ K^T) / sqrt(d)
        attn_logits = torch.matmul(query, K.t()) / self.attn_scale

        # Optional: treat learnable prior as a bias so the model can keep prior preference on top of attention
        if prior_weights is not None:
            # prior: [P] -> add log_softmax prior to logits
            prior_bias = torch.log_softmax(prior_weights, dim=0)  # [P]
            attn_logits = attn_logits + prior_bias.unsqueeze(0)   # [B, P]

        # Weights: [B, P]
        attn = torch.softmax(attn_logits, dim=-1)

        # Scale each prompt token by its weight: expand to [B, P, 1] and multiply the original prompts (not K)
        weighted_prompts = attn.unsqueeze(-1) * prompts.unsqueeze(0)  # [B, P, d]
        return weighted_prompts
    
    def _process_user_item_pair(self, user_input_ids, item_input_ids,
                                user_attention_mask=None, item_attention_mask=None):
        """
        Process a user–item pair: first obtain user/item sentence vectors (mean pool),
        then use them as queries to attend to their respective prompts, and finally
        concatenate the weighted prompts with the original token embeddings.
        """
        batch_size = user_input_ids.shape[0]

        # 1) Original token embeddings
        user_embeddings = self.model.embed_tokens(user_input_ids)  # [B, Lu, d]
        item_embeddings = self.model.embed_tokens(item_input_ids)  # [B, Li, d]

        # 2) Sentence vectors (could be last hidden state + pooling; here we use embedding mean-pool)
        u_pool = self._mean_pool(user_embeddings, user_attention_mask)  # [B, d]
        i_pool = self._mean_pool(item_embeddings, item_attention_mask)  # [B, d]

        # 3) Build queries
        ui = torch.cat([u_pool, i_pool], dim=-1)  # [B, 2d]
        iu = torch.cat([i_pool, u_pool], dim=-1)  # [B, 2d]
        q_u = self.user_query(ui)                 # [B, d] — query to weigh user prompts
        q_i = self.item_query(iu)                 # [B, d] — query to weigh item prompts

        # 4) Attention-weighted prompts for each sample (keep per-token independence)
        user_weighted_prompts = self._attend_prompts(
            self.user_prompts, q_u, self.user_prompt_key, prior_weights=self.user_prompt_weights
        )  # [B, Pu, d]

        item_weighted_prompts = self._attend_prompts(
            self.item_prompts, q_i, self.item_prompt_key, prior_weights=self.item_prompt_weights
        )  # [B, Pi, d]

        # 5) Concatenate with original embeddings
        user_enhanced_embeddings = torch.cat([user_weighted_prompts, user_embeddings], dim=1)  # [B, Pu+Lu, d]
        item_enhanced_embeddings = torch.cat([item_weighted_prompts, item_embeddings], dim=1)  # [B, Pi+Li, d]

        # 6) Extend attention_mask accordingly (set prepended prompt positions to 1)
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
        input_ids: torch.LongTensor = None,  # standard argument for compatibility
        attention_mask: Optional[torch.Tensor] = None,
        user_attention_mask: Optional[torch.Tensor] = None,  # user attention mask
        item_attention_mask: Optional[torch.Tensor] = None,  # item attention mask
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
        Forward pass.
        Args:
            user_input_ids: token ids for user text
            item_input_ids: token ids for item text
            input_ids: standard argument (compatibility only; do not use for this model)
            attention_mask: attention mask
            user_attention_mask: user attention mask
            item_attention_mask: item attention mask
            labels: labels (used to compute loss)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Handle standard argument name usage
        if input_ids is not None and user_input_ids is None and item_input_ids is None:
            raise ValueError("Please use user_input_ids and item_input_ids instead of input_ids")

        # Ensure required arguments exist
        if user_input_ids is None or item_input_ids is None:
            raise ValueError("Both user_input_ids and item_input_ids are required")

        # Handle attention masks
        if user_attention_mask is None:
            user_attention_mask = attention_mask
        if item_attention_mask is None:
            item_attention_mask = attention_mask
        
        # Process the user–item pair and prepend soft prompts
        user_enhanced_embeddings, item_enhanced_embeddings, user_enhanced_mask, item_enhanced_mask = \
            self._process_user_item_pair(user_input_ids, item_input_ids, user_attention_mask, item_attention_mask)
        
        # Run user path through Llama
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
        
        # Run item path through Llama
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
        
        # Pool user/item representations (before adapter/projection)
        user_pooled = self._get_pool_emb(user_hidden_states, user_enhanced_mask)
        item_pooled = self._get_pool_emb(item_hidden_states, item_enhanced_mask)

        # Keep a copy of pre-adapter/projection pooled representations for anchoring scheme A
        pre_adapter_pooled = torch.cat([user_pooled, item_pooled], dim=0)
        
        # In fp16 training, pooling may cast back to float32; align with adapter weight dtype (fp16 or fp32)
        target_dtype = self.adapter[0].weight.dtype
        if user_pooled.dtype != target_dtype:
            user_pooled = user_pooled.to(target_dtype)
        if item_pooled.dtype != target_dtype:
            item_pooled = item_pooled.to(target_dtype)
        
        # Through adapter
        user_embeddings = self.adapter(user_pooled)
        item_embeddings = self.adapter(item_pooled)
        
        # Final projection, e.g., 4096 -> 1024
        user_embeddings = self.final_projection(user_embeddings)
        item_embeddings = self.final_projection(item_embeddings)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Determine whether to use global-batch loss from model config
            use_global_batch = getattr(self, 'use_global_batch_loss', False)
            
            if use_global_batch:
                # Global-batch contrastive loss
                loss_fct = GlobalBatchContrastiveLoss(self.tau)
            else:
                # Local-batch contrastive loss
                loss_fct = UserItemContrastiveLoss(self.tau)
            
            loss = loss_fct(user_embeddings, item_embeddings)
        
        if not return_dict:
            output = (user_embeddings, item_embeddings) + user_transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=torch.cat([user_embeddings, item_embeddings], dim=0),  # compatibility (final representations)
            past_key_values=user_transformer_outputs.past_key_values,
            hidden_states=pre_adapter_pooled,  # scheme A: carry concatenated pooled reps before adapter/projection
            attentions=user_transformer_outputs.attentions,
        )
    
    def _get_pool_emb(self, hidden_states, attention_mask):
        """Pool the final representation."""
        if self.pool_type == "last":
            # use last token
            pooled_emb = hidden_states[:, -1, :]
        elif self.pool_type == "avg":
            # mean pooling
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
    Interaction-level contrastive loss (ICLL) for user–item pairs.
    Formula: L_icll = -sum_{<u,i> in B_r} log( exp(sim(e_u, e_i)/tau) / sum_{v in B} exp(sim(e_u + e_i, e_v)/tau) )
    """
    def __init__(self, tau=1.0):
        super().__init__()
        self.temperature = tau
    
    def forward(self, user_embeddings, item_embeddings):
        """
        Compute ICLL for user–item pairs.
        Args:
            user_embeddings: (batch_size, hidden_size) — user embeddings
            item_embeddings: (batch_size, hidden_size) — item embeddings
        """
        batch_size = user_embeddings.shape[0]
        
        # User–item similarity matrix (batch_size, batch_size)
        # sim(e_u, e_i) = e_u^T * e_i
        user_item_sim = (user_embeddings @ item_embeddings.T) / self.temperature
        
        # Concatenate all user/item embeddings
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # (2*batch_size, hidden_size)
        
        # Similarity between (e_u + e_i) and all entities in the batch
        user_item_sum = user_embeddings + item_embeddings  # (batch_size, hidden_size)
        user_item_all_sim = (user_item_sum @ all_embeddings.T) / self.temperature  # (batch_size, 2*batch_size)
        
        # For each pair <u,i>, positive is sim(e_u, e_i); negatives are sim(e_u + e_i, e_v) for all v in batch
        loss = 0.0
        for i in range(batch_size):
            positive_score = user_item_sim[i, i]       # diagonal
            negative_scores = user_item_all_sim[i, :]  # (2*batch_size,)
            log_prob = positive_score - torch.logsumexp(negative_scores, dim=0)
            loss -= log_prob
        
        return loss / batch_size


class GlobalBatchContrastiveLoss(nn.Module):
    """
    Global-batch contrastive loss.
    Gather embeddings across all GPUs to perform contrastive learning globally.
    """
    def __init__(self, tau=1.0):
        super().__init__()
        self.temperature = tau
    
    def forward(self, user_embeddings, item_embeddings):
        """
        Compute global-batch contrastive loss.
        Args:
            user_embeddings: (batch_size, hidden_size) — user embeddings
            item_embeddings: (batch_size, hidden_size) — item embeddings
        """
        import torch.distributed as dist
        import torch.distributed.nn.functional as dist_nn_f
        
        # Autograd-safe all_gather from all GPUs
        global_user_embeddings = dist_nn_f.all_gather(user_embeddings)  # tuple
        global_item_embeddings = dist_nn_f.all_gather(item_embeddings)  # tuple
        
        # Concatenate tensors from the tuple
        global_user_embeddings = torch.cat(global_user_embeddings, dim=0)  # (world_size * batch_size, hidden_size)
        global_item_embeddings = torch.cat(global_item_embeddings, dim=0)  # (world_size * batch_size, hidden_size)
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_batch_size = user_embeddings.shape[0]
        global_batch_size = global_user_embeddings.shape[0]

        # Normalize before similarity
        global_user_embeddings = F.normalize(global_user_embeddings, dim=-1)
        global_item_embeddings = F.normalize(global_item_embeddings, dim=-1)

        # Positive scores: element-wise product followed by sum
        pos_score = (global_user_embeddings * global_item_embeddings).sum(dim=-1)  # (global_batch_size,)
        pos_score = torch.exp(pos_score / self.temperature)
        
        # Negative scores: e_u^T * e_v + e_i^T * e_v
        node_emb = torch.cat([global_user_embeddings, global_item_embeddings], dim=0)  # (2*global_batch_size, H)
        ttl_u_score = torch.matmul(global_user_embeddings, node_emb.transpose(0, 1))   # (G, 2G)
        ttl_i_score = torch.matmul(global_item_embeddings, node_emb.transpose(0, 1))   # (G, 2G)
        ttl_score = torch.exp((ttl_u_score + ttl_i_score) / self.temperature).sum(dim=1)  # (G,)
        
        # Contrastive loss
        loss = -torch.log(pos_score / ttl_score).mean()
        
        return loss / global_batch_size


# Keep the original LlamaRSEmb for compatibility
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
        hidden_states = transformer_outputs[0]  # LLM output (bs, seq_len, hidden_size)

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

        # Save pooled hidden states for feature-based KD
        transformer_outputs.hidden_states = self._get_pool_emb(hidden_states, sequence_lengths, attention_mask)

        # Pooled logits
        pooled_logits = transformer_outputs.hidden_states

        # Split the pair of outputs
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
        """Get logits according to the selected pooling type."""
        if self.pool_type == "last":  # use last token as LLM embedding
            pooled_emb = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), 
                                       sequence_lengths]
        # avg-pool all tokens or attribute tokens as LLM embedding
        elif self.pool_type == "avg":
            pooled_emb = torch.sum(hidden_states * pooled_mask.unsqueeze(-1), dim=1) / torch.sum(pooled_mask, dim=1).unsqueeze(-1)

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
        loss = (Y_loss + X_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
    
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()