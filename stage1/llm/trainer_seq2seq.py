from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator

def is_deepspeed_zero3_enabled():
    """Check if DeepSpeed ZeRO-3 is enabled."""
    import os
    return os.environ.get("DEEPSPEED_ZERO_STAGE", None) == "3"
# from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback

from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.training_args import TrainingArguments
from transformers.utils import logging

from transformers import Trainer
from llm.llama import UserItemContrastiveLoss  # Added: import contrastive loss function

logger = logging.get_logger(__name__)


class Seq2SeqTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs


        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "position_ids" in inputs:
            gen_kwargs["position_ids"] = inputs.get("position_ids", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs["input_ids"] = generation_inputs
        generated_tokens = self.model.generate(**gen_kwargs)
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor



class MedRecTrainer(Trainer):

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):
        outputs = model(**inputs)  # LoRA ON, normal forward pass, includes contrastive loss
        loss = outputs.loss

        enable_anchor = getattr(model, "enable_lora_anchor", False)
        anchor_lambda = getattr(model, "lora_anchor_weight", 0.0)

        if enable_anchor and anchor_lambda > 0.0:
            pre_adapter_concat = getattr(outputs, "hidden_states", None)
            if pre_adapter_concat is not None:
                B = pre_adapter_concat.shape[0] // 2
                eu_pre_lora = pre_adapter_concat[:B]

                eu_pre_llm = None
                try:
                    if hasattr(model, "disable_adapter"):
                        with torch.no_grad():
                            with model.disable_adapter():
                                anchor_out = model(**{**inputs, "labels": None})
                                if getattr(anchor_out, "hidden_states", None) is not None:
                                    eu_pre_llm = anchor_out.hidden_states[:B].detach()
                    elif hasattr(model, "base_model") and hasattr(model.base_model, "disable_adapter"):
                        with torch.no_grad():
                            with model.base_model.disable_adapter():
                                anchor_out = model(**{**inputs, "labels": None})
                                if getattr(anchor_out, "hidden_states", None) is not None:
                                    eu_pre_llm = anchor_out.hidden_states[:B].detach()
                except Exception:
                    eu_pre_llm = None
                
                if eu_pre_llm is not None:
                    ei_pre_lora = pre_adapter_concat[B:]
                    if 'anchor_out' in locals() and getattr(anchor_out, 'hidden_states', None) is not None:
                        ei_pre_llm = anchor_out.hidden_states[B:].detach()
                    else:
                        ei_pre_llm = None

                    reg_u = torch.mean((eu_pre_lora - eu_pre_llm) ** 2)
                    if ei_pre_llm is not None:
                        reg_i = torch.mean((ei_pre_lora - ei_pre_llm) ** 2)
                        reg = 0.5 * (reg_u + reg_i)
                    else:
                        reg = reg_u
                    loss = loss + anchor_lambda * reg
                    step = getattr(self.state, "global_step", 0)
                    log_every = getattr(self.args, "logging_steps", 100)
                    if log_every and step > 0 and (step % log_every == 0) and self.is_world_process_zero():
                        try:
                            msg = f"[anchor] step={step} reg={reg.item():.6f} reg_u={reg_u.item():.6f}"
                            if ei_pre_llm is not None:
                                msg += f" reg_i={reg_i.item():.6f}"
                            msg += f" lambda={anchor_lambda}"
                            logger.info(msg)
                        except Exception:
                            pass

        enable_div = getattr(model, "enable_prompt_diversity", False)
        div_lambda = getattr(model, "prompt_div_weight", 0.0)
        if enable_div and div_lambda > 0.0:
            def _get_attr_chain(m, name: str):
                if hasattr(m, name):
                    return getattr(m, name)
                if hasattr(m, "base_model"):
                    base = m.base_model
                    if hasattr(base, name):
                        return getattr(base, name)
                    if hasattr(base, "model") and hasattr(base.model, name):
                        return getattr(base.model, name)
                return None

            user_prompts = _get_attr_chain(model, "user_prompts")
            item_prompts = _get_attr_chain(model, "item_prompts")

            div_terms = []
            def diversity_loss(P: torch.Tensor):
                # P: [L, H]
                if P is None or not isinstance(P, torch.Tensor) or P.dim() != 2 or P.size(0) < 2:
                    return None

                denom = torch.norm(P, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                Pn = P / denom
                G = torch.matmul(Pn, Pn.T)  # [L, L]
                I = torch.eye(P.size(0), device=G.device, dtype=G.dtype)
                off = G - I
                return (off.pow(2).sum()) / (P.size(0) * (P.size(0) - 1))

            for P in (user_prompts, item_prompts):
                dl = diversity_loss(P)
                if dl is not None:
                    div_terms.append(dl)

            if div_terms:
                div_loss = torch.stack(div_terms).mean()
                loss = loss + div_lambda * div_loss

                step = getattr(self.state, "global_step", 0)
                log_every = getattr(self.args, "logging_steps", 100)
                if log_every and step > 0 and (step % log_every == 0) and self.is_world_process_zero():
                    try:
                        logger.info(f"[prompt_div] step={step} div_loss={div_loss.item():.6f} lambda={div_lambda}")
                    except Exception:
                        pass
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], 
                        prediction_loss_only: bool, ignore_keys: List[str]):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = "labels" in inputs

        # Prepare the inputs
        inputs = self._prepare_inputs(inputs)
        
        model_inputs = {}
        
        if "user_attention_mask" in inputs:
            model_inputs["user_attention_mask"] = inputs["user_attention_mask"]
        elif "attention_mask" in inputs:
            model_inputs["user_attention_mask"] = inputs["attention_mask"]
            
        if "item_attention_mask" in inputs:
            model_inputs["item_attention_mask"] = inputs["item_attention_mask"]
        elif "attention_mask" in inputs:
            model_inputs["item_attention_mask"] = inputs["attention_mask"]
            
        if "position_ids" in inputs:
            model_inputs["position_ids"] = inputs["position_ids"]
            
        model_inputs["user_input_ids"] = inputs["user_input_ids"]
        model_inputs["item_input_ids"] = inputs["item_input_ids"]

        with torch.no_grad():
            outputs = model(**model_inputs)
        
        loss = None
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        batch_size = logits.shape[0] // 2
        user_embeddings = logits[:batch_size]
        item_embeddings = logits[batch_size:]
        
        combined_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        try:
            contrastive_loss_fct = UserItemContrastiveLoss(self.model.tau if hasattr(self.model, "tau") else 1.0)
            loss = contrastive_loss_fct(user_embeddings, item_embeddings).detach()
        except Exception as e:
            logger.warning(f"Contrastive loss computation failed in prediction_step: {e}")

        return (loss, logits, combined_embeddings)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        import os
        import torch
        import json
        
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting to save lightweight parameters to {output_dir}")
        
        if hasattr(self.model, "save_pretrained") and hasattr(self.model, "peft_config"):
            self.model.save_pretrained(output_dir)
            logger.info("✅ Saved LoRA parameters to adapter_config.json and adapter_model.bin")
        
        prompt_state_dict = {}
        adapter_state_dict = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(key in name for key in ["user_prompts", "item_prompts", "user_prompt_weights", "item_prompt_weights"]):
                    prompt_state_dict[name] = param.cpu().clone()
                elif any(key in name for key in ["adapter", "final_projection", "tau"]):
                    adapter_state_dict[name] = param.cpu().clone()
        
        # 3. Save soft-prompt parameters
        if prompt_state_dict:
            torch.save(prompt_state_dict, os.path.join(output_dir, "prompt_params.bin"))
            logger.info(f"✅ Saved soft-prompt parameters ({len(prompt_state_dict)}): {list(prompt_state_dict.keys())}")
        
        # 4. Save model adapter parameters
        if adapter_state_dict:
            torch.save(adapter_state_dict, os.path.join(output_dir, "model_adapter_params.bin"))
            logger.info(f"✅ Saved model adapter parameters ({len(adapter_state_dict)}): {list(adapter_state_dict.keys())}")
        
        # 5. Save training config
        if hasattr(self.model, 'config'):
            training_config = {
                "model_choice": "useritem_prompt",
                "user_prompt_length": getattr(self.model, 'user_prompt_length', 10),
                "item_prompt_length": getattr(self.model, 'item_prompt_length', 10),
                "pool_type": getattr(self.model, 'pool_type', 'avg'),
                "tau": float(getattr(self.model, 'tau', 1.0)),
                "use_global_batch_loss": getattr(self.model, 'use_global_batch_loss', False),
                "hidden_size": self.model.config.hidden_size,
            }
            
            with open(os.path.join(output_dir, "training_config.json"), "w") as f:
                json.dump(training_config, f, indent=2)
            logger.info("✅ Saved training config to training_config.json")
        
        # 6. Save tokenizer (lightweight)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info("✅ Saved tokenizer config")
        
        logger.info("=== Lightweight parameter saving completed ===")
        logger.info(f"Total files saved: LoRA params + {len(prompt_state_dict)} soft-prompt params + {len(adapter_state_dict)} adapter params")


        logger.info("=== Lightweight parameter saving completed ===")
        logger.info(f"Total files saved: LoRA params + {len(prompt_state_dict)} soft-prompt params + {len(adapter_state_dict)} adapter params")


        logger.info("=== Lightweight parameter saving completed ===")
        logger.info(f"Total files saved: LoRA params + {len(prompt_state_dict)} soft-prompt params + {len(adapter_state_dict)} adapter params")

    def _compute_contrastive_loss(self, eu_pre_lora, eu_pre_llm, ei_pre_lora, ei_pre_llm, temperature=1.0):
        """
        Compute contrastive anchor loss (as a replacement for MSE loss).

        Contrastive strategy:
        1) Positive: the same sample's representation under LoRA vs. the original LLM should be closer.
        2) Negative: different samples' representations should be pushed apart.

        Implementation:
        - Concatenate user and item reps for LoRA:   [eu_pre_lora, ei_pre_lora]
        - Concatenate user and item reps for LLM:    [eu_pre_llm,  ei_pre_llm]
        - Compute similarity matrix; diagonal = positives, off-diagonals = negatives
        - Use cross-entropy loss so positive similarities are higher and negatives are lower.

        Args:
            eu_pre_lora: user reps under LoRA          [B, hidden_size]
            eu_pre_llm:  user reps under original LLM  [B, hidden_size]
            ei_pre_lora: item reps under LoRA          [B, hidden_size]
            ei_pre_llm:  item reps under original LLM  [B, hidden_size]
            temperature: temperature for contrastive strength

        Returns:
            contrastive_loss: contrastive learning loss
        """
        import torch.nn.functional as F
        
        try:

            lora_embeddings = torch.cat([eu_pre_lora, ei_pre_lora], dim=0)  # [2B, hidden_size]
            llm_embeddings = torch.cat([eu_pre_llm, ei_pre_llm], dim=0)    # [2B, hidden_size]
            
            lora_embeddings = F.normalize(lora_embeddings, dim=-1)
            llm_embeddings = F.normalize(llm_embeddings, dim=-1)
            
            similarity_matrix = torch.matmul(lora_embeddings, llm_embeddings.T) / temperature
            
            batch_size = eu_pre_lora.shape[0]
            labels = torch.arange(2 * batch_size, device=similarity_matrix.device)
            
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)
            
            return contrastive_loss
            
        except Exception as e:
            print(f"Contrastive loss computation failed: {e}")
            return None