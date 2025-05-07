import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import KVmix 


from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

_CONFIG_FOR_DOC = "LlamaConfig"


def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
class LlamaAttention_KVmix(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, quant_bits: dict, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.k_bits = quant_bits[layer_idx]['k_bits']  # Quantized bits of Key
        self.v_bits = quant_bits[layer_idx]['v_bits']  # Quantized bits of Value
        self.k_residual_ratio = quant_bits[layer_idx]['k_residual_ratio']  # Key full precision ratio generated for each token
        self.v_residual_ratio = quant_bits[layer_idx]['v_residual_ratio']  # Value full precision ratio generated for each token
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.Vgroup_size = config.group_size 
        self.Kgroup_size = config.group_size
        if self.k_bits == 3:
            self.Kgroup_size = 11 
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_attention_weights(self, att_qkquant: Optional[torch.Tensor], att_qkfull: torch.Tensor) -> torch.Tensor:
        if att_qkquant is not None:
            att_qkquant = att_qkquant.to(torch.float32)
            att_qkfull = att_qkfull.to(torch.float32)
            attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1)
        else:
            attn_weights = att_qkfull.to(torch.float32)
        return attn_weights / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Current sequence length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
    
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Get cache data from past_key_value
            key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, \
            value_states_quant, value_states_full, value_scale, value_mn, _ = past_key_value

            # Splice the new key_states and value_states into the full-precision part
            key_states_full = key_states if key_states_full is None else torch.cat([key_states_full, key_states], dim=2)
            value_states_full = value_states if value_states_full is None else torch.cat([value_states_full, value_states], dim=2)

            # Calculate the length of the current full-precision part
            current_k_full_length = key_states_full.shape[-2]
            current_v_full_length = value_states_full.shape[-2]

            # Calculate the number of full-precision tokens that should be retained based on the current length and residual ratio
            k_full_length = int(self.k_residual_ratio * current_k_full_length)
            v_full_length = int(self.v_residual_ratio * current_v_full_length)

            # Handling full-precision and quantized parts for key
            if current_k_full_length > k_full_length and k_full_length >= 0:
                to_quantize_length = current_k_full_length - k_full_length
                # Make sure to_quantize_length is an integer multiple of group_size
                to_quantize_length = (to_quantize_length // self.Kgroup_size) * self.Kgroup_size
                if to_quantize_length > 0:
                    to_quantize = key_states_full[:, :, :to_quantize_length, :]
                    key_states_full = key_states_full[:, :, to_quantize_length:, :]

                    key_quant_new, key_scale_new, key_mn_new = KVmix.quantize_pack_last_dim_cuda(
                        to_quantize.transpose(2, 3).contiguous(), self.Kgroup_size, self.k_bits
                    )
                    key_states_quant_trans = key_quant_new if key_states_quant_trans is None else \
                        torch.cat([key_states_quant_trans, key_quant_new], dim=3)
                    key_scale_trans = key_scale_new if key_scale_trans is None else \
                        torch.cat([key_scale_trans, key_scale_new], dim=3)
                    key_mn_trans = key_mn_new if key_mn_trans is None else \
                        torch.cat([key_mn_trans, key_mn_new], dim=3)
                else:
                    # If to_quantize_length <= 0, do not quantize
                    key_states_quant_trans = key_states_quant_trans if key_states_quant_trans is not None else None
                    key_scale_trans = key_scale_trans if key_scale_trans is not None else None
                    key_mn_trans = key_mn_trans if key_mn_trans is not None else None
            else:
                # If quantization is not required
                key_states_quant_trans = key_states_quant_trans if key_states_quant_trans is not None else None
                key_scale_trans = key_scale_trans if key_scale_trans is not None else None
                key_mn_trans = key_mn_trans if key_mn_trans is not None else None

            # Handles full-precision and quantized parts for value
            if current_v_full_length > v_full_length and v_full_length >= 0:
                to_quantize_length = current_v_full_length - v_full_length
                to_quantize_length = (to_quantize_length // self.Vgroup_size) * self.Vgroup_size
                if to_quantize_length > 0:
                    to_quantize = value_states_full[:, :, :to_quantize_length, :]
                    value_states_full = value_states_full[:, :, to_quantize_length:, :]

                    value_quant_new, value_scale_new, value_mn_new = KVmix.quantize_pack_last_dim_cuda(
                        to_quantize.contiguous(), self.Vgroup_size, self.v_bits
                    )
                    value_states_quant = value_quant_new if value_states_quant is None else \
                        torch.cat([value_states_quant, value_quant_new], dim=2)
                    value_scale = value_scale_new if value_scale is None else \
                        torch.cat([value_scale, value_scale_new], dim=2)
                    value_mn = value_mn_new if value_mn is None else \
                        torch.cat([value_mn, value_mn_new], dim=2)
                else:
                    value_states_quant = value_states_quant if value_states_quant is not None else None
                    value_scale = value_scale if value_scale is not None else None
                    value_mn = value_mn if value_mn is not None else None
            else:
                value_states_quant = value_states_quant if value_states_quant is not None else None
                value_scale = value_scale if value_scale is not None else None
                value_mn = value_mn if value_mn is not None else None
            key_states_quant_trans_repeat = repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups)
            key_scale_trans_repeat = repeat_kv_quant(key_scale_trans, self.num_key_value_groups)
            key_mn_trans_repeat = repeat_kv_quant(key_mn_trans, self.num_key_value_groups)
            # Calculating attention weight
            att_qkquant = KVmix.quantized_gemm(
                self.Kgroup_size, query_states, key_states_quant_trans_repeat, key_scale_trans_repeat, key_mn_trans_repeat, self.k_bits
            ) if key_states_quant_trans is not None else None

            key_states_full_repeat = repeat_kv_quant(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))
            attn_weights = self._merge_attention_weights(att_qkquant, att_qkfull)

            # Apply mask and softmax
            if attention_mask is not None:
                if attention_mask.shape[-2:] != attn_weights.shape[-2:]:
                    raise ValueError(f"Attention mask shape {attention_mask.shape} does not match weights shape {attn_weights.shape}")
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.clamp(attn_weights, min=torch.finfo(attn_weights.dtype).min)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Calculating attention output
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv_quant(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat)
            else:
                value_full_length = value_states_full.shape[-2]
                value_states_quant_repeat = repeat_kv_quant(value_states_quant, self.num_key_value_groups)
                value_scale_repeat = repeat_kv_quant(value_scale, self.num_key_value_groups)
                value_mn_repeat = repeat_kv_quant(value_mn, self.num_key_value_groups)
                attn_output_quant = KVmix.quantized_gemm(
                    self.Vgroup_size, attn_weights[:, :, :, :-value_full_length], value_states_quant_repeat,
                    value_scale_repeat, value_mn_repeat, self.v_bits
                )
                value_states_full_repeat = repeat_kv_quant(value_states_full, self.num_key_value_groups)
                attn_output_full = torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full_repeat)
                attn_output = attn_output_quant + attn_output_full

        else:
            # No cache
            current_k_full_length = key_states.shape[-2]
            current_v_full_length = value_states.shape[-2]

            # Calculate the number of full-precision tokens that should be retained
            k_full_length = int(self.k_residual_ratio * current_k_full_length)
            v_full_length = int(self.v_residual_ratio * current_v_full_length)

            # Split key
            if k_full_length < current_k_full_length and k_full_length >= 0:
                to_quantize_length = current_k_full_length - k_full_length
                to_quantize_length = (to_quantize_length // self.Kgroup_size) * self.Kgroup_size
                if to_quantize_length > 0:
                    key_states_quant = key_states[:, :, :to_quantize_length, :].contiguous()
                    key_states_full = key_states[:, :, to_quantize_length:, :].contiguous()
                    key_states_quant_trans, key_scale_trans, key_mn_trans = KVmix.quantize_pack_last_dim_cuda(
                        key_states_quant.transpose(2, 3).contiguous(), self.Kgroup_size, self.k_bits
                    )
                else:
                    key_states_quant_trans, key_scale_trans, key_mn_trans = None, None, None
                    key_states_full = key_states
            else:
                key_states_quant_trans, key_scale_trans, key_mn_trans = None, None, None
                key_states_full = key_states

            # Split value
            if v_full_length < current_v_full_length and v_full_length >= 0:
                to_quantize_length = current_v_full_length - v_full_length
                to_quantize_length = (to_quantize_length // self.Vgroup_size) * self.Vgroup_size
                if to_quantize_length > 0:
                    value_states_quant = value_states[:, :, :to_quantize_length, :].contiguous()
                    value_states_full = value_states[:, :, to_quantize_length:, :].contiguous()
                    value_states_quant, value_scale, value_mn = KVmix.quantize_pack_last_dim_cuda(
                        value_states_quant.contiguous(), self.Vgroup_size, self.v_bits
                    )
                else:
                    value_states_quant, value_scale, value_mn = None, None, None
                    value_states_full = value_states
            else:
                value_states_quant, value_scale, value_mn = None, None, None
                value_states_full = value_states

            key_states_repeat = repeat_kv_quant(key_states, self.num_key_value_groups)
            value_states_repeat = repeat_kv_quant(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states_repeat.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                if attention_mask.shape[-2:] != attn_weights.shape[-2:]:
                    raise ValueError(f"Attention mask shape {attention_mask.shape} does not match weights shape {attn_weights.shape}")
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.clamp(attn_weights, min=torch.finfo(attn_weights.dtype).min)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_repeat)

        # update KV Cache
        if use_cache:
            past_key_value = (
                key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
                value_states_quant, value_states_full, value_scale, value_mn,kv_seq_len
            )
        else:
            past_key_value = None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class LlamaDecoderLayer_KVmix(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, quant_bits: Optional[dict] = None, **kwargs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # if quant_bits and layer_idx not in quant_bits:
        #     quant_bits[layer_idx] = {
        #         'k_bits': getattr(config, 'k_bits', 4),
        #         'v_bits': getattr(config, 'v_bits', 4),
        #         'k_residual_ratio': 0.2,
        #         'v_residual_ratio': 0.2
        #     }
        
        self.self_attn = (
            LlamaAttention_KVmix(config, quant_bits, layer_idx)
            if not getattr(config, "use_flash", False)
            else LlamaFlashAttention_KVmix(config, quant_bits, layer_idx)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_KVmix(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, quant_bits: Optional[dict] = None, **kwargs):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_KVmix(config, i, quant_bits) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        print("quant_bits inference start",quant_bits)
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_KVmix(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, quant_bits: Optional[dict] = None, **kwargs):
        super().__init__(config)
        self.model = LlamaModel_KVmix(config,quant_bits)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
  
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None
        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
