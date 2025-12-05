# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch RWKV07BMoE model.
base code from SmerkyG @ recursal.ai, featherless.ai
hxa07B implementation RWKV07B + NoPE Hybrid Attention + Mixture of Experts

"""

import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache#, DynamicCache, CacheLayerMixin
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask#, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
# from transformers.modeling_layers import (
#     GenericForQuestionAnswering,
#     GenericForSequenceClassification,
#     GenericForTokenClassification,
#     GradientCheckpointingLayer,
# )
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from transformers.processing_utils import Unpack
#from transformers.utils import TransformersKwargs#, auto_docstring, can_return_tuple
# from transformers.utils.generic import check_model_inputs

from .configuration_qwen3vlmoetext import RWKV07BMoEConfig

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention,Qwen3MoeSparseMoeBlock,Qwen3MoeMLP,Qwen3MoeDecoderLayer,Qwen3MoeRMSNorm

from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel

class RWKV07BState():
    def __init__(self) -> None:
        #super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.layer_kv_states: List[torch.Tensor] = []
        self.layer_shift_states:  List[torch.Tensor] = []
        self.cumulative_scores: List[torch.Tensor] = []
        self.sin: List[torch.Tensor] = []
        self.cos: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.layer_kv_states)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Linear Attention variants do not have a maximum length
        return new_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError('Cannot reorder Linear Attention state')

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    def crop(self, max_length: int):
        # can't implement this for linear attention variants
        return

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset
    
    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        return True #all(layer.is_compileable for layer in self.layers)
        
    @torch.no_grad
    def update(
        self,
        kv_state: torch.Tensor,
        shift_state: torch.Tensor,
        layer_idx: int,
        token_count: int = 0,
        is_attention_layer: bool = True,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:        
        # Update the number of seen tokens
        if layer_idx == 0:
            if is_attention_layer:
                token_count = kv_state.size(-2)
            self._seen_tokens += token_count

        #print(f'self._seen_tokens = {self._seen_tokens} layer_idx = {layer_idx} is_attention_layer = {is_attention_layer} kv_state.size(-2) = {kv_state.size(-2)}')

        # Update the cache
        if kv_state is not None:
            # There may be skipped layers, fill them with empty lists
            if layer_idx >= len(self.layer_kv_states):
                for _ in range(len(self.layer_kv_states), layer_idx):
                    if is_attention_layer:
                        self.layer_kv_states.append(torch.tensor([], dtype=kv_state.dtype, device=kv_state.device)) # acts as key_cache
                        self.layer_shift_states.append(torch.tensor([], dtype=shift_state.dtype, device=shift_state.device)) # acts as value_cache
                    else:
                        self.layer_kv_states.append(torch.zeros_like(kv_state).requires_grad_(False))
                        self.layer_shift_states.append(torch.zeros_like(shift_state).requires_grad_(False))
                self.layer_kv_states.append(kv_state) # acts as key_cache
                self.layer_shift_states.append(shift_state) # acts as value_cache
            else:
                if is_attention_layer:
                    self.layer_kv_states[layer_idx] = torch.cat([self.layer_kv_states[layer_idx], kv_state], dim=-2) # acts as key_cache
                    self.layer_shift_states[layer_idx] = torch.cat([self.layer_shift_states[layer_idx], shift_state], dim=-2) # acts as value_cache
                else:
                    self.layer_kv_states[layer_idx].copy_(kv_state)
                    self.layer_shift_states[layer_idx].copy_(shift_state)

        return self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx]

# try:
#     from fla.ops.rwkv7.chunk import chunk_rwkv7
#     from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
# except ImportError:
#     print("Required module is not installed. Please install it using the following commands:")
#     print("pip install --no-use-pep517 flash-linear-attention")
#     print("Additionally, ensure you have at least version 2.2.0 of Triton installed:")
#     print("pip install triton>=2.2.0")

# def is_layer_attention(config, layer_id):
#     return layer_id >= config.first_attention_layer and layer_id < config.first_post_attention_layer and  (layer_id > min(config.num_hidden_layers, config.last_striping_layer) or (min(config.num_hidden_layers-1, config.last_striping_layer) - layer_id) % config.attention_striping == 0)

def is_layer_attention(config, layer_id):
    return layer_id in config.transformer_layers

def repeat_kv_rwkv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads along the head dimension (GQA).
    Input:  (B, T, H_kv, D)
    Output: (B, T, H_kv * n_rep, D)
    """
    B, T, H_kv, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Expand head dim
    hidden_states = hidden_states[:, :, :, None, :]  # (B, T, H_kv, 1, D)
    hidden_states = hidden_states.expand(B, T, H_kv, n_rep, D)  # (B, T, H_kv, n_rep, D)
    return hidden_states.reshape(B, T, H_kv * n_rep, D).contiguous()

def T5RMSNorm(hidden_states,weight,variance_epsilon:float=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)

def compute_qwen3_rope_cache(seq_len, rotary_dim, device, dtype, rope_theta):
            half_dim = rotary_dim // 2
            freq_seq = torch.arange(half_dim, dtype=dtype, device=device)
            inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
            positions = torch.arange(seq_len, dtype=dtype, device=device)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos.unsqueeze(0), sin.unsqueeze(0), inv_freq

def compute_qwen3_mrope_cache_text_only(
    seq_len: int,
    rotary_dim: int,
    device,
    dtype=torch.float32,
    rope_theta: float = 5000000,
    mrope_section=(24, 20, 20),  # Qwen3VL のデフォルト想定
):
    """
    Qwen3VL の text-only MRoPE と互換な cos/sin キャッシュを作る版。
    戻り値の cos/sin shape は (1, seq_len, rotary_dim) で、
    既存の apply_rotary_pos_emb からそのまま使える想定。
    """
    half_dim = rotary_dim // 2

    # 1D RoPE と同じ inv_freq
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))

    # positions: 0..T-1
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # (T,)

    # text-only なので T/H/W すべて同じ positions を使う: (3, 1, T)
    position_ids = positions.view(1, 1, seq_len).expand(3, 1, -1)

    # (3, 1, half_dim, 1) と (3, 1, 1, T) から freqs: (3, 1, T, half_dim)
    inv_freq_expanded = inv_freq.view(1, 1, half_dim, 1).expand(3, 1, half_dim, 1)
    pos_expanded = position_ids.view(3, 1, 1, seq_len)
    freqs = torch.matmul(inv_freq_expanded, pos_expanded).transpose(2, 3)  # (3, 1, T, half_dim)

    # --- Qwen3VL の apply_interleaved_mrope 相当 ---
    # freqs[0]: T 軸用をベースにして、H/W 軸の一部をインターリーブ
    freqs_t = freqs[0]  # (1, T, half_dim)

    # dim=1,2 が H,W 軸
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3          # 例: 20 * 3 = 60
        end = min(length, half_dim)              # 安全のため half_dim を超えないように
        idx = slice(offset, end, 3)              # 1,4,7,... / 2,5,8,... みたいなインターリーブ位置
        freqs_t[..., idx] = freqs[dim, ..., idx]

    # 最後に [freqs_t, freqs_t] を結合して rotary_dim にする
    emb = torch.cat([freqs_t, freqs_t], dim=-1)  # (1, T, rotary_dim)

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin, inv_freq.to(dtype)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: RWKV07BMoEConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rms_norm(hidden_states, eps = 1e-6):
    #print('ugyuugyu')
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)

def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    #inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float).to(device) / dim))

    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((angles, angles), dim=-1)
    return torch.stack([emb.cos(), emb.sin()], dim=0)
    #return torch.polar(torch.ones_like(angles), angles)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(x, cos, sin, unsqueeze_dim=1):
    return (x * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(x) * sin.unsqueeze(unsqueeze_dim))

from typing import Callable, Optional, Tuple, Union
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0) # IMPORTANT FOR BATCHED INFERENCE IN LM EVAL!
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, create_mask
from functools import lru_cache

block_mask = None



def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.float()
    #attn_weight = stable_softmax(attn_weight, dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(attn_weight.isnan(), 0) # IMPORTANT FOR BATCHED INFERENCE IN LM EVAL!
    #attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value.float()
 

    
class Attention_Causal(Qwen3MoeAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        # v_first: Optional[torch.Tensor] = None, 
        # k_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        




        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask, is_causal=attention_mask is None and L==S)
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, attn_weights#, v_first, k_first
      
    
class RWKV07BAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        C = self.hidden_size = config.hidden_size
        H = self.num_heads = config.num_attention_heads
        H_kv = config.num_key_value_heads
        N = self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.receptance = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.key = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.value = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.output = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.r_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        

        lora_rank_decay = config.lora_rank_decay
        lora_rank_iclr = config.lora_rank_iclr
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix
        lora_rank_key_residual_mix = config.lora_rank_key_residual_mix
        lora_rank_gate = config.lora_rank_gate

        print(f"v lora projection = {lora_rank_value_residual_mix} k lora projection={lora_rank_key_residual_mix}")


        self.w0 = nn.Parameter(torch.empty(1,1,H*N))
        self.w1 = nn.Parameter(torch.empty(C, lora_rank_decay))
        self.w2 = nn.Parameter(torch.empty(lora_rank_decay, H*N))

        self.a0 = nn.Parameter(torch.empty(1,1,H*N))
        self.a1 = nn.Parameter(torch.empty(C, lora_rank_iclr))
        self.a2 = nn.Parameter(torch.empty(lora_rank_iclr, H*N))


        #self.v0 = nn.Parameter(torch.empty(1,1,H_kv*N))
        self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, H*N))

        #self.k0 = nn.Parameter(torch.empty(1,1,H_kv*N))
        self.k1 = nn.Parameter(torch.empty(C, lora_rank_key_residual_mix))
        self.k2 = nn.Parameter(torch.empty(lora_rank_key_residual_mix, H*N))

      
        self.g1 = nn.Parameter(torch.empty(C, lora_rank_gate))
        self.g2 = nn.Parameter(torch.empty(lora_rank_gate, H*N))

        self.D_MK_LoRA_Scaling = 0.1
        self.D_MV_LoRA_Scaling = 0.2

        #self.r_k = nn.Parameter(torch.empty(H,N))


    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        k_first: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RWKV07BState] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) in (2, 4)
        
        output_shift_state = hidden_states[:, -1:].detach().clone()

        x = hidden_states

        B, T, C = hidden_states.shape
        H = self.num_heads
        N = self.head_dim
        
        q_len = T

        if use_cache and past_key_values is not None and len(past_key_values) > self.layer_idx:
            #print(f'use past state layer {self.layer_idx}')
            input_vk_state, input_shift_state = past_key_values[self.layer_idx]
        else:
            input_vk_state, input_shift_state = torch.zeros(B,H,N,N, dtype=torch.bfloat16,device=x.device), torch.zeros_like(x[:, -1:])

        xr = xw = xk = xv = xa = xg = x

        r = self.r_norm(self.receptance(xr).view(B,T,-1,N))
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) -0.5
        k = self.k_norm(self.key(xk).view(B,T,-1,N))
        v = self.value(xv).view(B,T,-1,N)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            r, k = apply_rotary_pos_emb(r, k, cos, sin, unsqueeze_dim=2)

        if attention_mask is not None:
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    # [B, S]
                    mask = attention_mask[:, -T:]             # [B, T]
                    v = v * mask[:, :, None, None]            # → [B, T, 1, 1] に拡張して掛け算
                elif attention_mask.ndim == 4:
                    # [B, 1, L, S]
                    mask = attention_mask[:, 0, -1, -T:]      # [B, T]
                    v = v * mask[:, :, None, None]            # 同上


        # repeat k/v heads if n_kv_heads < n_heads
        # add LoRA Projection after expand
        k = repeat_kv_rwkv(k, self.num_key_value_groups).view(B, T, -1)# + (((x @ self.k1) @ self.k2) * self.D_MK_LoRA_Scaling)
        v = repeat_kv_rwkv(v, self.num_key_value_groups).view(B, T, -1) + (((x @ self.v1) @ self.v2) * self.D_MV_LoRA_Scaling)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        kk = (k).view(B,T,H,-1).float()
        kk = (kk / (torch.norm(kk, dim=-1, keepdim=True) + 1e-12)).view(B,T,-1).to(k.dtype)
        k = k * (1.0 - w + a)

        aa = -kk
        bb = kk * a
        w = -w.exp()

        r_,w_,k_,v_,aa_,bb_ = [i.view(B,T,H,N) for i in [r,w,k,v,aa,bb]]

        x, output_vk_state = fused_recurrent_rwkv7(r_, w_, k_, v_, aa_, bb_, scale=1.0, initial_state=input_vk_state, output_final_state=True, head_first=False)

        x = x.view(B,T,-1) * (float(N) ** -0.5)

        x = x * g
        x = self.output(x)

        if past_key_values is not None:
            past_key_values.update(output_vk_state, output_shift_state, self.layer_idx, q_len, is_layer_attention(self.config, self.layer_idx))

        return x, v_first, k_first
    

class Qwen3VLMoeTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                gated_output = up * self.act_fn(gate)
                out = gated_output @ self.down_proj[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3VLMoeTextExperts(config)

        # since all the models use norm_topk_prob, we don't need to have a extra check for it
        # self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out, router_logits

class RWKV07BMoEDecoderLayer(nn.Module):
    def __init__(self, config: RWKV07BMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.layer_idx = layer_idx

        if is_layer_attention(config, layer_idx):
            print(f'layer {layer_idx} : attention')
            att_fn = Attention_Causal #Qwen3KeyQuant #Qwen3SWAPrefill #Qwen3DropoutSWASink #Qwen3AttentionNoPE #Qwen3MOBA #Qwen3AttentionVerticalSparse # Qwen3DoubleAttention # Qwen3SymPow #Qwen3Chunk #Qwen3Power #Qwen3MOBA #Qwen3Attention # Qwen3NewAttention # Qwen3AttentionAdapted
        else:
            print(f'layer {layer_idx} : rwkv')
            att_fn = RWKV07BAttention
        
        self.self_attn = att_fn(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states,self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            frozen_residual=frozen_residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            #is_causal=True,
        )
       
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        #print(f'output_attentions = {output_attentions} self_attn_weights = {self_attn_weights}')
        if output_attentions:
            outputs += (self_attn_weights,)

        #outputs = (hidden_states, v_first,k_first,)

        return outputs
  

#@auto_docstring
class RWKV07BMoEPreTrainedModel(PreTrainedModel):
    config: RWKV07BMoEConfig
    config_class = RWKV07BMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV07BMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

class Qwen3MoeMRoPERotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: RWKV07BMoEConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

        # Qwen3VL と同じセクション指定を想定（config にあればそれを使う）
        self.mrope_section = self.config.rope_parameters.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[RWKV07BMoEConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Qwen3 系の通常 RoPE と同じ inv_freq を作る
        """
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # このタイプの RoPE では未使用

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section):
        """
        Qwen3VLTextRotaryEmbedding の apply_interleaved_mrope と互換のロジック。
        freqs: (3, B, T, dim_half)  [0:T, 1:H, 2:W]
        戻り値: (B, T, dim_half)
        """
        # T 軸の周波数をベースにする
        freqs_t = freqs[0]  # (B, T, dim_half)
        _, _, _, dim_half = freqs.shape

        # dim=1: H, dim=2: W
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            length = min(length, dim_half)          # 安全のため head_dim//2 を超えないようにする
            if length <= offset:
                continue
            idx = slice(offset, length, 3)          # 1,4,7,... / 2,5,8,... といったインターリーブ位置
            freqs_t[..., idx] = freqs[dim, ..., idx]

        return freqs_t  # (B, T, dim_half)

    @torch.no_grad()
    @dynamic_rope_update  # RoPE の動的スケーリングにはそのまま対応
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        x: (B, T, hidden_size) 相当を想定（dtype / device を取得するため）
        position_ids: (B, T) または (3, B, T) を想定
        戻り値:
            cos, sin: (B, T, head_dim) で既存 apply_rotary_pos_emb と互換
        """
        device = x.device
        dtype = x.dtype

        # position_ids を (3, B, T) に正規化
        if position_ids.ndim == 2:
            # text-only なので T/H/W すべて同じ position を使う
            position_ids_3 = position_ids.unsqueeze(0).expand(3, -1, -1)  # (3, B, T)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            position_ids_3 = position_ids
        else:
            raise ValueError(
                f"position_ids must be (B,T) or (3,B,T), but got shape {position_ids.shape}"
            )

        B, T = position_ids_3.shape[1], position_ids_3.shape[2]
        dim_half = self.inv_freq.shape[0]  # head_dim // 2

        # inv_freq: (dim_half,) -> (3, B, dim_half, 1)
        inv_freq_expanded = (
            self.inv_freq.view(1, 1, dim_half, 1)
            .float()
            .expand(3, B, dim_half, 1)
            .to(device)
        )

        # position_ids: (3, B, T) -> (3, B, 1, T)
        position_ids_expanded = position_ids_3.float().view(3, B, 1, T)

        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # 強制 float32
            # (3, B, dim_half, 1) @ (3, B, 1, T) -> (3, B, dim_half, T) -> (3, B, T, dim_half)
            freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(2, 3)

            # MRoPE のインターリーブを適用して (B, T, dim_half) を得る
            freqs_t = self.apply_interleaved_mrope(freqs, self.mrope_section)

            # rotary_dim (=head_dim) にするために 2 倍に連結
            emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (B, T, head_dim)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


#@auto_docstring
class RWKV07BMoEModel(RWKV07BMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: RWKV07BMoEConfig
    """

    def __init__(self, config: RWKV07BMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RWKV07BMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeMRoPERotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        
        # Initialize weights and apply final processing
        self.post_init()
    def get_input_embeddings(self):
        # HF の PreTrainedModel から呼ばれる想定の実装
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        # HF の resize_token_embeddings などが使えるように
        self.embed_tokens = new_embeddings

    #@check_model_inputs
    #@auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,#: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, RWKV07BState):
            past_key_values = RWKV07BState()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if self.config.use_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        v_first = None
        k_first = None
        frozen_residual = None
        
        for decoder_layer in self.layers:
            if not is_layer_attention(self.config, decoder_layer.layer_idx):
                frozen_residual = hidden_states#rms_norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            attention_mask = causal_mask_mapping[decoder_layer.attention_type]
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = None
            #attention_mask = None

            layer_outputs = decoder_layer(
                hidden_states,
                frozen_residual=frozen_residual,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
 
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        #if return_legacy_cache:
        #    next_cache = next_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class DualModel(RWKV07BMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLMoeVisionModel._from_config(config.vision_config)
        self.language_model = RWKV07BMoEModel._from_config(config.text_config)
        self.post_init()
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )
        return outputs
        

class RWKV07BMoEForCausalLM(RWKV07BMoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DualModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    #@can_return_tuple
    #@auto_docstring
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#@auto_docstring
class RWKV07BQwen3ForSequenceClassification(RWKV07BMoEPreTrainedModel):
    pass

#@auto_docstring
class RWKV07BQwen3ForTokenClassification(RWKV07BMoEPreTrainedModel):
    pass

#@auto_docstring
class RWKV07BQwen3ForQuestionAnswering(RWKV07BMoEPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`
