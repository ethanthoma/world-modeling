from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F

import weights


def attention(
    params: weights.Attention_Params,
    x: torch.Tensor,  # x: [ seq_len, dim ]
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_len, _ = x.shape
    head_dim = params.query.shape[0] // config.num_attention_heads

    q = torch.einsum("sd,dh->sh", x, params.query)
    k = torch.einsum("sd,dh->sh", x, params.key)
    v = torch.einsum("sd,dh->sh", x, params.value)

    q = q.view(seq_len, config.num_attention_heads, head_dim).transpose(0, 1)
    k = k.view(seq_len, config.num_attention_heads, head_dim).transpose(0, 1)
    v = v.view(seq_len, config.num_attention_heads, head_dim).transpose(0, 1)

    scores = torch.einsum("nqd,nkd->nqk", q, k) / (head_dim**0.5)

    if mask is not None:
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    out = torch.einsum("hqk,nkd->nqd", attn, v)
    out = out.transpose(0, 1).contiguous().view(seq_len, -1)

    out = torch.einsum("sd,dh->sh", out, params.output)
    return layer_norm(out, params.layernorm)


def feed_forward(params: weights.Feed_Forward_Params, x: torch.Tensor) -> torch.Tensor:
    h = torch.einsum("td,df->tf", x, params.intermediate)
    h = torch.nn.functional.gelu(h)
    h = torch.einsum("td,df->tf", h, params.output)
    return layer_norm(h, params.layernorm)


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / torch.sqrt(var + eps)


def transformer_layer(
    params: weights.Layer_Params,
    h: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    h_attn = attention(params.attention, h, mask)
    h = h + h_attn

    h_ffn = feed_forward(params.feed_forward, h)
    h = h + h_ffn

    return h


def bert_model(
    params: weights.BERT_Params,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

    word_embeddings = params.embeddings[input_ids]
    position_embeddings = params.position_embeddings[position_ids]
    token_type_embeddings = params.token_type_embeddings[token_type_ids]

    embeddings = word_embeddings + position_embeddings + token_type_embeddings
    h = layer_norm(embeddings, params.layernorm)

    for layer_params in params.layers:
        h = transformer_layer(layer_params, h, attention_mask)

    pooled_output = torch.tanh(torch.einsum("td,df->tf", h[:, 0], params.pooler))

    return h, pooled_output
