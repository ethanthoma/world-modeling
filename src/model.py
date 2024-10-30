import functools
from typing import Optional

import torch
import torch.nn.functional as F

import alibi
import weights


@torch.jit.script
def layer_norm(
    weight: torch.Tensor, x: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / torch.sqrt(var + eps)


def attention(
    params: weights.Attention_Params,
    x: torch.Tensor,
    num_attention_heads: int = 6,
    memory: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k_v_states = memory if memory is not None else x
    mask = cross_attention_mask if memory is not None else attention_mask

    seq_len = x.shape[0]
    kv_seq_len = k_v_states.shape[0]

    q_head_dim = params["query"].shape[0] // num_attention_heads
    kv_head_dim = params["key"].shape[0] // num_attention_heads

    q = torch.einsum("sd,dh->sh", x, params["query"])
    k = torch.einsum("sd,dh->sh", k_v_states, params["key"])
    v = torch.einsum("sd,dh->sh", k_v_states, params["value"])

    q = q.view(seq_len, num_attention_heads, q_head_dim).transpose(0, 1)
    k = k.view(kv_seq_len, num_attention_heads, kv_head_dim).transpose(0, 1)
    v = v.view(kv_seq_len, num_attention_heads, kv_head_dim).transpose(0, 1)

    scores = torch.einsum("nqd,nkd->nqk", q, k) / (q_head_dim**0.5)

    if memory is None:
        bias = alibi.build_alibi_bias(num_attention_heads, seq_len, x.device)[
            :, :seq_len, :kv_seq_len
        ]
        scores = scores + bias

    if mask is not None:
        if memory is None:
            mask = mask.unsqueeze(0) * mask.unsqueeze(1)
        else:
            mask = mask.unsqueeze(0).expand(seq_len, -1)

        scores = scores.masked_fill(
            mask == 0, -0.7 * float(torch.finfo(scores.dtype).max)
        )

    attn = torch.softmax(scores, dim=-1)

    out = torch.einsum("hqk,nkd->nqd", attn, v)
    out = out.transpose(0, 1).contiguous().view(seq_len, -1)
    out = torch.einsum("sd,dh->sh", out, params["output"])

    return out


def feed_forward(params: weights.Feed_Forward_Params, x: torch.Tensor) -> torch.Tensor:
    gated = params["intermediate"].shape[-1] != params["output"].shape[0]

    h = torch.einsum("td,df->tf", x, params["intermediate"])

    if gated:
        h_gated, h_ungated = h.chunk(2, dim=1)
        h = F.gelu(h_gated) * h_ungated
    else:
        h = F.gelu(h)

    h = torch.einsum("td,df->tf", h, params["output"])

    return h


def transformer_layer(
    params: weights.Layer_Params,
    h: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    memory: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    residual = h

    if memory is not None:
        h = layer_norm(params["ln_1"], h)

    h = attention(
        params["attention"],
        h,
        memory=memory,
        attention_mask=attention_mask,
        cross_attention_mask=cross_attention_mask,
    )

    h = h + residual

    if memory is None:
        h = layer_norm(params["ln_1"], h)

    residual = h
    h = feed_forward(params["feed_forward"], h)
    h = h + residual
    h = layer_norm(params["ln_2"], h)

    return h


def bert_model(
    params: weights.Transformer_Params,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    h = params["embeddings"][input_ids]

    for layer in params["layers"]:
        h = transformer_layer(layer, h, attention_mask=attention_mask)

    h = layer_norm(params["layernorm"], h)

    return h


def aggregate_encodings(
    params: weights.Aggregator_Params,
    text_hidden: torch.Tensor,
    graph_hidden: torch.Tensor,
) -> torch.Tensor:
    combined = torch.cat([text_hidden, graph_hidden], dim=0)

    h = layer_norm(params["layernorm"], combined)

    for layer in params["layers"]:
        h = transformer_layer(layer, h)

    return h


def gpt2(
    params: weights.Transformer_Params,
    input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_length = input_ids.size(0)
    causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    causal_mask = (~causal_mask).to(input_ids.device)

    if attention_mask is not None:
        causal_mask = causal_mask.float()
        causal_mask = causal_mask * attention_mask

    h = params["embeddings"][input_ids]

    for layer in params["layers"]:
        h = transformer_layer(
            layer,
            h,
            attention_mask=causal_mask,
            memory=encoder_hidden_states,
            cross_attention_mask=encoder_attention_mask,
        )

    h = layer_norm(params["layernorm"], h)

    logits = torch.einsum("sd,vd->sv", h, params["embeddings"])

    return logits


def worldformer(
    params: weights.Worldformer_Params,
    textual_input_ids: torch.Tensor,
    graph_input_ids: torch.Tensor,
    textual_input_attention_mask: torch.Tensor,
    graph_input_attention_mask: torch.Tensor,
    action_target_ids: torch.Tensor,
    graph_target_ids: torch.Tensor,
    action_target_attention_mask: torch.Tensor,
    graph_target_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_encoded = bert_model(
        params["text_encoder"],
        textual_input_ids,
        textual_input_attention_mask,
    )

    graph_encoded = bert_model(
        params["graph_encoder"],
        graph_input_ids,
        graph_input_attention_mask,
    )

    combined_state = aggregate_encodings(
        params["aggregator"], text_encoded, graph_encoded
    )

    encoder_attention_mask = None
    if (
        textual_input_attention_mask is not None
        and graph_input_attention_mask is not None
    ):
        encoder_attention_mask = torch.cat(
            [textual_input_attention_mask, graph_input_attention_mask], dim=0
        )

    action_logits = None
    if action_target_ids is not None:
        action_logits = gpt2(
            params["action_decoder"],
            action_target_ids,
            combined_state,
            attention_mask=action_target_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

    graph_logits = None
    if graph_target_ids is not None:
        graph_logits = gpt2(
            params["graph_decoder"],
            graph_target_ids,
            combined_state,
            attention_mask=graph_target_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

    return action_logits, graph_logits
