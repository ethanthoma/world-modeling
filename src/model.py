from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

import alibi
import weights


def attention(
    params: weights.Attention_Params,
    x: torch.Tensor,
    memory: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k_v_states = memory if memory is not None else x

    seq_len = x.shape[0]
    kv_seq_len = k_v_states.shape[0]
    head_dim = params.query.shape[0] // params.num_attention_heads

    q = torch.einsum("sd,dh->sh", x, params.query)
    k = torch.einsum("sd,dh->sh", k_v_states, params.key)
    v = torch.einsum("sd,dh->sh", k_v_states, params.value)

    q = q.view(seq_len, params.num_attention_heads, head_dim).transpose(0, 1)
    k = k.view(kv_seq_len, params.num_attention_heads, head_dim).transpose(0, 1)
    v = v.view(kv_seq_len, params.num_attention_heads, head_dim).transpose(0, 1)

    scores = torch.einsum("nqd,nkd->nqk", q, k) / (head_dim**0.5)

    bias = alibi.build_alibi_bias(params.num_attention_heads, seq_len, x.device)[
        :, :seq_len, :kv_seq_len
    ]

    # for GPT2 models as the k_v_states is twice as large
    if bias.shape[-1] != scores.shape[-1]:
        bias = bias.repeat(1, 1, 2)

    scores = scores + bias

    if mask is not None:
        scores = scores.masked_fill(mask[None, None, :] == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    out = torch.einsum("hqk,nkd->nqd", attn, v)
    out = out.transpose(0, 1).contiguous().view(seq_len, -1)

    return torch.einsum("sd,dh->sh", out, params.output)


def feed_forward(params: weights.Feed_Forward_Params, x: torch.Tensor) -> torch.Tensor:
    gated = params.intermediate.shape[-1] != params.output.shape[0]

    h = torch.einsum("td,df->tf", x, params.intermediate)

    if gated:
        h_gated, h_ungated = h.chunk(2, dim=1)
        h = torch.nn.functional.gelu(h_gated) * h_ungated
    else:
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
    memory: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if memory is None:
        h_attn = attention(params.attention, h, mask=mask)
        h = h + h_attn
        h = layer_norm(h, params.attention.layernorm)

    # for GPT2 models for cross-attention
    if memory is not None:
        h = layer_norm(h, params.attention.layernorm)
        h_attn = attention(params.attention, h, memory=memory)
        h = h + h_attn

    h_ffn = feed_forward(params.feed_forward, h)
    h = h + h_ffn
    h = layer_norm(h, params.feed_forward.layernorm)

    return h


def bert_model(
    params: weights.BERT_Params,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    embeddings = params.embeddings[input_ids]

    h = layer_norm(embeddings, params.layernorm)

    for layer in params.layers:
        h = transformer_layer(layer, h, mask=attention_mask)

    return h


def aggregate_encodings(
    params: weights.Aggregator_Params,
    text_hidden: torch.Tensor,
    graph_hidden: torch.Tensor,
) -> torch.Tensor:
    combined = torch.cat([text_hidden, graph_hidden], dim=0)

    h = layer_norm(params.layernorm, combined)

    for layer in params.layers:
        h = transformer_layer(layer, h)

    return h


def gpt2(
    params: weights.GPT2_Params,
    input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    seq_length = input_ids.size(0)
    causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    causal_mask = causal_mask.to(input_ids.device)

    h = params.embeddings[input_ids]

    for layer in params.layers:
        h = transformer_layer(
            layer,
            h,
            memory=encoder_hidden_states,
            mask=causal_mask,
        )

    return h


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_encoded = bert_model(
        params.text_encoder,
        textual_input_ids,
        textual_input_attention_mask,
    )

    graph_encoded = bert_model(
        params.graph_encoder,
        graph_input_ids,
        graph_input_attention_mask,
    )

    combined_state = aggregate_encodings(params.aggregator, text_encoded, graph_encoded)

    action_output = (
        gpt2(
            params.action_decoder,
            action_target_ids,
            combined_state,
        )
        if action_target_ids is not None
        else None
    )

    graph_output = (
        gpt2(
            params.graph_decoder,
            graph_target_ids,
            combined_state,
        )
        if graph_target_ids is not None
        else None
    )

    return action_output, graph_output
