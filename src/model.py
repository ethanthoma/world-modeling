from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F

import weights


def attention(
    params: weights.Attention_Params,
    x: torch.Tensor,
    memory: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_len, _ = x.shape
    head_dim = params.query.shape[0] // config.num_attention_heads

    q = torch.einsum("sd,dh->sh", x, params.query)

    if memory is not None:
        k = torch.einsum("sd,dh->sh", memory, params.key)
        v = torch.einsum("sd,dh->sh", memory, params.value)
        kv_len = memory.size(0)
    else:
        k = torch.einsum("sd,dh->sh", x, params.key)
        v = torch.einsum("sd,dh->sh", x, params.value)
        kv_len = seq_len

    q = q.view(seq_len, config.num_attention_heads, head_dim).transpose(0, 1)
    k = k.view(kv_len, config.num_attention_heads, head_dim).transpose(0, 1)
    v = v.view(kv_len, config.num_attention_heads, head_dim).transpose(0, 1)

    scores = torch.einsum("nqd,nkd->nqk", q, k) / (head_dim**0.5)

    if cross_attention_mask is not None:
        scores = scores.masked_fill(
            cross_attention_mask[:, None, None, :] == 0, float("-inf")
        )
    elif mask is not None:
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,nkd->nqd", attn, v)
    out = out.transpose(0, 1).contiguous().view(seq_len, -1)

    return torch.einsum("sd,dh->sh", out, params.output)


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
    memory: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    h_attn = attention(params.attention, h, mask=mask)
    h = h + h_attn
    h = layer_norm(h, params.attention.layernorm)

    if memory is not None:
        h_cross = attention(
            params.cross_attention,
            h,
            memory=memory,
            cross_attention_mask=cross_attention_mask,
        )
        h = h + h_cross
        h = layer_norm(h, params.cross_attention.layernorm)

    h_ffn = feed_forward(params.feed_forward, h)
    h = h + h_ffn
    h = layer_norm(h, params.feed_forward.layernorm)

    return h


def encoder(
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


def decoder(
    params: weights.GPT2_Params,
    input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_length = input_ids.size(1)
    causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    causal_mask = causal_mask.to(input_ids.device)

    h = params.embeddings[input_ids]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    h = h + params.position_embeddings[position_ids]

    for layer in params.layers:
        h = transformer_layer(
            layer,
            h,
            memory=encoder_hidden_states,
            mask=causal_mask,
            cross_attention_mask=encoder_attention_mask,
        )

    h = torch.einsum("bsd,dv->bsv", h, params.lm_head)
    return h


def aggregate_encodings(
    params: weights.Layer_Params,
    text_encoding: torch.Tensor,
    graph_encoding: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    combined = torch.cat([text_encoding, graph_encoding], dim=-1)

    h = feed_forward(params.feed_forward, combined)

    for layer in params.layers:
        h = transformer_layer(layer, h, mask=mask)

    return h


def worldformer(
    params: WorldformerParams,
    input_ids: torch.Tensor,
    graph_input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    graph_attention_mask: Optional[torch.Tensor] = None,
    action_input_ids: Optional[torch.Tensor] = None,
    graph_target_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_encoded, _ = bert_model(
        params.text_encoder,
        input_ids,
        torch.zeros_like(input_ids),  # token_type_ids
        attention_mask,
    )

    graph_encoded, _ = bert_model(
        params.graph_encoder,
        graph_input_ids,
        torch.zeros_like(graph_input_ids),  # token_type_ids
        graph_attention_mask,
    )

    combined_state = aggregate_encodings(
        params.aggregator, text_encoded, graph_encoded, attention_mask
    )

    action_output = (
        decoder(
            params.action_decoder,
            action_input_ids,
            combined_state,
            attention_mask=attention_mask,
        )
        if action_input_ids is not None
        else None
    )

    graph_output = (
        decoder(
            params.graph_decoder,
            graph_target_ids,
            combined_state,
            attention_mask=graph_attention_mask,
        )
        if graph_target_ids is not None
        else None
    )

    return action_output, graph_output
