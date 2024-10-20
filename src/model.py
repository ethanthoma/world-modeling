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

    q = torch.einsum("sd,dh->sh", x, params.q)
    k = torch.einsum("sd,dh->sh", x, params.k)
    v = torch.einsum("sd,dh->sh", x, params.v)

    q = q.view(seq_len, params.head_n, params.head_dim).transpose(0, 1)
    k = k.view(seq_len, params.head_n, params.head_dim).transpose(0, 1)
    v = v.view(seq_len, params.head_n, params.head_dim).transpose(0, 1)

    scores = torch.einsum("nqd,nkd->nqk", q, k) / (params.head_dim**0.5)

    if mask is not None:
        scores = scores.masked_fill(
            mask[:, None, None, :] == 0, float(torch.finfo(scores.dtype).min)
        )

    attn = F.softmax(scores, dim=-1)

    out = torch.einsum("hqk,nkd->nqd", attn, v)  # [ head_n, seq_len, head_dim ]

    out = out.transpose(0, 1).contiguous().view(seq_len, -1)

    return torch.einsum("sd,dh->sh", out, params.out)  # [  seq_len, dim ]


def feed_forward(params: weights.Feed_Forward_Params, x: torch.Tensor) -> torch.Tensor:
    h1 = torch.einsum("td,df->tf", x, params.weight_1)
    h1 = F.silu(h1)

    h2 = torch.einsum("td,df->tf", x, params.weight_3)

    h = h1 * h2

    x = torch.einsum("td,df->tf", h, params.weight_2)
    return x


def rms_norm(
    params: weights.RMS_Norm_Params,
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    h = torch.pow(x, 2).mean(-1, keepdim=True) + eps
    h = torch.rsqrt(h)
    h = torch.einsum("sd,sd->sd", x, h)
    h = torch.einsum("d,sd->sd", params.weight, h)
    return h


def transformer_layer(
    params: weights.Layer_Params,
    h: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    h_norm = rms_norm(params.rms_norm_params_attention, h)

    h_attn = attention(params.attention_params, h_norm)

    h = h.add(h_attn)

    h_norm = rms_norm(params.rms_norm_params_ffn, h)

    h = h.add(feed_forward(params.feed_forward_params, h_norm))

    return h


def transformer(
    params: weights.Transformer_Params,
    x: torch.Tensor,
) -> torch.Tensor:
    h = x  # [ seq_len, dim ]

    for layer_params in params.layers:
        h = transformer_layer(layer_params, h, None)  # [ seq_len, dim ]

    h = rms_norm(params.norm, h)  # [ seq_len, dim ]

    logits = torch.einsum("sd,dv->sv", h, params.weight)  # [ seq_len, vocab_size ]

    return logits
