from dataclasses import dataclass
from typing import List

import torch
import torch.nn.init as init

import config


@dataclass
class Attention_Params:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    head_n: int
    head_dim: int


def init_attention_weights(c: config.Attention_Config) -> Attention_Params:
    weights = Attention_Params(
        q=torch.empty(c.dim, c.dim),
        k=torch.empty(c.dim, c.dim),
        v=torch.empty(c.dim, c.dim),
        out=torch.empty(c.dim, c.dim),
        head_n=c.head_n,
        head_dim=c.head_dim,
    )

    init.xavier_uniform_(weights.q)
    init.xavier_uniform_(weights.k)
    init.xavier_uniform_(weights.v)
    init.xavier_uniform_(weights.out)

    return weights


@dataclass
class Feed_Forward_Params:
    weight_1: torch.Tensor
    weight_2: torch.Tensor
    weight_3: torch.Tensor


def init_feed_forward_weights(c: config.Feed_Forward_Config) -> Feed_Forward_Params:
    weights = Feed_Forward_Params(
        weight_1=torch.empty(c.dim, c.hidden_dim),
        weight_2=torch.empty(c.hidden_dim, c.dim),
        weight_3=torch.empty(c.dim, c.hidden_dim),
    )

    init.xavier_uniform_(weights.weight_1)
    init.xavier_uniform_(weights.weight_2)
    init.xavier_uniform_(weights.weight_3)

    return weights


@dataclass
class RMS_Norm_Params:
    weight: torch.Tensor


def init_rms_norm_weights(c: config.RMS_Norm_Config) -> RMS_Norm_Params:
    weights = RMS_Norm_Params(
        weight=torch.empty(c.dim),
    )

    init.uniform_(weights.weight)

    return weights


@dataclass
class Layer_Params:
    rms_norm_params_attention: RMS_Norm_Params
    attention_params: Attention_Params
    rms_norm_params_ffn: RMS_Norm_Params
    feed_forward_params: Feed_Forward_Params


def init_layer_weights(c: config.Layer_Config) -> Layer_Params:
    return Layer_Params(
        rms_norm_params_attention=init_rms_norm_weights(c.rms_norm_config_attention),
        attention_params=init_attention_weights(c.attention_config),
        rms_norm_params_ffn=init_rms_norm_weights(c.rms_norm_config_ffn),
        feed_forward_params=init_feed_forward_weights(c.feed_forward_config),
    )


@dataclass
class Transformer_Params:
    layers: List[Layer_Params]
    norm: RMS_Norm_Params
    weight: torch.Tensor


def init_transformer_weights(c: config.Transformer_Config) -> Transformer_Params:
    weights = Transformer_Params(
        layers=[init_layer_weights(c.layer_config) for _ in range(c.layers_n)],
        norm=init_rms_norm_weights(c.norm),
        weight=torch.empty(c.dim, c.vocab_size),
    )

    init.xavier_uniform_(weights.weight)

    return weights
