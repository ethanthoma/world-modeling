import copy
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.init as init

import config


@dataclass
class Attention_Params:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    layernorm: torch.Tensor
    num_attention_heads: int


@dataclass
class Feed_Forward_Params:
    intermediate: torch.Tensor
    output: torch.Tensor
    layernorm: torch.Tensor


@dataclass
class Layer_Params:
    attention: Attention_Params
    feed_forward: Feed_Forward_Params


@dataclass
class BERT_Params:
    embeddings: torch.Tensor
    layernorm: torch.Tensor
    layers: List[Layer_Params]


@dataclass
class GPT2_Params:
    embeddings: torch.Tensor
    layernorm: torch.Tensor
    layers: List[Layer_Params]


@dataclass
class Aggregator_Params:
    layernorm: torch.Tensor
    layers: List[Layer_Params]


@dataclass
class Worldformer_Params:
    text_encoder: BERT_Params
    graph_encoder: BERT_Params
    aggregator: Aggregator_Params
    action_decoder: GPT2_Params
    graph_decoder: GPT2_Params


def load_bert_weights(
    config: config.BERT_Config,
    ckpt_path: pathlib.Path,
    prefix: str = "bert",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> BERT_Params:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    layers = []
    for i in range(config.num_hidden_layers):
        base = f"{prefix}.encoder.layer.{i}"

        qkv_weights = state_dict[f"{base}.attention.self.Wqkv.weight"]
        q, k, v = split_qkv_weights(qkv_weights)

        attention_params = Attention_Params(
            query=q,
            key=k,
            value=v,
            output=state_dict[f"{base}.attention.output.dense.weight"],
            layernorm=state_dict[f"{base}.attention.output.LayerNorm.weight"],
            num_attention_heads=config.num_attention_heads,
        )

        feed_forward_params = Feed_Forward_Params(
            intermediate=state_dict[f"{base}.mlp.gated_layers.weight"].T,
            output=state_dict[f"{base}.mlp.wo.weight"].T,
            layernorm=state_dict[f"{base}.mlp.layernorm.weight"],
        )

        layers.append(
            Layer_Params(attention=attention_params, feed_forward=feed_forward_params)
        )

    return BERT_Params(
        embeddings=state_dict[f"{prefix}.embeddings.word_embeddings.weight"],
        layernorm=state_dict[f"{prefix}.embeddings.LayerNorm.weight"],
        layers=layers,
    )


def load_gpt2_weights(
    config: config.GPT2_Config,
    ckpt_path: pathlib.Path,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> GPT2_Params:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    layers = []
    for i in range(config.num_hidden_layers):
        qkv_weights = state_dict[f"h.{i}.attn.c_attn.weight"]
        q, k, v = split_qkv_weights(qkv_weights)

        attention_params = Attention_Params(
            query=q,
            key=k,
            value=v,
            output=state_dict[f"h.{i}.attn.c_proj.weight"],
            layernorm=state_dict[f"h.{i}.ln_1.weight"],
            num_attention_heads=config.num_attention_heads,
        )

        feed_forward_params = Feed_Forward_Params(
            intermediate=state_dict[f"h.{i}.mlp.c_fc.weight"],
            output=state_dict[f"h.{i}.mlp.c_proj.weight"],
            layernorm=state_dict[f"h.{i}.ln_2.weight"],
        )

        layer = Layer_Params(
            attention=attention_params,
            feed_forward=feed_forward_params,
        )

        layers.append(layer)

    return GPT2_Params(
        embeddings=state_dict["wte.weight"],
        layernorm=state_dict["ln_f.weight"],
        layers=layers,
    )


def split_qkv_weights(
    attn_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim1, dim2 = attn_weight.shape
    split_dim = 1 if dim1 < dim2 else 0
    hidden_size = min(attn_weight.shape)
    return tuple(attn_weight.split(hidden_size, dim=split_dim))


def init_aggregator_weights(
    config: config.Aggregator_Config,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Aggregator_Params:
    def init_linear(out_features: int, in_features: int) -> torch.Tensor:
        weight = torch.empty(out_features, in_features, device=device)
        init.normal_(weight, mean=0.0, std=0.02)
        return weight

    layers: List[Layer_Params] = []
    for _ in range(config.num_hidden_layers):
        attention_params = Attention_Params(
            query=init_linear(config.hidden_size, config.hidden_size),
            key=init_linear(config.hidden_size, config.hidden_size),
            value=init_linear(config.hidden_size, config.hidden_size),
            output=init_linear(config.hidden_size, config.hidden_size),
            layernorm=torch.ones(config.hidden_size, device=device),
            num_attention_heads=config.num_attention_heads,
        )

        feed_forward_params = Feed_Forward_Params(
            intermediate=init_linear(config.hidden_size, config.intermediate_size),
            output=init_linear(config.intermediate_size, config.hidden_size),
            layernorm=torch.ones(config.hidden_size, device=device),
        )

        layers.append(
            Layer_Params(attention=attention_params, feed_forward=feed_forward_params)
        )

    return Aggregator_Params(
        layernorm=torch.ones(config.hidden_size, device=device),
        layers=layers,
    )


def init_worldformer(
    config: config.Worldformer_Config,
    bert_path: pathlib.Path,
    gpt2_path: pathlib.Path,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Worldformer_Params:
    text_encoder = load_bert_weights(config.encoder_config, bert_path, device=device)
    graph_encoder = copy.deepcopy(text_encoder)

    action_decoder = load_gpt2_weights(config.decoder_config, gpt2_path, device=device)
    graph_decoder = copy.deepcopy(action_decoder)

    aggregator = init_aggregator_weights(
        config.aggregator_config,
        device=device,
    )

    return Worldformer_Params(
        text_encoder=text_encoder,
        graph_encoder=graph_encoder,
        aggregator=aggregator,
        action_decoder=action_decoder,
        graph_decoder=graph_decoder,
    )
