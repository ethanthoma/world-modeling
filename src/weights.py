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
    position_embeddings: torch.Tensor
    token_type_embeddings: torch.Tensor
    layernorm: torch.Tensor
    layers: List[Layer_Params]
    pooler: torch.Tensor


@dataclass
class GPT2_Layer_Params:
    attention: Attention_Params
    mlp: Feed_Forward_Params
    ln_1: torch.Tensor
    ln_2: torch.Tensor


@dataclass
class GPT2_Params:
    wte: torch.Tensor
    wpe: torch.Tensor
    ln_f: torch.Tensor
    layers: List[GPT2_Layer_Params]


@dataclass
class Aggregator_Params:
    projection: torch.Tensor
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
    ckpt_path: pathlib.Path, config: config.BERT_Config, prefix: str = "bert"
) -> BERT_Params:
    state_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu"), weights_only=True
    )

    layers = []
    for i in range(config.num_hidden_layers):
        base = f"{prefix}.encoder.layer.{i}"
        attention_params = Attention_Params(
            query=state_dict[f"{base}.attention.self.query.weight"],
            key=state_dict[f"{base}.attention.self.key.weight"],
            value=state_dict[f"{base}.attention.self.value.weight"],
            output=state_dict[f"{base}.attention.output.dense.weight"],
            layernorm=state_dict[f"{base}.attention.output.LayerNorm.gamma"],
        )

        feed_forward_params = Feed_Forward_Params(
            intermediate=state_dict[f"{base}.intermediate.dense.weight"],
            output=state_dict[f"{base}.output.dense.weight"],
            layernorm=state_dict[f"{base}.output.LayerNorm.gamma"],
        )

        layers.append(
            Layer_Params(attention=attention_params, feed_forward=feed_forward_params)
        )

    return BERT_Params(
        embeddings=state_dict[f"{prefix}.embeddings.word_embeddings.weight"],
        position_embeddings=state_dict[
            f"{prefix}.embeddings.position_embeddings.weight"
        ],
        token_type_embeddings=state_dict[
            f"{prefix}.embeddings.token_type_embeddings.weight"
        ],
        layernorm=state_dict[f"{prefix}.embeddings.LayerNorm.gamma"],
        layers=layers,
        pooler=state_dict[f"{prefix}.pooler.dense.weight"],
    )


def load_gpt2_weights(
    ckpt_path: pathlib.Path, config: config.GPT2_Config
) -> GPT2_Params:
    state_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu"), weights_only=True
    )

    layers = []
    for i in range(config.num_hidden_layers):
        # Split concatenated QKV weights
        qkv_weights = state_dict[f"h.{i}.attn.c_attn.weight"]
        q, k, v = split_qkv_weights(qkv_weights)

        attention_params = Attention_Params(
            query=q,
            key=k,
            value=v,
            output=state_dict[f"h.{i}.attn.c_proj.weight"],
            layernorm=state_dict[f"h.{i}.ln_1.weight"],
        )

        feed_forward_params = Feed_Forward_Params(
            intermediate=state_dict[f"h.{i}.mlp.c_fc.weight"],
            output=state_dict[f"h.{i}.mlp.c_proj.weight"],
            layernorm=state_dict[f"h.{i}.ln_2.weight"],
        )

        layer = GPT2_Layer_Params(
            attention=attention_params,
            mlp=feed_forward_params,
            ln_1=state_dict[f"h.{i}.ln_1.weight"],
            ln_2=state_dict[f"h.{i}.ln_2.weight"],
        )

        layers.append(layer)

    return GPT2_Params(
        wte=state_dict["wte.weight"],
        wpe=state_dict["wpe.weight"],
        ln_f=state_dict["ln_f.weight"],
        layers=layers,
    )


def split_qkv_weights(
    c_attn_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = c_attn_weight.shape[0]
    split_size = hidden_size
    return tuple(c_attn_weight.split(split_size, dim=1))
