import pathlib
from dataclasses import dataclass
from typing import List

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


def load_bert_weights(
    ckpt_path: pathlib.Path, bert_config: config.BERT_Config
) -> BERT_Params:
    state_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu"), weights_only=True
    )

    layers = []
    for i in range(bert_config.num_hidden_layers):
        attention_params = Attention_Params(
            query=state_dict[f"bert.encoder.layer.{i}.attention.self.query.weight"],
            key=state_dict[f"bert.encoder.layer.{i}.attention.self.key.weight"],
            value=state_dict[f"bert.encoder.layer.{i}.attention.self.value.weight"],
            output=state_dict[f"bert.encoder.layer.{i}.attention.output.dense.weight"],
            layernorm=state_dict[
                f"bert.encoder.layer.{i}.attention.output.LayerNorm.gamma"
            ],
        )

        feed_forward_params = Feed_Forward_Params(
            intermediate=state_dict[
                f"bert.encoder.layer.{i}.intermediate.dense.weight"
            ],
            output=state_dict[f"bert.encoder.layer.{i}.output.dense.weight"],
            layernorm=state_dict[f"bert.encoder.layer.{i}.output.LayerNorm.gamma"],
        )

        layer_params = Layer_Params(
            attention=attention_params,
            feed_forward=feed_forward_params,
        )

        layers.append(layer_params)

    return BERT_Params(
        embeddings=state_dict["bert.embeddings.word_embeddings.weight"],
        position_embeddings=state_dict["bert.embeddings.position_embeddings.weight"],
        token_type_embeddings=state_dict[
            "bert.embeddings.token_type_embeddings.weight"
        ],
        layernorm=state_dict["bert.embeddings.LayerNorm.gamma"],
        layers=layers,
        pooler=state_dict["bert.pooler.dense.weight"],
    )
