import copy
import functools
import pathlib
from typing import Any, Literal, TypedDict, TypeVar, Union

import torch
import torch.nn.init as init

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Weight_Dict = TypeVar("Weight_Dict", bound="Base_Weight_Dict")


class Base_Weight_Dict(TypedDict):
    """Base type for all weight dictionaries"""

    pass


class Layer_Norm_Params(Base_Weight_Dict):
    weight: torch.Tensor
    bias: torch.Tensor


class Attention_Params(Base_Weight_Dict):
    query: torch.Tensor
    query_bias: torch.Tensor
    key: torch.Tensor
    key_bias: torch.Tensor
    value: torch.Tensor
    value_bias: torch.Tensor
    output: torch.Tensor


class Feed_Forward_Params(Base_Weight_Dict):
    intermediate: torch.Tensor
    output: torch.Tensor
    output_bias: torch.Tensor


class Layer_Params(Base_Weight_Dict):
    attention: Attention_Params
    feed_forward: Feed_Forward_Params
    ln_1: Layer_Norm_Params
    ln_2: Layer_Norm_Params


class Transformer_Params(Base_Weight_Dict):
    embeddings: torch.Tensor
    layers: list[Layer_Params]
    ln: Layer_Norm_Params


class Aggregator_Params(Base_Weight_Dict):
    layers: list[Layer_Params]
    ln: Layer_Norm_Params


class Worldformer_Params(Base_Weight_Dict):
    text_encoder: Transformer_Params
    graph_encoder: Transformer_Params
    aggregator: Aggregator_Params
    action_decoder: Transformer_Params
    graph_decoder: Transformer_Params


def load_bert_weights(
    config: config.BERT_Config,
    ckpt_path: pathlib.Path,
    prefix: str = "bert",
    device: torch.device = DEVICE,
) -> Transformer_Params:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    state_dict[f"{prefix}.embeddings.word_embeddings.weight"] = increase_embed_size(
        config, state_dict[f"{prefix}.embeddings.word_embeddings.weight"], std=0.02
    )

    layers: list[Layer_Params] = []
    for i in range(config.num_hidden_layers):
        base = f"{prefix}.encoder.layer.{i}"

        qkv_weights = state_dict[f"{base}.attention.self.Wqkv.weight"]
        qkv_bias = state_dict[f"{base}.attention.self.Wqkv.bias"]
        q_weight, k_weight, v_weight = split_qkv_weights(qkv_weights)
        q_bias, k_bias, v_bias = split_qkv_weights(qkv_bias)

        attention_params: Attention_Params = {
            "query": q_weight,
            "query_bias": q_bias,
            "key": k_weight,
            "key_bias": k_bias,
            "value": v_weight,
            "value_bias": v_bias,
            "output": state_dict[f"{base}.attention.output.dense.weight"],
        }

        feed_forward_params: Feed_Forward_Params = {
            "intermediate": state_dict[f"{base}.mlp.gated_layers.weight"].T,
            "output": state_dict[f"{base}.mlp.wo.weight"].T,
            "output_bias": state_dict[f"{base}.mlp.wo.bias"],
        }

        ln_1_params: Layer_Norm_Params = {
            "weight": state_dict[f"{base}.attention.output.LayerNorm.weight"],
            "bias": state_dict[f"{base}.attention.output.LayerNorm.bias"],
        }

        ln_2_params: Layer_Norm_Params = {
            "weight": state_dict[f"{base}.mlp.layernorm.weight"],
            "bias": state_dict[f"{base}.mlp.layernorm.bias"],
        }

        layers.append(
            {
                "attention": attention_params,
                "feed_forward": feed_forward_params,
                "ln_1": ln_1_params,
                "ln_2": ln_2_params,
            }
        )

    ln_params: Layer_Norm_Params = {
        "weight": state_dict[f"{prefix}.embeddings.LayerNorm.weight"],
        "bias": state_dict[f"{prefix}.embeddings.LayerNorm.bias"],
    }

    return {
        "embeddings": state_dict[f"{prefix}.embeddings.word_embeddings.weight"],
        "layers": layers,
        "ln": ln_params,
    }


def load_gpt2_weights(
    config: config.GPT2_Config,
    ckpt_path: pathlib.Path,
    device: torch.device = DEVICE,
) -> Transformer_Params:
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    state_dict["wte.weight"] = increase_embed_size(
        config, state_dict["wte.weight"], std=0.02
    )

    layers: list[Layer_Params] = []
    for i in range(config.num_hidden_layers):
        base = f"h.{i}"

        qkv_weights = state_dict[f"{base}.attn.c_attn.weight"]
        qkv_bias = state_dict[f"{base}.attn.c_attn.bias"]
        q_weight, k_weight, v_weight = split_qkv_weights(qkv_weights)
        q_bias, k_bias, v_bias = split_qkv_weights(qkv_bias)

        attention_params: Attention_Params = {
            "query": q_weight,
            "query_bias": q_bias,
            "key": k_weight,
            "key_bias": k_bias,
            "value": v_weight,
            "value_bias": v_bias,
            "output": state_dict[f"{base}.attn.c_proj.weight"],
        }

        feed_forward_params: Feed_Forward_Params = {
            "intermediate": state_dict[f"{base}.mlp.c_fc.weight"],
            "output": state_dict[f"{base}.mlp.c_proj.weight"],
            "output_bias": state_dict[f"{base}.mlp.c_proj.bias"],
        }

        ln_1_params: Layer_Norm_Params = {
            "weight": state_dict[f"{base}.ln_1.weight"],
            "bias": state_dict[f"{base}.ln_1.bias"],
        }

        ln_2_params: Layer_Norm_Params = {
            "weight": state_dict[f"{base}.ln_2.weight"],
            "bias": state_dict[f"{base}.ln_2.bias"],
        }

        layers.append(
            {
                "attention": attention_params,
                "feed_forward": feed_forward_params,
                "ln_1": ln_1_params,
                "ln_2": ln_2_params,
            }
        )

    ln_params: Layer_Norm_Params = {
        "weight": state_dict["ln_f.weight"],
        "bias": state_dict["ln_f.bias"],
    }

    return {
        "embeddings": state_dict["wte.weight"],
        "layers": layers,
        "ln": ln_params,
    }


def increase_embed_size(
    config: TypedDict,
    old_embeddings: torch.Tensor,
    std: float = 0.02,
) -> torch.Tensor:
    """
    Increase embedding matrix size for new tokens while preserving existing embeddings
    """
    new_embeddings = init.normal_(
        torch.empty(
            config.vocab_size,
            old_embeddings.shape[1],
            dtype=old_embeddings.dtype,
            device=old_embeddings.device,
        ),
        mean=0.0,
        std=std,
    )

    new_embeddings[: old_embeddings.shape[0]] = old_embeddings
    return new_embeddings


def split_qkv_weights(
    attn_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split concatenated QKV tensor into separate Q, K, V tensors
    """
    dim, size = max(enumerate(attn_weight.shape), key=lambda x: x[1])
    hidden_size = size // 3
    return tuple(attn_weight.split(hidden_size, dim=dim))


def init_aggregator_weights(
    config: config.Aggregator_Config,
    device: torch.device = DEVICE,
) -> Aggregator_Params:
    def init_linear(
        out_features: int, in_features: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.empty(out_features, in_features, device=device)
        bias = torch.zeros(out_features, device=device)
        init.normal_(weight, mean=0.0, std=0.02)
        return weight, bias

    layers: list[Layer_Params] = []
    for _ in range(config.num_hidden_layers):
        q_weight, q_bias = init_linear(config.hidden_size, config.hidden_size)
        k_weight, k_bias = init_linear(config.hidden_size, config.hidden_size)
        v_weight, v_bias = init_linear(config.hidden_size, config.hidden_size)
        o_weight, o_bias = init_linear(config.hidden_size, config.hidden_size)

        attention_params: Attention_Params = {
            "query": q_weight,
            "key": k_weight,
            "value": v_weight,
            "output": o_weight,
            "query_bias": q_bias,
            "key_bias": k_bias,
            "value_bias": v_bias,
            "output_bias": o_bias,
        }

        intermediate_weight, _ = init_linear(
            config.hidden_size, config.intermediate_size
        )
        output_weight, output_bias = init_linear(
            config.intermediate_size, config.hidden_size
        )

        feed_forward_params: Feed_Forward_Params = {
            "intermediate": intermediate_weight,
            "output": output_weight,
            "output_bias": output_bias,
        }

        ln_1_params: Layer_Norm_Params = {
            "weight": torch.ones(config.hidden_size, device=device),
            "bias": torch.zeros(config.hidden_size, device=device),
        }

        ln_2_params: Layer_Norm_Params = {
            "weight": torch.ones(config.hidden_size, device=device),
            "bias": torch.zeros(config.hidden_size, device=device),
        }

        layers.append(
            {
                "attention": attention_params,
                "feed_forward": feed_forward_params,
                "ln_1": ln_1_params,
                "ln_2": ln_2_params,
            }
        )

        ln_params: Layer_Norm_Params = {
            "weight": torch.ones(config.hidden_size, device=device),
            "bias": torch.zeros(config.hidden_size, device=device),
        }

    return {
        "layers": layers,
        "ln": ln_params,
    }


def init_worldformer(
    config: config.Worldformer_Config,
    bert_path: pathlib.Path,
    gpt2_path: pathlib.Path,
    device: torch.device = DEVICE,
) -> Worldformer_Params:
    text_encoder = load_bert_weights(config.encoder_config, bert_path, device=device)
    graph_encoder = copy.deepcopy(text_encoder)

    action_decoder = load_gpt2_weights(config.decoder_config, gpt2_path, device=device)
    graph_decoder = copy.deepcopy(action_decoder)

    return {
        "text_encoder": text_encoder,
        "graph_encoder": graph_encoder,
        "aggregator": init_aggregator_weights(config.aggregator_config, device),
        "action_decoder": action_decoder,
        "graph_decoder": graph_decoder,
    }
