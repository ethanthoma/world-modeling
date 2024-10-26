from typing import Tuple

import torch
import transformers

import config
import preprocess


def tokenize_input(
    input: preprocess.Input,
    input_length: int = 1024,
    bert_tokenizer: transformers.BertTokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased"
    ),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    textual_inputs, graph_inputs = zip(*input)

    textual_encodings = bert_tokenizer(
        list(textual_inputs),
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    graph_encodings = bert_tokenizer(
        list(graph_inputs),
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    return (
        textual_encodings.input_ids,
        graph_encodings.input_ids,
        textual_encodings.attention_mask,
        graph_encodings.attention_mask,
    )


def tokenize_target(
    target: preprocess.Target,
    input_length: int = 1024,
    gpt2_tokenizer: transformers.GPT2Tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        "gpt2"
    ),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    action_targets, graph_targets = zip(*target)

    action_encodings = gpt2_tokenizer(
        list(action_targets),
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    graph_encodings = gpt2_tokenizer(
        list(graph_targets),
        max_length=input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    return (
        action_encodings.input_ids,
        graph_encodings.input_ids,
        action_encodings.attention_mask,
        graph_encodings.attention_mask,
    )
