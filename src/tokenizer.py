from typing import Tuple

import torch
import transformers

import config
import preprocess

BERT_NEW_TOKENS = ["[ACT]", "[TRIPLE]", "[OBS]", "[GRAPH]"]
GPT2_NEW_TOKENS = ["[ACT]", "[TRIPLE]", "[GRAPH]"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
BERT_TOKENIZER.add_tokens(BERT_NEW_TOKENS, special_tokens=True)

GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained("gpt2")
GPT2_TOKENIZER.add_tokens(GPT2_NEW_TOKENS, special_tokens=True)


def tokenize_input(
    input: preprocess.Input,
    input_length: int = 1024,
    bert_tokenizer: transformers.BertTokenizer = BERT_TOKENIZER,
    device: torch.device = DEVICE,
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


def create_sequence_boundaries(
    tokens: torch.Tensor,
    separator_token_id: int,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    boundaries = torch.zeros_like(tokens, dtype=torch.float32)

    boundaries[:, 0] = 1

    is_separator = tokens == separator_token_id
    boundaries[:, 1:] = is_separator[:, :-1].float()

    return boundaries


def tokenize_target(
    target: preprocess.Target,
    input_length: int = 1024,
    gpt2_tokenizer: transformers.GPT2Tokenizer = GPT2_TOKENIZER,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    action_targets, graph_targets = zip(*target)

    act_token_id = gpt2_tokenizer.convert_tokens_to_ids("[ACT]")
    triple_token_id = gpt2_tokenizer.convert_tokens_to_ids("[TRIPLE]")

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

    action_boundaries = create_sequence_boundaries(
        action_encodings.input_ids, act_token_id, device
    )

    graph_boundaries = create_sequence_boundaries(
        graph_encodings.input_ids, triple_token_id, device
    )

    return (
        action_encodings.input_ids,
        graph_encodings.input_ids,
        action_boundaries,
        graph_boundaries,
    )
