import functools
import itertools
import pathlib
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
import torch.func as func
import torch.jit as jit
import torch.nn.functional as F

import config
import data
import model
import preprocess
import spinner
import tokenizer
import weights

NUM_EPOCHS = 5
BATCH_SIZE = 1
BERT_PATH = pathlib.Path("./weights/bert.bin")
GPT2_PATH = pathlib.Path("./weights/gpt2.bin")
TRAIN_DATA_PATH = pathlib.Path("./data/jericho-world/train.json")


def train(num_epochs: int = NUM_EPOCHS) -> None:
    # ** Dataset **
    raw_data = data.data(TRAIN_DATA_PATH)

    preprocessed_data = map(preprocess.preprocess, raw_data)

    batched_data = itertools.starmap(
        zip, itertools.batched(preprocessed_data, BATCH_SIZE)
    )

    # ** Model **
    params = weights.init_worldformer(config.WORLDFORMER_CONFIG, BERT_PATH, GPT2_PATH)

    batched_predict = func.vmap(model.worldformer, in_dims=(None, *([0] * 8)))

    # ** Training Loop **
    for epoch in range(num_epochs):
        start_time = time.time()
        step = None

        total_loss = 0

        with spinner.spinner() as w:
            w.write(f"Epoch: {epoch + 1}/{num_epochs} | Step: 0")

            for step, (input, target) in enumerate(batched_data):
                params, loss = update(params, batched_predict, input, target)

                total_loss += loss

                w.write(
                    f"Epoch: {epoch + 1}/{num_epochs} | Step: {step} | Loss: {loss}"
                )

        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Total Steps: {steps} | Avg. Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s"
        )


def update(
    params: weights.Worldformer_Params,
    predict: Callable[[Any, Any], torch.Tensor],
    input: preprocess.Input,
    target: preprocess.Target,
    lr: float = 3e-4,
) -> Tuple[weights.Worldformer_Params, torch.Tensor]:
    grads, loss_value = func.grad_and_value(loss, argnums=0)(
        params, predict, input, target
    )

    def update_params(p: dict, g: dict) -> dict:
        updated = {}
        for k, v in p.items():
            if isinstance(v, torch.Tensor):
                updated[k] = v - lr * g[k]
            elif isinstance(v, dict):
                updated[k] = update_params(v, g[k])
            elif isinstance(v, list):
                updated[k] = [
                    update_params(layer, g_layer) for layer, g_layer in zip(v, g[k])
                ]
            else:
                updated[k] = v
        return updated

    return update_params(params, grads), loss_value.item()


def loss(
    params: weights.Worldformer_Params,
    predict: Callable[[Any, Any], torch.Tensor],
    input: preprocess.Input,
    target: preprocess.Target,
) -> torch.Tensor:
    tokens_input = tokenizer.tokenize_input(input)
    tokens_target = tokenizer.tokenize_target(target)

    action_logits, graph_logits = predict(params, *tokens_input, *tokens_target)

    action_targets, graph_targets, action_boundaries, graph_boundaries = tokens_target

    action_logits = action_logits[..., :-1, :]
    graph_logits = graph_logits[..., :-1, :]

    action_targets = action_targets[..., 1:]
    graph_targets = graph_targets[..., 1:]
    action_boundaries = action_boundaries[..., 1:]
    graph_boundaries = graph_boundaries[..., 1:]

    action_loss = compute_sos_loss(action_logits, action_targets, action_boundaries)
    graph_loss = compute_sos_loss(graph_logits, graph_targets, graph_boundaries)

    total_loss = action_loss + graph_loss

    return total_loss.squeeze()


def compute_sos_loss(
    logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    targets: torch.Tensor,  # [batch_size, seq_len]
    boundaries: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:
    batch_size = logits.size(0)
    device = logits.device

    attention_mask = (targets != tokenizer.GPT2_TOKENIZER.pad_token_id).float()

    total_loss = torch.zeros(1, device=device)
    total_sequences = 0

    for b in range(batch_size):
        seq_starts = torch.where(boundaries[b] == 1)[0]

        if len(seq_starts) == 0:
            continue

        seq_starts = torch.cat(
            [seq_starts, torch.tensor([targets.size(1)], device=device)]
        )

        for i in range(len(seq_starts) - 1):
            start_idx = seq_starts[i]
            end_idx = seq_starts[i + 1]

            seq_logits = logits[b, start_idx:end_idx]
            seq_targets = targets[b, start_idx:end_idx]
            seq_mask = attention_mask[b, start_idx:end_idx]

            if seq_mask.sum() == 0:
                continue

            seq_loss = F.cross_entropy(
                seq_logits.view(-1, seq_logits.size(-1)),
                seq_targets.view(-1),
                reduction="none",
                ignore_index=tokenizer.GPT2_TOKENIZER.pad_token_id,
            )

            valid_tokens = seq_mask.sum()
            if valid_tokens > 0:
                seq_loss = (seq_loss * seq_mask).sum() / valid_tokens
                total_loss += seq_loss
                total_sequences += 1

    return total_loss / max(total_sequences, 1)


def get_sequence_mask(
    token_ids: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_length = token_ids.size()
    device = token_ids.device

    mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    mask = mask.view(1, seq_length, seq_length).expand(batch_size, -1, -1)

    for b in range(batch_size):
        seq_starts = torch.where(boundaries[b] == 1)[0]

        if len(seq_starts) == 0:
            continue

        seq_starts = torch.cat([seq_starts, torch.tensor([seq_length], device=device)])

        for i in range(len(seq_starts) - 1):
            start_idx = seq_starts[i]
            end_idx = seq_starts[i + 1]

            mask[b, start_idx:end_idx, :start_idx] = 0
            mask[b, start_idx:end_idx, end_idx:] = 0

    return mask * (token_ids != tokenizer.GPT2_TOKENIZER.pad_token_id).unsqueeze(1)
