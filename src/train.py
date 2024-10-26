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
BATCH_SIZE = 16
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
            w.write(f"Epoch {epoch + 1}/{num_epochs} - Step 0")

            for step, (input, target) in enumerate(batched_data):
                params = update(params, batched_predict, input, target)

                w.write(f"Epoch {epoch + 1}/{num_epochs} - Step {step}")

        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Steps: {steps}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s"
        )


def update(
    params: weights.Worldformer_Params,
    predict: Callable[[Any, Any], torch.Tensor],
    input: preprocess.Input,
    target: preprocess.Target,
) -> weights.Worldformer_Params:
    grads = func.grad(loss, argnums=0)(params, predict, input, target)
    return params


def loss(
    params: weights.Worldformer_Params,
    predict: Callable[[Any, Any], torch.Tensor],
    input: preprocess.Input,
    target: preprocess.Target,
) -> torch.Tensor:
    tokens_input = tokenizer.tokenize_input(input)
    tokens_target = tokenizer.tokenize_target(target)

    action_logits, graph_logits = predict(params, *tokens_input, *tokens_target)

    action_logits = action_logits[..., :-1, :]
    graph_logits = graph_logits[..., :-1, :]

    action_targets, graph_targets, action_mask, graph_mask = tokens_target

    action_targets = action_targets[..., 1:]
    graph_targets = graph_targets[..., 1:]
    action_mask = action_mask[..., 1:]
    graph_mask = graph_mask[..., 1:]

    # ** Action Loss **
    action_loss = F.cross_entropy(
        action_logits.reshape(-1, action_logits.size(-1)),
        action_targets.reshape(-1),
        reduction="none",
    ).reshape_as(action_targets)

    action_loss = (action_loss * action_mask).sum() / action_mask.sum()

    # ** Graph Loss **
    graph_loss = F.cross_entropy(
        graph_logits.reshape(-1, graph_logits.size(-1)),
        graph_targets.reshape(-1),
        reduction="none",
    ).reshape_as(graph_targets)

    graph_loss = (graph_loss * graph_mask).sum() / graph_mask.sum()

    # ** Total Loss **
    total_loss = action_loss + graph_loss

    return total_loss
