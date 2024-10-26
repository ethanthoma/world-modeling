import functools
import itertools
import pathlib

import torch
import torch.func as func

import config
import data
import model
import preprocess
import tokenizer
import weights

BATCH_SIZE = 1
BERT_PATH = pathlib.Path("./weights/bert.bin")
GPT2_PATH = pathlib.Path("./weights/gpt2.bin")
TRAIN_DATA_PATH = pathlib.Path("./data/jericho-world/train.json")
INPUT_LENGTH = 1024


def main():
    # ** Model **
    params = weights.init_worldformer(config.WORLDFORMER_CONFIG, BERT_PATH, GPT2_PATH)

    batched_worldformer = func.vmap(model.worldformer, in_dims=(None, *([0] * 8)))

    # ** Data **
    raw_data = data.data(TRAIN_DATA_PATH)

    preprocessed_data = map(preprocess.preprocess, raw_data)

    batched_data = itertools.starmap(
        zip, itertools.batched(preprocessed_data, BATCH_SIZE)
    )

    for X, y in batched_data:
        tokens_input = tokenizer.tokenize_input(X, INPUT_LENGTH)
        tokens_target = tokenizer.tokenize_target(y, INPUT_LENGTH)

        batched_worldformer(params, *tokens_input, *tokens_target)

        return
