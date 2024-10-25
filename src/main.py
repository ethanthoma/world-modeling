import functools
import itertools
import pathlib

import torch

import config
import data
import preprocess
import tokenizer
import weights

BATCH_SIZE = 1
BERT_PATH = pathlib.Path("./weights/bert.bin")
GPT2_PATH = pathlib.Path("./weights/gpt2.bin")
TRAIN_DATA_PATH = pathlib.Path("./data/jericho-world/train.json")


def main():
    # ** Model **
    bert_params = weights.load_bert_weights(BERT_PATH, config.BERT_CONFIG)

    gpt2_params = weights.load_gpt2_weights(GPT2_PATH, config.GPT2_CONFIG)

    # ** Data **
    raw_data = data.data(TRAIN_DATA_PATH)
    preprocessed_data = map(preprocess.preprocess, raw_data)

    batched_data = itertools.starmap(
        zip, itertools.batched(preprocessed_data, BATCH_SIZE)
    )

    for X, y in batched_data:
        tokens = itertools.starmap(tokenizer.tokenize, X)

        return
