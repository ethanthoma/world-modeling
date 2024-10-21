import itertools
import pathlib

import torch
import torch.func as func
import torch.jit as jrt

import config
import data
import weights

BATCH_SIZE = 32
BERT_BASE_PATH = pathlib.Path("./weights/bert.bin")


def main():
    bert_params = weights.load_bert_weights(BERT_BASE_PATH, config.BERT_BASE_CONFIG)

    data_generator = itertools.batched(
        data.data_generator("data/jericho-world/train.json"), BATCH_SIZE
    )
