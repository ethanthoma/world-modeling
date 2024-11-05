import dataclasses
import functools
import itertools
import math
import pathlib
import random

import config
import data
import preprocess
import tokenizer


@dataclasses.dataclass
class Split_State:
    train_count: int = 0
    valid_count: int = 0
    previous_p: float = 0.0

    @property
    def total_count(self):
        return self.train_count + self.valid_count

    @property
    def current_ratio(self):
        return 0.0 if self.total_count == 0 else self.valid_count / self.total_count


def input_pipeline(
    data_path: pathlib.Path,
    batch_size: int,
    bert_config: config.BERT_Config,
    gpt2_config: config.GPT2_Config,
    target_ratio: float = 0.1,
    seed: int = 1453,
):
    split_state = Split_State(previous_p=target_ratio)

    raw_data = data.data(data_path)

    train_set, valid_set = split(
        raw_data,
        split_state,
        target_ratio,
        seed,
    )

    def preprocessing(iterable):
        preprocessed_data = map(preprocess.preprocess, iterable)

        batched_data = itertools.starmap(
            zip, itertools.batched(preprocessed_data, batch_size)
        )

        tokenized_data = itertools.starmap(
            lambda x, y: (
                tokenizer.tokenize_input(bert_config, x),
                tokenizer.tokenize_target(gpt2_config, y),
            ),
            batched_data,
        )

        return tokenized_data

    return (lambda: preprocessing(train_set()), lambda: preprocessing(valid_set()))


def split(iterable, split_state: Split_State, target_ratio: float, seed: int):
    iter1, iter2 = itertools.tee(iterable)

    def iter_factory():
        nonlocal iter1, iter2
        iter1, iter2 = itertools.tee(iter2)
        return iter1

    train_set = []
    valid_set = []

    is_empty = False
    current_iter = iter_factory()

    sample = lambda p: p > split_ratio

    def _split(seed):
        nonlocal is_empty, current_iter

        rnd = random.Random(seed)

        for element in current_iter:
            p = rnd.uniform(0, 1)
            pred = p > split_probablity(split_state, target_ratio)

            if pred:
                train_set.append(element)
                split_state.train_count += 1
            else:
                valid_set.append(element)
                split_state.valid_count += 1

            yield

        is_empty = True

    gen = _split(seed)

    def reset():
        nonlocal train_set, valid_set, is_empty, gen, current_iter
        train_set.clear()
        valid_set.clear()
        is_empty = False
        current_iter = iter_factory()
        gen = _split(seed)

    def dataset(data: list):
        nonlocal is_empty

        while True:
            if is_empty and len(data) == 0:
                reset()
            try:
                while len(data) == 0 and not is_empty:
                    next(gen)
                if len(data) > 0:
                    yield data.pop()
                else:
                    break
            except StopIteration:
                if len(data) > 0:
                    yield data.pop()
                else:
                    break

    return (lambda: dataset(train_set), lambda: dataset(valid_set))


def split_probablity(
    split_state: Split_State,
    target_ratio: float,
    gain_factor: float = 0.01,
) -> float:
    p_req = (target_ratio * (split_state.total_count + 1)) - split_state.valid_count

    direction = p_req - target_ratio

    p = split_state.previous_p + gain_factor * direction

    p = max(0, min(1, p))

    return p
