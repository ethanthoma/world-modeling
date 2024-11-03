import itertools
import pathlib

import config
import data
import preprocess
import tokenizer


def input_pipeline(
    data_path: pathlib.Path,
    batch_size: int,
    bert_config: config.BERT_Config,
    gpt2_config: config.GPT2_Config,
):
    raw_data = data.data(data_path)

    preprocessed_data = map(preprocess.preprocess, raw_data)

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
