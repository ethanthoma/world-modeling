from typing import Dict, Tuple

import torch
import transformers


def tokenize(
    textual_encoder_input: str,
    graph_encoder_input: str,
    bert_tokenizer: transformers.BertTokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased"
    ),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[Dict[str, torch.Tensor]]:
    textual_encodings = bert_tokenizer(
        textual_encoder_input, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    graph_encodings = bert_tokenizer(
        graph_encoder_input, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    return textual_encodings, graph_encodings
