import torch


def tokenizer():
    return transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def prepare_inputs(
    text: str, max_length: int = 512, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = tokenizer()

    inputs = t(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)

    return input_ids, attention_mask, token_type_ids
