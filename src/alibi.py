import functools
import math

import torch


@functools.cache
def get_slopes(n: int) -> torch.Tensor:
    def get_slopes_power_of_2(n: int) -> torch.Tensor:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return torch.tensor([start * ratio**i for i in range(n)])

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return torch.cat(
            [
                get_slopes_power_of_2(closest_power_of_2),
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ],
            ]
        )


@functools.cache
def build_alibi_bias(
    num_attention_heads: int, max_seq_len: int, device: torch.device
) -> torch.Tensor:
    slopes = get_slopes(num_attention_heads)
    slopes = slopes.to(device)

    pos = torch.arange(max_seq_len, device=device)
    diff = pos[None, :] - pos[:, None]  # [seq_len, seq_len]

    alibi = slopes[:, None, None] * diff[None, :, :]  # [num_heads, seq_len, seq_len]

    return alibi
