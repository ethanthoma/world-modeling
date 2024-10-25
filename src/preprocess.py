import itertools
from typing import Iterable, NamedTuple, Set, Tuple

import data


class Input(NamedTuple):
    textual_observations: str
    valid_actions: Set[Tuple[str, str]]
    graph: Set[Tuple[str, str, str]]


class Target(NamedTuple):
    valid_actions_next: Set[Tuple[str, str]]
    graph_additions: Set[Tuple[str, str, str]]


def preprocess(sample: data.JerichoSample) -> Tuple[Tuple[str, str], Target]:
    input = Input(
        textual_observations=sample["state"]["obs"],
        valid_actions={(k, v) for k, v in sample["state"]["valid_acts"].items()},
        graph={(s, r, o) for s, r, o in sample["state"]["graph"]},
    )
    input = to_text(input)

    target = Target(
        valid_actions_next={
            (k, v) for k, v in sample["next_state"]["valid_acts"].items()
        },
        graph_additions=knowledge_graph_additions(sample),
    )

    return input, target


def knowledge_graph_additions(sample: data.JerichoSample) -> Set[Tuple[str, str, str]]:
    current_graph = {(s, r, o) for s, r, o in sample["state"]["graph"]}
    next_graph = {(s, r, o) for s, r, o in sample["next_state"]["graph"]}

    return next_graph - current_graph


def to_text(input: Input) -> Tuple[str, str]:
    valid_actions_string = " ".join(
        itertools.starmap(lambda _, act: f"[ACT] {act}", input.valid_actions)
    )
    textual_encoder_input = f"[OBS] {input.textual_observations} {valid_actions_string}"

    graph_string = " [TRIPLE]".join(
        itertools.starmap(lambda s, r, o: f" {s}, {r}, {o}", input.graph)
    )
    graph_encoder_input = f"[GRAPH]{graph_string}"

    return textual_encoder_input, graph_encoder_input
