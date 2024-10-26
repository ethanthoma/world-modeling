import itertools
from typing import Iterable, NamedTuple, Set, Tuple

import data


class Input(NamedTuple):
    textual_encoder_input: str
    graph_encoder_input: str


class Target(NamedTuple):
    valid_actions_next: Set[Tuple[str, str]]
    graph_additions: Set[Tuple[str, str, str]]


def preprocess(sample: data.JerichoSample) -> Tuple[Input, Target]:
    input = input_from_sample(sample)
    target = target_from_sample(sample)

    return input, target


def input_from_sample(sample: data.JerichoSample) -> Input:
    textual_observations = sample["state"]["obs"]
    valid_actions = {(k, v) for k, v in sample["state"]["valid_acts"].items()}
    graph = {(s, r, o) for s, r, o in sample["state"]["graph"]}

    valid_actions_string = " ".join(
        itertools.starmap(lambda _, act: f"[ACT] {act}", valid_actions)
    )
    textual_encoding = f"[OBS] {textual_observations} {valid_actions_string}"

    graph_string = " [TRIPLE]".join(
        itertools.starmap(lambda s, r, o: f" {s}, {r}, {o}", graph)
    )
    graph_encoding = f"[GRAPH]{graph_string}"

    return Input(textual_encoding, graph_encoding)


def target_from_sample(sample: data.JerichoSample) -> Target:
    valid_actions_next = {(k, v) for k, v in sample["next_state"]["valid_acts"].items()}

    current_graph = {(s, r, o) for s, r, o in sample["state"]["graph"]}
    next_graph = {(s, r, o) for s, r, o in sample["next_state"]["graph"]}
    graph_additions = next_graph - current_graph

    actions_encoding = " ".join(
        itertools.starmap(lambda _, act: f"[ACT] {act}", valid_actions_next)
    )

    graph_string = " [TRIPLE]".join(
        itertools.starmap(lambda s, r, o: f" {s}, {r}, {o}", graph_additions)
    )
    graph_encoding = f"[GRAPH]{graph_string}"

    return Target(actions_encoding, graph_encoding)
