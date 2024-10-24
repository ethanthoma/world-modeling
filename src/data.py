import json
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypedDict,
)


class LocationDict(TypedDict):
    name: str
    num: int


class StateDict(TypedDict):
    walkthrough_act: str
    walkthrough_diff: str
    obs: str
    loc_desc: str
    inv_desc: str
    inv_objs: Dict[str, str]
    inv_attrs: Dict[str, List[str]]
    location: LocationDict
    surrounding_objs: Dict[str, List[str]]
    surrounding_attrs: Dict[str, List[str]]
    graph: List[List[str]]  # List of [subject, relation, object] triples
    valid_acts: Dict[str, str]
    score: int


class JerichoSample(TypedDict):
    rom: str
    state: StateDict
    next_state: StateDict
    graph_diff: List[List[str]]  # List of [subject, relation, object] triples
    action: str
    reward: int


class Input(NamedTuple):
    textual_observations: str
    valid_actions: Set[Tuple[str, str]]
    graph: Set[Tuple[str, str, str]]


class Target(NamedTuple):
    valid_actions_next: Set[Tuple[str, str]]
    graph_additions: Set[Tuple[str, str, str]]


def data_generator(
    filepath: str, chunk_size: int = 4096
) -> Iterable[Tuple[str, str, str]]:
    return map(preprocess, json_generator(filepath=filepath, chunk_size=chunk_size))


def preprocess(sample: JerichoSample) -> Tuple[Input, str]:
    X = Input(
        textual_observations=sample["state"]["obs"],
        valid_actions={(k, v) for k, v in sample["state"]["valid_acts"].items()},
        graph={(s, r, o) for s, r, o in sample["state"]["graph"]},
    )

    y = Target(
        valid_actions_next={
            (k, v) for k, v in sample["next_state"]["valid_acts"].items()
        },
        graph_additions=knowledge_graph_additions(sample),
    )

    return X, y


def knowledge_graph_additions(sample: JerichoSample) -> Set[Tuple[str, str, str]]:
    current_graph = {(s, r, o) for s, r, o in sample["state"]["graph"]}
    next_graph = {(s, r, o) for s, r, o in sample["next_state"]["graph"]}

    return next_graph - current_graph


def json_generator(filepath: str, chunk_size: int) -> Iterable[JerichoSample]:
    with open(filepath, "r") as f:
        buffer = ""
        chunk = f.read(2)

        while True:
            result, buffer = find_complete_json(buffer)

            if result is not None:
                yield result
            else:
                chunk = f.read(chunk_size)

                if not chunk:
                    break  # end of file

                buffer += chunk


REQUIRED_KEYS = {k for k in JerichoSample.__annotations__}


def find_complete_json(buffer: str) -> Tuple[Optional[str], str]:
    bracket_stack = []
    in_string = False
    current_key = None
    found_keys = set()
    json_start = -1
    brace_count = 0
    token = ""

    for i, char in enumerate(buffer):
        match char:
            case '"':
                if len(token) != 0 and token[-1] == "\\":
                    token += char
                    continue

                if not in_string:
                    token = ""

                in_string = not in_string
            case _:
                if in_string:
                    token += char

        if not in_string:
            match char:
                case "{":
                    brace_count += 1
                    bracket_stack.append(char)

                    if brace_count == 1:
                        json_start = i
                case "}":
                    if not bracket_stack:
                        return None, buffer
                    if bracket_stack[-1] == "{":
                        bracket_stack.pop()
                        brace_count -= 1
                    else:
                        return None, buffer
                case ":":
                    if brace_count == 1:
                        found_keys.add(token)
                    token = ""

            if not bracket_stack and brace_count == 0 and i > 0:
                if found_keys >= REQUIRED_KEYS:
                    try:
                        parsed_json = json.loads(buffer[json_start : i + 1])
                        return parsed_json, buffer[i + 1 :]
                    except json.JSONDecodeError:
                        return None, buffer
                else:
                    return None, buffer[
                        i + 1 :
                    ]  # Skip this object if it doesn't have all required keys

    return None, buffer
