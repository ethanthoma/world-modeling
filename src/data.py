import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypedDict


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


def data_generator(
    filepath: str, chunk_size: int = 4096
) -> Iterable[Tuple[str, str, str]]:
    return map(preprocess, json_generator(filepath=filepath, chunk_size=chunk_size))


def preprocess(sample: JerichoSample) -> Tuple[str, str, str]:
    observation = sample["state"]["obs"]
    valid_actions = list(sample["state"]["valid_acts"].values())
    current_graph = sample["state"]["graph"]
    next_graph_diff = sample["graph_diff"]
    next_valid_actions = list(sample["next_state"]["valid_acts"].values())

    input_sequence = f"[OBS] {observation} [ACT] {' [ACT] '.join(valid_actions)} [GRAPH] {' '.join([' '.join(triple) for triple in current_graph])}"

    graph_target = (
        f"[GRAPH] {' '.join([' '.join(triple) for triple in next_graph_diff])}"
    )
    action_target = f"[ACT] {' [ACT] '.join(next_valid_actions)}"

    return input_sequence, graph_target, action_target


def encode_text(obs: str, valid_actions: Dict[str, str]):
    return obs + " [SEP] " + " [ACT] ".join(valid_actions.keys())


def encode_graph(graph: List[Tuple[str, str, str]], max_graph_length: int = 1024):
    encoded = []
    for s, r, o in graph:
        if len(encoded) >= max_graph_length:
            break


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


REQUIRED_KEYS = {k for k in data.JerichoSample.__annotations__}


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
