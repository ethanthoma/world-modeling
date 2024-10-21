import json
from typing import Any, Dict, Optional, Tuple

REQUIRED_KEYS = {"state", "next_state", "action", "reward", "rom"}


def data_generator(filepath: str, chunk_size: int = 4096) -> Any:
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
