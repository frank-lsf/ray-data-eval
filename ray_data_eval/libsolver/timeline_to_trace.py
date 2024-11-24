import argparse
import json
from typing import Iterator

NUM_CPUS = 8
NUM_GPUS = 4
OPERATORS = ["P", "C", "I"]

COLOR_MAP = {
    "P": "rail_response",
    "C": "cq_build_passed",
    "I": "rail_load",
    "T": "cq_build_failed",
}

OPERATOR_TIME = {
    "P": 8,
    "C": 4,
    "I": 1,
}

MICROSECS_PER_SEC = 1e6


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    return parser.parse_args()


def get_events_from_line(line: str, executor_name: str) -> Iterator[dict]:
    tick = 0
    n = len(line)
    i = 0
    while i < n:
        op = line[i]
        if op != " ":
            dur = OPERATOR_TIME.get(op, 0)
            print(op, tick, dur)
            yield {
                "cat": "task",
                "name": op,
                "pid": 1,
                "tid": executor_name,
                "ts": tick * MICROSECS_PER_SEC,
                "dur": dur * MICROSECS_PER_SEC,
                "ph": "X",
                "cname": COLOR_MAP.get(op, "olive"),
            }
            tick += dur
            i += dur * 2
        else:
            print(" ", tick)
            tick += 1
            i += 2


def get_events_from_buffer_line(line: str, i: int) -> Iterator[dict]:
    i = i - NUM_CPUS - NUM_GPUS
    idx = i // 2
    if idx >= len(OPERATORS) - 1:
        return
    upstream_op = OPERATORS[idx]
    downstream_op = OPERATORS[idx + 1]
    executor_name = f"Buffer {upstream_op} -> {downstream_op}"
    if i % 2 == 1:
        executor_name += " (consumable)"
    tick = 0
    for x in line.split():
        yield {
            "cat": "task",
            "name": x,
            "pid": 1,
            "tid": executor_name,
            "ts": tick * MICROSECS_PER_SEC,
            "dur": 1 * MICROSECS_PER_SEC,
            "ph": "X",
            "cname": "grey",
        }
        tick += 1


def get_events(lines: list[str]) -> list[dict]:
    ret = []
    for i, line in enumerate(lines):
        events = []
        if i < NUM_CPUS:
            events = get_events_from_line(line.rstrip(), f"CPU {i}")
        elif i < NUM_CPUS + NUM_GPUS:
            events = get_events_from_line(line.rstrip(), f"GPU {i - NUM_CPUS}")
        else:
            events = get_events_from_buffer_line(line, i)
        ret.extend(events)
        print("--")
    return ret


def main(args: argparse.Namespace):
    with open(args.input_file) as f:
        lines = f.readlines()

    events = get_events(lines)

    output_file = args.input_file.split(".")[0] + ".json"
    with open(output_file, "w") as f:
        json.dump(events, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
