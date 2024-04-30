import os
import time

import numpy as np

import ray
from ray.data.block import DataBatch

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

NUM_FILES = 100
NUM_SLICES_PER_FILE = 4
CONSUMER_BATCH_SIZE = 8

FILE_SIZE = 5 * MB
DECODED_FILE_SIZE = 100 * MB

TICK = 0.5
LOAD_TIME = 1 * TICK
TRANSFORM_TIME = 10 * TICK
CONSUMER_TIME = 1 * TICK


def load(_batch: DataBatch) -> list[DataBatch]:
    time.sleep(LOAD_TIME)
    return [
        {
            "data": np.zeros(FILE_SIZE, dtype=np.uint8),
        }
        for _ in range(NUM_SLICES_PER_FILE)
    ]


def transform(batch: DataBatch) -> DataBatch:
    time.sleep(TRANSFORM_TIME)
    return {
        "data": np.zeros(DECODED_FILE_SIZE, dtype=np.uint8),
    }


class Consumer:
    def __call__(self, _batch: DataBatch) -> DataBatch:
        time.sleep(CONSUMER_TIME)
        return {"label": "test"}


def benchmark():
    ray.init(num_cpus=8, num_gpus=1)

    ds = ray.data.range(NUM_FILES)
    ds = ds.flat_map(load)
    ds = ds.for_each(transform)
    ds = ds.map_batches(
        Consumer,
        batch_size=CONSUMER_BATCH_SIZE,
        num_gpus=1,
        concurrency=1,
        zero_copy_batch=True,
        max_concurrency=2,
    )
    ds.take_all()

    ray.timeline("three_stage_problem.json")


if __name__ == "__main__":
    benchmark()
