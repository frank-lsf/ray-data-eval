"""
Testing Ray Data's streaming execution.

Ray cluster setting:

ray stop -f && ray start --head --num-gpus=200
"""

import random
import time

import numpy as np
import ray
import wandb


DATA_SIZE_BYTES = 1000 * 1000 * 1  # 1 MB
DATA_SIZE = DATA_SIZE_BYTES // 8  # 8 bytes per float64
CONSUMER_WORK = 1  # How much work the consumer does


def memory_blowup(row, *, time_skew: bool = False, space_skew: bool = False):
    """
    For task i, this task will:
    - Return i MB of data (if `space_skew`)
    - Take i times longer to execute than the first task (if `time_skew`)
    """
    i = row["item"]
    data_size = DATA_SIZE * i if space_skew else DATA_SIZE
    if time_skew:
        data = sum(np.random.rand(data_size) for _ in range(i + 1))
    else:
        data = np.random.rand(data_size)
    return {"data": data, "idx": i}


def memory_shrink(row, *, time_skew: bool = False):
    data_size = row["data"].size
    data = sum(np.random.rand(data_size) for _ in range(CONSUMER_WORK))
    if time_skew:
        raise NotImplementedError("Time skew not implemented yet")
    return {"result": data.sum()}


def run_experiment(
    *,
    parallelism: int = -1,
    num_parts: int = 100,
    producer_time_skew: bool = False,
    producer_space_skew: bool = False,
    consumer_time_skew: bool = False,
    random_shuffle: bool = True,
):
    start = time.perf_counter()

    items = list(range(num_parts))
    if random_shuffle:
        random.shuffle(items)
    ds = ray.data.from_items(items, parallelism=parallelism)

    ds = ds.map(
        memory_blowup,
        fn_kwargs={
            "time_skew": producer_time_skew,
            "space_skew": producer_space_skew,
        },
    )
    ds = ds.map(memory_shrink, fn_kwargs={"time_skew": consumer_time_skew}, num_gpus=1)

    ret = 0
    for row in ds.iter_rows():
        # print(f"Time: {time.perf_counter() - start:.4f}s")
        # print(f"Row: {row}")
        ret += row["result"]

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    wandb.log({"run_time": run_time})
    return ret


def main():
    ray.init("auto")

    wandb.init(project="ray-data-eval", entity="raysort")

    config = {
        "kind": "nonuniform",  # Each task runs for the same amount of time
        "parallelism": -1,
        "num_parts": 400,
        "producer_time_skew": False,
        "producer_space_skew": True,
        "consumer_time_skew": False,
    }
    config["data_size"] = DATA_SIZE_BYTES * config["num_parts"]
    config["data_size_gb"] = config["data_size"] / 10**9
    wandb.config.update(config)

    run_experiment(
        parallelism=config["parallelism"],
        num_parts=config["num_parts"],
        producer_time_skew=config["producer_time_skew"],
        producer_space_skew=config["producer_space_skew"],
        consumer_time_skew=config["consumer_time_skew"],
    )


if __name__ == "__main__":
    main()
