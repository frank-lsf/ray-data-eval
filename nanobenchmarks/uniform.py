"""
Testing Ray Data's streaming execution.

Ray cluster setting:

ray stop -f && ray start --head --num-gpus=200
"""

import time

import numpy as np
import ray
import wandb


DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_BASIS = 0.1  # How many seconds should time_factor=1 take


def memory_blowup(row, *, time_factor: int = 1):
    i = row["item"]
    data = np.full(DATA_SIZE_BYTES, 1, dtype=np.uint8)
    time.sleep(TIME_BASIS * time_factor)
    return {"data": data, "idx": i}


def memory_shrink(row, *, time_factor: int = 1):
    data = row["data"]
    time.sleep(TIME_BASIS * time_factor)
    return {"result": data.sum()}


def run_experiment(
    *,
    parallelism: int = -1,
    num_parts: int = 100,
    producer_time: int = 1,
    consumer_time: int = 1,
):
    start = time.perf_counter()

    items = list(range(num_parts))
    ds = ray.data.from_items(items, parallelism=parallelism)

    ds = ds.map(memory_blowup, fn_kwargs={"time_factor": producer_time})
    ds = ds.map(memory_shrink, fn_kwargs={"time_factor": consumer_time}, num_gpus=1)

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


def config_ray_data(config):
    ctx = ray.data.DataContext.get_current()
    for k, v in config.items():
        ctx.set_config(k, v)


def main():
    ray.init("auto")

    wandb.init(project="ray-data-eval", entity="raysort")

    config = {
        "kind": "uniform",  # Each task runs for the same amount of time
        "parallelism": 100,
        "total_data_size_gb": 100,
        "producer_time": 1,
        "consumer_time": 9,
        "ray_data_config": {
            "backpressure_policies.enabled": [],
        },
    }
    config["total_data_size"] = config["total_data_size_gb"] * 10**9
    config["num_parts"] = config["total_data_size"] // DATA_SIZE_BYTES
    config["producer_consumer_ratio"] = (
        config["producer_time"] / config["consumer_time"]
    )
    wandb.config.update(config)
    config_ray_data(config.get("ray_data_config", {}))

    run_experiment(
        parallelism=config["parallelism"],
        num_parts=config["num_parts"],
        producer_time=config["producer_time"],
        consumer_time=config["consumer_time"],
    )


if __name__ == "__main__":
    main()
