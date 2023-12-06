"""
Testing Ray Data's streaming execution.

Ray cluster setting:

ray stop -f && ray start --head --num-gpus=8 --object-store-memory=1048576000
"""

import time

import numpy as np
import ray


DATA_SIZE = 1000 * 1000 * 1  # 1 MB


def memory_blowup(row, *, slow: bool = False):
    """
    For task i, this task will:
    - Return i MB of data
    - Take i times longer to execute than the first task (if `slow`)
    """
    i = row["item"]
    if slow:
        data = sum(np.random.rand(DATA_SIZE * i) for _ in range(i + 1))
    else:
        data = np.random.rand(DATA_SIZE * i)
    return {"data": data, "idx": i}


def memory_shrink(row):
    data = row["data"]
    return {"result": data.sum()}


def run_experiment(*, parallelism: int = -1, size: int = 100, blowup: int = 20):
    start = time.perf_counter()

    items = list(range(size))
    ds = ray.data.from_items(items, parallelism=parallelism)

    ds = ds.map(memory_blowup)
    ds = ds.map(memory_shrink, num_gpus=1)

    ret = 0
    for row in ds.iter_rows():
        print(f"Time: {time.perf_counter() - start:.4f}s")
        print(f"Row: {row}")
        ret += row["result"]

    end = time.perf_counter()
    print(f"\n{ret:,}")
    print(f"Time: {end - start:.4f}s")
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    return ret


def main():
    ray.init("auto")
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    run_experiment(size=100)


if __name__ == "__main__":
    main()
