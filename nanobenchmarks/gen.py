import time

import numpy as np
import ray
from ray.data.aggregate import AggregateFn


DATA_SIZE = 1000 * 100


def gen_data(_):
    return {"data": np.random.rand(DATA_SIZE)}


def memory_blowup(row, *, blowup: int):
    x = row["data"]
    return {"data": np.concatenate([x + np.random.rand(DATA_SIZE) for _ in range(blowup)])}


def run_experiment(*, blowup: int = 0, parallelism: int = -1, size: int = -1):
    start = time.perf_counter()

    ds = ray.data.range(size, parallelism=parallelism)
    ds = ds.map(gen_data)
    if blowup > 0:
        ds = ds.map(memory_blowup, fn_kwargs={"blowup": blowup})
    ret = ds.aggregate(
        AggregateFn(
            init=lambda _col: 0,
            accumulate_row=lambda a, row: a + row["data"].nbytes,
            merge=lambda a1, a2: a1 + a2,
            name="nbytes",
        )
    )["nbytes"]

    end = time.perf_counter()
    print(f"\n{ret:,}")
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    ray.init()
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    run_experiment(parallelism=16, size=10000, blowup=2)


if __name__ == "__main__":
    main()
