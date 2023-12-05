import time

import numpy as np
import ray


DATA_SIZE = 1000 * 200


def gen_data(row):
    i = row["id"]
    print(f"gen_data {i}: {DATA_SIZE * i:,}")
    data = np.random.rand(DATA_SIZE * i)
    for _ in range(i):
        data += np.random.rand(DATA_SIZE * i)
    return {"data": data, **row}


def memory_blowup(row, *, blowup: int):
    i = row["id"]
    x = row["data"]
    print(f"memory_blowup {i}: {x.nbytes:,} * {blowup} = {x.nbytes * blowup:,}")
    return {"data": np.concatenate([x + np.random.rand(DATA_SIZE * i * i) for _ in range(blowup)])}


def run_experiment(*, blowup: int = 0, parallelism: int = -1, size: int = -1, random_shuffle: bool = False):
    start = time.perf_counter()

    ds = ray.data.range(size, parallelism=parallelism)
    if random_shuffle:
        ds = ds.random_shuffle()
    ds = ds.map(gen_data)
    if blowup > 0:
        ds = ds.map(memory_blowup, fn_kwargs={"blowup": blowup})
        # ds = ds.flat_map(memory_blowup_flat, fn_kwargs={"blowup": blowup})

    ret = 0
    for row in ds.iter_rows():
        ret += row["data"].nbytes

    end = time.perf_counter()
    print(f"\n{ret:,}")
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    ray.init("auto")
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    # run_experiment(parallelism=4, size=100, blowup=0)
    run_experiment(parallelism=4, size=100, blowup=0, random_shuffle=True)


if __name__ == "__main__":
    main()
