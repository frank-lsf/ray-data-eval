import time

import numpy as np
import ray
import wandb


DATA_SIZE = 1000 * 100


def gen_data(row):
    return {"data": np.random.rand(DATA_SIZE), **row}


def memory_blowup(row, *, blowup: int):
    print(f"memory_blowup({blowup})", row["id"])
    x = row["data"]
    return {"data": np.concatenate([x + np.random.rand(DATA_SIZE) for _ in range(blowup)])}


def memory_blowup_flat(row, *, blowup: int):
    x = row["data"]
    return [{"data": x + np.random.rand(DATA_SIZE)} for _ in range(blowup)]


def run_experiment(*, blowup: int = 0, parallelism: int = -1, size: int = -1):
    start = time.perf_counter()

    ds = ray.data.range(size, parallelism=parallelism)
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
    wandb.log({
        "Execution Time": end - start, 
        "Data Size": DATA_SIZE, 
        "Blowup Factor": blowup, 
        "Parallelism": parallelism, 
        "Dataset Size": size, 
        "Memory Used": ret
        })
    return ret


def main():
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    wandb.init(project='gen.py', entity='ronyw') # raysort 

    # run_experiment(parallelism=-1, size=10000, blowup=20)
    run_experiment(parallelism=2, size=100, blowup=20)


if __name__ == "__main__":
    main()
