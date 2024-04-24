import time

import numpy as np
import ray

Batch = dict[str, np.ndarray]

NUM_ROWS = 10000
BATCH_SIZE = 1000


def preprocess(i) -> Batch:
    return {"transformed_image": np.zeros((3, 224, 224), dtype=np.uint8)}


def preprocess_batch(batch: Batch) -> Batch:
    batch_size = len(next(iter(batch.values())))
    return {"transformed_image": np.zeros((batch_size, 3, 224, 224), dtype=np.uint8)}


class Model:
    def __call__(self, batch: Batch):
        batch_size = len(next(iter(batch.values())))
        time.sleep(1)
        with open("timing.csv", "a") as fout:
            fout.write(f"{time.time()}\n")
        return {
            "predicted_label": ["dummy_label"] * batch_size,
        }


def run_iter_batches(ds):
    model = Model()
    start_time = time.time()
    last_batch_time = time.time()
    for batch in ds.iter_batches(batch_size=BATCH_SIZE):
        model(batch)
        print(f"Total batch time: {time.time() - last_batch_time:.4f}")
        last_batch_time = time.time()
    print(f"Total time: {time.time() - start_time:.4f}")
    print(ds.stats())
    ray.timeline("iter_batches.json")


def run_map_batches(ds):
    ds = ds.map_batches(
        Model,
        concurrency=1,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        max_concurrency=2,
    )
    start_time = time.time()
    ds.take_all()
    print(f"Total time: {time.time() - start_time:.4f}")
    print(ds.stats())
    ray.timeline("map_batches.json")


ds = ray.data.range(NUM_ROWS).map(preprocess)

# run_iter_batches(ds)
run_map_batches(ds)
