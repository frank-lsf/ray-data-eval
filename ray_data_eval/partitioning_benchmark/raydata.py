import time
from typing import Any

import ray

ROW_SIZE = 1000 * 1000
NUM_ROWS = 8192
TIME_PER_ROW = 0.01


def busy(duration: float):
    start_time = time.time()
    while time.time() - start_time < duration:
        pass


def process(batch: dict[str, Any]):
    num_rows = len(batch["data"])
    # if num_rows >= 100:
    #     print(f"Process {num_rows} rows")
    busy(num_rows * TIME_PER_ROW)
    return batch


def drop(batch: dict[str, Any]):
    num_rows = len(batch["data"])
    # if num_rows >= 100:
    #     print(f"Drop {num_rows} rows")
    busy(num_rows * TIME_PER_ROW)
    return {"data": [0]}


def bench(num_rows_in_block: int):
    start_time = time.time()

    ds = ray.data.range_tensor(
        NUM_ROWS,
        shape=(1000, 125),  # np.uint64 is 8 bytes
        override_num_blocks=NUM_ROWS // num_rows_in_block,
    )
    ds = ds.map_batches(
        process,
        zero_copy_batch=True,
        concurrency=8,
    )
    ds = ds.map_batches(
        drop,
        num_cpus=0.99,
        zero_copy_batch=True,
        concurrency=8,
    )
    ds.take_all()

    ray.timeline(f"timeline_{num_rows_in_block}.json")

    duration = time.time() - start_time
    print(f"{num_rows_in_block=}, {duration=}")
    return duration


def main():
    bench(1000)
    print("Warmup done")
    bench(1)
    bench(2)
    bench(4)
    bench(8)
    bench(16)
    bench(32)
    bench(64)
    bench(128)
    bench(256)
    bench(512)
    bench(1024)


if __name__ == "__main__":
    ray.init(object_store_memory=15e9)

    data_context = ray.data.DataContext.get_current()
    data_context.op_resource_reservation_ratio = 0
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = 1024 * 1024 * 1024
    data_context.is_budget_policy = False
    main()
