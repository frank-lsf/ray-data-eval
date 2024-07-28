import time
import os
import numpy as np
import ray
import argparse
# import timeline_utils

MB = 1024 * 1024
GB = 1024 * MB
TIME_UNIT = 0.5


def bench(mem_limit):
    os.environ["RAY_DATA_OP_RESERVATION_RATIO"] = "0"

    NUM_CPUS = 8
    NUM_GPUS = 4
    NUM_ROWS_PER_TASK = 10
    NUM_TASKS = 16 * 5
    NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
    BLOCK_SIZE = 10 * 1024 * 1024 * 10

    def produce(batch):
        time.sleep(TIME_UNIT * 10)
        for id in batch["id"]:
            yield {
                "id": [id],
                "image": [np.zeros(BLOCK_SIZE, dtype=np.uint8)],
            }

    def consume(batch):
        time.sleep(TIME_UNIT)
        return {"id": batch["id"], "image": [np.ones(BLOCK_SIZE, dtype=np.uint8)]}

    def inference(batch):
        time.sleep(TIME_UNIT)
        return {"id": batch["id"]}

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = BLOCK_SIZE
    data_context.is_budget_policy = True

    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, object_store_memory=mem_limit * GB)

    ds = ray.data.range(NUM_ROWS_TOTAL, override_num_blocks=NUM_TASKS)
    ds = ds.map_batches(produce, batch_size=NUM_ROWS_PER_TASK)
    ds = ds.map_batches(consume, batch_size=1, num_cpus=0.99)
    ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)

    start_time = time.time()
    for _ in ds.iter_batches(batch_size=NUM_ROWS_PER_TASK):
        pass
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    # timeline_utils.save_timeline_with_cpus_gpus(f"timeline_ray_data_{mem_limit}", NUM_CPUS, NUM_GPUS)
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()
    bench(args.mem_limit)
