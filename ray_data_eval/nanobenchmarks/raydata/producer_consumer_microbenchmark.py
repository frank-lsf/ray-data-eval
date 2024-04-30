import os
import time

import numpy as np

import ray
from ray_data_eval.common import logger as logger_util

LOG_ADDR = "logs/microbenchmark_ray_data.jsonl"
logger = logger_util.Logger(LOG_ADDR)

import resource

def set_memory_limit(soft_limit, hard_limit):
    # Set the memory limit in bytes
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))

set_memory_limit(1024 * 1024 * 1024, 1024 * 1024 * 1024)  # 1 GB limit


def bench():
    os.environ["RAY_DATA_OP_RESERVATION_RATIO"] = "0"

    NUM_CPUS = 8
    NUM_ROWS_PER_TASK = 10
    NUM_TASKS = 16 * 5
    NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
    BLOCK_SIZE = 10 * 1024 * 1024 * 10
    TIME_UNIT = 0.5
    logger.record_start()

    def produce(batch):
        time.sleep(TIME_UNIT * 10)
        for id in batch["id"]:
            yield {
                "id": [id],
                "image": [np.zeros(BLOCK_SIZE, dtype=np.uint8)],
            }
        logger.log(
            {
                "producer_finished": int(batch["id"][0] / NUM_ROWS_PER_TASK),
            }
        )

    def consume(batch):
        time.sleep(TIME_UNIT)

        logger.log(
            {
                "consumer_finished": int(batch["id"]),
            }
        )
        return {"id": batch["id"], "result": [0 for _ in batch["id"]]}

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = BLOCK_SIZE

    ray.init(num_cpus=NUM_CPUS, object_store_memory=25 * BLOCK_SIZE)

    ds = ray.data.range(NUM_ROWS_TOTAL, override_num_blocks=NUM_TASKS)
    ds = ds.map_batches(produce, batch_size=NUM_ROWS_PER_TASK)
    ds = ds.map_batches(consume, batch_size=None, num_cpus=0.99)
    start_time = time.time()
    for _, _ in enumerate(ds.iter_batches(batch_size=NUM_ROWS_PER_TASK)):
        logger.log(
            {
                "memory_usage_in_bytes": logger_util.get_process_and_children_memory_usage_in_bytes(),
            }
        )
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    ray.timeline("timeline.json")


if __name__ == "__main__":
    bench()
    logger_util.plot_from_jsonl(LOG_ADDR, "logs/microbenchmark_ray_data.pdf")
