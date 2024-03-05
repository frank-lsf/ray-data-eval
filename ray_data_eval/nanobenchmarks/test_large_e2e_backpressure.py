import json
import time

import numpy as np
import ray
from ray.data._internal.execution.backpressure_policy import (
    ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY,
    StreamingOutputBackpressurePolicy,
)

LOG_FILE = "test_large_e2e_backpressure.log"


class Logger:
    def __init__(self, filename: str = LOG_FILE):
        self._filename = filename
        self._start_time = time.time()

    def record_start(self):
        self._start_time = time.time()
        with open(self._filename, "w"):
            pass

    def log(self, payload: dict):
        payload = {
            **payload,
            "time": time.time() - self._start_time,
        }
        with open(self._filename, "a") as f:
            f.write(json.dumps(payload) + "\n")


logger = Logger()


def test_large_e2e_backpressure(enable_backpressure: bool = True):
    """Test backpressure on a synthetic large-scale workload."""
    # The cluster has 10 CPUs and 200MB object store memory.
    # The dataset will have 200MB * 25% = 50MB memory budget.
    #
    # Each produce task generates 10 blocks, each of which has 10MB data.
    #
    # Without any backpressure, the producer tasks will output at most
    # 10 * 10 * 10MB = 1000MB data.
    #
    # With StreamingOutputBackpressurePolicy and the following configuration,
    # the executor will still schedule 10 produce tasks, but only the first task is
    # allowed to output all blocks. The total size of pending blocks will be
    # (10 + 9 * 1 + 1) * 10MB = 200MB, where
    # - 10 is the number of blocks in the first task.
    # - 9 * 1 is the number of blocks pending at the streaming generator level of
    #   the other 15 tasks.
    # - 1 is the number of blocks pending at the output queue.

    NUM_CPUS = 10
    NUM_ROWS_PER_TASK = 10
    NUM_TASKS = 20
    NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
    BLOCK_SIZE = 10 * 1024 * 1024
    STREMING_GEN_BUFFER_SIZE = 1
    OP_OUTPUT_QUEUE_SIZE = 1
    max_pending_block_bytes = (
        NUM_ROWS_PER_TASK + (NUM_CPUS - 1) * STREMING_GEN_BUFFER_SIZE + OP_OUTPUT_QUEUE_SIZE
    ) * BLOCK_SIZE
    print(f"max_pending_block_bytes: {max_pending_block_bytes/1024/1024}MB")

    ray.init(num_cpus=NUM_CPUS, object_store_memory=200 * 1024 * 1024)

    def produce(batch):
        logger.log({"name": "producer_start", "id": [int(x) for x in batch["id"]]})
        time.sleep(0.1)
        for id in batch["id"]:
            logger.log({"name": "produce", "id": int(id)})
            yield {
                "id": [id],
                "image": [np.zeros(BLOCK_SIZE, dtype=np.uint8)],
            }

    def consume(batch):
        logger.log({"name": "consume", "id": int(batch["id"])})
        time.sleep(0.01)
        return {"id": batch["id"], "result": [0 for _ in batch["id"]]}

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = BLOCK_SIZE
    if enable_backpressure:
        data_context.set_config(
            ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY,
            [
                StreamingOutputBackpressurePolicy,
            ],
        )
        data_context.set_config(
            StreamingOutputBackpressurePolicy.MAX_BLOCKS_IN_OP_OUTPUT_QUEUE_CONFIG_KEY,
            OP_OUTPUT_QUEUE_SIZE,
        )
        data_context.set_config(
            StreamingOutputBackpressurePolicy.MAX_BLOCKS_IN_GENERATOR_BUFFER_CONFIG_KEY,
            STREMING_GEN_BUFFER_SIZE,
        )
    else:
        data_context.set_config(ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY, [])

    ds = ray.data.range(NUM_ROWS_TOTAL, parallelism=NUM_TASKS)
    ds = ds.map_batches(produce, batch_size=NUM_ROWS_PER_TASK)
    ds = ds.map_batches(consume, batch_size=None, num_cpus=0.9)

    logger.record_start()

    start_time = time.time()
    logger.log({"name": "execution_start"})
    for i, _ in enumerate(ds.iter_batches(batch_size=NUM_ROWS_PER_TASK)):
        logger.log({"name": "iteration", "id": i})
        pass
    end_time = time.time()
    print(ds.stats())
    print(f"Total time: {end_time - start_time:.4f}s")
    ray.timeline("timeline.json")


if __name__ == "__main__":
    test_large_e2e_backpressure()
