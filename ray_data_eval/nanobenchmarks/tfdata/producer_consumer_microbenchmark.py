import tensorflow as tf
import time
import os
import numpy as np
from ray_data_eval.common import logger as logger_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
TF_PROFILER_LOGS = "logs/tf"
LOG_ADDR = "logs/microbenchmark_tf_data.jsonl"
logger = logger_util.Logger(LOG_ADDR)


def busy_loop_for_seconds(time_diff):
    start = time.time()
    i = 0
    while time.time() - start < time_diff:
        i += 1
        continue


def bench():
    NUM_ROWS_PER_TASK = 10
    NUM_TASKS = 16 * 5
    BLOCK_SIZE = 10 * 1024 * 1024 * 10
    TIME_UNIT = 0.5

    options = tf.data.Options()
    options.autotune.enabled = True
    options.autotune.cpu_budget = 80
    # options.autotune.autotune_algorithm = tf.data.experimental.AutotuneAlgorithm.GRADIENT_DESCENT
    options.autotune.ram_budget = 3 * BLOCK_SIZE
    logger.record_start()

    def producer_fn(idx):
        busy_loop_for_seconds(10 * TIME_UNIT)
        for i in range(NUM_ROWS_PER_TASK):
            data = {
                "idx": idx * NUM_ROWS_PER_TASK + i,
                "data": np.full(BLOCK_SIZE, i, dtype=np.uint8),
            }
            yield data
        # logger.log(
        #     {
        #         "producer_finished": int(idx),
        #     }
        # )

    def consumer_fn(idx, data):
        busy_loop_for_seconds(TIME_UNIT)
        # logger.log(
        #     {
        #         "consumer_finished": int(idx),
        #     }
        # )
        return len(data)

    start = time.perf_counter()

    items = list(range(NUM_TASKS - 1))
    ds = tf.data.Dataset.from_tensor_slices(items)
    ds = ds.with_options(options).interleave(
        lambda item: tf.data.Dataset.from_generator(
            producer_fn,
            args=(item,),
            output_signature={
                "idx": tf.TensorSpec(shape=(), dtype=tf.int64),
                "data": tf.TensorSpec(shape=(BLOCK_SIZE,), dtype=tf.uint8),
            },
            name="producer",
        ),
        cycle_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="producer_flat_map",
    )
    ds = ds.with_options(options).map(
        lambda item: tf.numpy_function(
            consumer_fn,
            inp=[item["idx"], item["data"]],  # Pass tensors individually
            Tout=tf.int64,
            name="consumer",
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="consumer_map",
    )

    ret = 0
    for row in ds:
        print(time.perf_counter() - start, row)
        ret += row
        # logger.log(
        #     {
        #         "memory_usage_in_bytes": logger_util.get_process_and_children_memory_usage_in_bytes(),
        #     }
        # )

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(f"Run time: {run_time:.2f} seconds")


if __name__ == "__main__":
    if not os.path.exists(TF_PROFILER_LOGS):
        os.makedirs(TF_PROFILER_LOGS)
    tf.profiler.experimental.start(TF_PROFILER_LOGS)
    bench()
    tf.profiler.experimental.stop()

    print("Check if log directory exists:", os.path.exists(TF_PROFILER_LOGS))
    print("Contents of the log directory:", os.listdir(TF_PROFILER_LOGS))
    # logger_util.plot_from_jsonl(LOG_ADDR, "logs/microbenchmark_tf_data.pdf")
