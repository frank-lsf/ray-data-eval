import tensorflow as tf
import time
import os
import numpy as np
import argparse
import resource

TF_PROFILER_LOGS = "logs/tf"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

MB = 1024 * 1024
GB = 1024 * MB
NUM_CPUS = 8
NUM_GPUS = 4
NUM_ROWS_PER_TASK = 10
NUM_TASKS = 16 * 5
NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
ROW_SIZE = 100 * MB
TIME_UNIT = 0.5


def limit_cpu_memory(mem_limit):
    # limit cpu memory with resources
    mem_limit_bytes = mem_limit * GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))


def bench(mem_limit):
    limit_cpu_memory(mem_limit)

    options = tf.data.Options()
    # https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutotuneOptions
    options.autotune.enabled = True
    options.autotune.cpu_budget = NUM_CPUS + NUM_GPUS
    # When autotuning is enabled (through autotune), determines the RAM budget to use. Values greater than the available RAM in bytes may result in OOM. If None, defaults to half of the available RAM in bytes.
    # Doesn't work at the moment.
    options.autotune.ram_budget = mem_limit * GB

    def producer_fn(idx):
        time.sleep(10 * TIME_UNIT)
        for i in range(NUM_ROWS_PER_TASK):
            data = {
                "idx": idx * NUM_ROWS_PER_TASK + i,
                "data": np.full(ROW_SIZE, i, dtype=np.uint8),
            }
            yield data

    def consumer_fn(idx, data):
        time.sleep(TIME_UNIT)
        return np.zeros(ROW_SIZE, dtype=np.uint8)

    def inference_fn(data):
        return 1

    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = tf.data.Dataset.from_tensor_slices(items)

    ds = ds.with_options(options).interleave(
        lambda item: tf.data.Dataset.from_generator(
            producer_fn,
            args=(item,),
            output_signature={
                "idx": tf.TensorSpec(shape=(), dtype=tf.int64),
                "data": tf.TensorSpec(shape=(ROW_SIZE,), dtype=tf.uint8),
            },
            name="producer",
        ),
        block_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE if mem_limit > 10 else 1,
        name="producer_interleave",
    )

    ds = ds.with_options(options).map(
        lambda items: tf.numpy_function(
            consumer_fn,
            inp=[items["idx"], items["data"]],
            Tout=tf.uint8,
            name="consumer",
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if mem_limit > 10 else 1,
        name="consumer_map",
    )

    ds = ds.with_options(options).map(
        lambda items: tf.numpy_function(
            inference_fn,
            inp=[items],
            Tout=tf.int64,
            name="inference",
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE if mem_limit > 10 else 1,
        name="inference_map",
    )

    ret = 0
    for row in ds:
        ret += row.numpy()
    run_time = time.perf_counter() - start
    print(f"Sum: {ret:,}")
    print(f"Run time: {run_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()

    if not os.path.exists(TF_PROFILER_LOGS):
        os.makedirs(TF_PROFILER_LOGS)
    bench(args.mem_limit)
