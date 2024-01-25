import os
import time

import numpy as np

# import wandb

from ray_data_eval.common.types import SchedulingProblem, test_problem


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("DEBUG")

DATA_SIZE_MB = 100
DATA_SIZE_BYTES = DATA_SIZE_MB * 1000 * 1000
TIME_UNIT = 1  # seconds
TF_PROFILER_LOGS = "logs/tf"


def producer_factory(cfg: SchedulingProblem):
    def producer(i: int):
        data = np.full(DATA_SIZE_BYTES * cfg.producer_output_size[i], i, dtype=np.uint8)
        time.sleep(TIME_UNIT * cfg.producer_time[i])
        return data

    return producer


def consumer_factory(cfg: SchedulingProblem):
    def consumer(data):
        i = data[0]
        time.sleep(TIME_UNIT * cfg.consumer_time[i])
        return len(data)

    return consumer


def get_options(cfg: SchedulingProblem):
    options = tf.data.Options()
    options.autotune.autotune_algorithm = tf.data.experimental.AutotuneAlgorithm.GRADIENT_DESCENT
    options.autotune.cpu_budget = cfg.num_execution_slots
    options.autotune.ram_budget = int(DATA_SIZE_BYTES * cfg.buffer_size_limit * 1.1)
    return options


def run_tf_data(cfg: SchedulingProblem):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")

    options = get_options(cfg)
    start = time.perf_counter()

    items = list(range(cfg.num_producers))
    ds = tf.data.Dataset.from_tensor_slices(items)
    ds = ds.with_options(options).map(
        lambda item: tf.numpy_function(
            producer_factory(cfg),
            [item],
            Tout=tf.uint8,
        ),
        # num_parallel_calls=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.with_options(options).map(
        lambda item: tf.numpy_function(
            consumer_factory(cfg),
            [item],
            Tout=tf.int64,
        ),
        # num_parallel_calls=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    ret = 0
    for row in ds:
        print(time.perf_counter() - start, row)
        ret += row

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(f"Run time: {run_time:.2f} seconds")
    # wandb.log({"run_time": run_time})
    return ret


def run_experiment(cfg: SchedulingProblem):
    # wandb.init(project="tf-data-eval", entity="raysort", sync_tensorboard=True)
    # wandb.config.update(cfg)
    tf.profiler.experimental.start(TF_PROFILER_LOGS)
    run_tf_data(cfg)
    tf.profiler.experimental.stop()


def main():
    run_experiment(test_problem)


if __name__ == "__main__":
    main()
