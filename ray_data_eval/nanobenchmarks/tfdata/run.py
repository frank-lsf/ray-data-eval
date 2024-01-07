import time

import numpy as np
import tensorflow as tf
import wandb

from ray_data_eval.common.types import SchedulingProblem, test_problem

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
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
    options.autotune.cpu_budget = 1
    # TODO(@lsf): memory budget
    # options.experimental_optimization.map_and_batch_fusion = False
    options.threading.max_intra_op_parallelism = 1
    return options


def run_tf_data(cfg: SchedulingProblem):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")
    start = time.perf_counter()

    items = list(range(cfg.num_producers))
    ds = tf.data.Dataset.from_tensor_slices(items).with_options(get_options(cfg))
    ds = ds.map(
        lambda item: tf.numpy_function(
            producer_factory(cfg),
            [item],
            Tout=tf.uint8,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(
        lambda item: tf.numpy_function(
            consumer_factory(cfg),
            [item],
            Tout=tf.int64,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    ret = 0
    for row in ds:
        print(time.perf_counter() - start, row)
        ret += row

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    wandb.log({"run_time": run_time})
    return ret


def run_experiment(cfg: SchedulingProblem):
    wandb.init(project="tf-data-eval", entity="raysort", sync_tensorboard=True)
    wandb.config.update(cfg)
    tf.profiler.experimental.start(TF_PROFILER_LOGS)
    run_tf_data(cfg)


def main():
    run_experiment(test_problem)


if __name__ == "__main__":
    main()
