import tensorflow as tf
import numpy as np
import time
import wandb

TF_PROFILER_LOGS = "logs/tf"

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_BASIS = 0.1  # How many seconds should time_factor=1 take


def benchmark(
    dataset,
    num_epochs: int = 1,
    time_factor: float = TIME_BASIS,
):
    ret = 0
    for _ in range(num_epochs):
        for sample in dataset:
            time.sleep(TIME_BASIS * time_factor)
            ret += len(sample)
    print(ret)


def memory_blowup(_item, time_factor):
    data = np.full(DATA_SIZE_BYTES, 1, dtype=np.uint8)
    time.sleep(TIME_BASIS * time_factor)
    return data


def run_experiment(
    *,
    num_parts: int,
    parallelism: int,
    preprocessing_time: int,
    training_time: int,
):
    if parallelism < 0:
        parallelism = tf.data.experimental.AUTOTUNE

    start = time.perf_counter()
    ds = tf.data.Dataset.from_tensor_slices(np.arange(num_parts))
    ds = ds.map(
        lambda item: tf.numpy_function(
            memory_blowup,
            [item, preprocessing_time],
            Tout=tf.uint8,
        ),
        num_parallel_calls=parallelism,
    )
    benchmark(
        ds,
        time_factor=training_time,
    )

    run_time = time.perf_counter() - start
    wandb.log({"run_time": run_time})


def main():
    wandb.init(project="tf-data-eval", entity="raysort")
    tf.profiler.experimental.start(TF_PROFILER_LOGS)

    config = {
        "kind": "uniform",  # Each task runs for the same amount of time
        "parallelism": -1,
        "total_data_size_gb": 10,
        "producer_time": 1,
        "consumer_time": 9,
        "memory_limit": 10**9 * 20,
    }
    config["total_data_size"] = config["total_data_size_gb"] * 10**9
    config["num_parts"] = config["total_data_size"] // DATA_SIZE_BYTES
    config["producer_consumer_ratio"] = (
        config["producer_time"] / config["consumer_time"]
    )
    wandb.config.update(config)

    # Run the experiment
    run_experiment(
        num_parts=config["num_parts"],
        parallelism=config["parallelism"],
        preprocessing_time=config["producer_time"],
        training_time=config["consumer_time"],
    )
    tf.profiler.experimental.stop()


if __name__ == "__main__":
    main()
