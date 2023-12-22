import tensorflow as tf
import numpy as np
import time
import wandb

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_BASIS = 0.1  # How many seconds should time_factor=1 take


def memory_blowup(_item, time_factor):
    data = np.full(DATA_SIZE_BYTES, 1, dtype=np.uint8)
    time.sleep(TIME_BASIS * time_factor)
    return data


def memory_shrink(data, time_factor):
    time.sleep(TIME_BASIS * time_factor)
    return np.sum(data)


def run_experiment(
    num_parts: int,
    producer_time: int,
    consumer_time: int,
):
    items = tf.data.Dataset.from_tensor_slices(np.arange(num_parts))
    ds = items.map(
        lambda item: tf.numpy_function(
            memory_blowup,
            [item, producer_time],
            Tout=tf.uint8,
        )
    )
    ds = ds.map(
        lambda data: tf.numpy_function(
            memory_shrink,
            [data, consumer_time],
            Tout=tf.uint64,
        )
    )

    ret = 0
    for row in ds.as_numpy_iterator():
        print(row)
        ret += row

    print(f"\n{ret:,}")
    return ret


def main():
    # Initialize wandb
    wandb.init(project="tf-data-eval", entity="raysort")

    config = {
        "kind": "uniform",  # Each task runs for the same amount of time
        "parallelism": 100,
        "total_data_size_gb": 100,
        "producer_time": 1,
        "consumer_time": 9,
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
        producer_time=config["producer_time"],
        consumer_time=config["consumer_time"],
    )


if __name__ == "__main__":
    main()
