import tensorflow as tf
import time
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
TF_PROFILER_LOGS = "logs/tf"

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
    options.autotune.cpu_budget = 8
    options.autotune.autotune_algorithm = tf.data.experimental.AutotuneAlgorithm.GRADIENT_DESCENT
    options.autotune.ram_budget = 25 * BLOCK_SIZE

    def produce(batch):
        start_time = time.time()
        tf.print(start_time, 'produce', batch, TIME_UNIT * 10)
        busy_loop_for_seconds(TIME_UNIT * 10)
        end_time = time.time()
        tf.print(end_time, 'produce finish sleep', batch)
        return tf.data.Dataset.from_tensor_slices({
            "id": tf.range(batch * NUM_ROWS_PER_TASK, batch * NUM_ROWS_PER_TASK + NUM_ROWS_PER_TASK),
            "image": tf.tile(tf.expand_dims(tf.zeros([BLOCK_SIZE], dtype=tf.uint8), 0), [NUM_ROWS_PER_TASK, 1]),
        })

    def consume(item):
        start_time = time.time()
        tf.print(start_time, 'consume', item["id"])
        busy_loop_for_seconds(TIME_UNIT)
        end_time = time.time()
        tf.print(end_time, 'consume finished', item["id"])
        return {"id": item["id"], "result": tf.zeros([], dtype=tf.int32)}

    start_time = time.time()
    ds = tf.data.Dataset.range(NUM_TASKS)
    ds = ds.with_options(options)
    # flat_map does not take num_parallel_calls
    # Will create NUM_TASKS producer datasets. 
    ds = ds.with_options(options).interleave(produce,
                        # How many datasets to process in parallel.
                       cycle_length=1, 
                        # How many elements for each dataset before cycling to the next dataset.
                       block_length=1,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    ds = ds.with_options(options).map(consume, num_parallel_calls=tf.data.experimental.AUTOTUNE)  

    for _ in ds:  # Execute the entire dataset to measure total time
        pass
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f}s")


def run_bench_second():
    

    NUM_ROWS_PER_TASK = 10
    NUM_TASKS = 16 * 5
    BLOCK_SIZE = 10 * 1024 * 1024 * 10
    TIME_UNIT = 0.5

    options = tf.data.Options()
    options.autotune.enabled = True
    options.autotune.cpu_budget = 8
    options.autotune.autotune_algorithm = tf.data.experimental.AutotuneAlgorithm.GRADIENT_DESCENT
    options.autotune.ram_budget = 25 * BLOCK_SIZE
    
    def producer_fn(idx):
        tf.print(time.time(), 'produce', idx)
        time.sleep(10 * TIME_UNIT)
        for i in range(NUM_ROWS_PER_TASK):
            data = {
                'idx': idx * NUM_ROWS_PER_TASK + i,
                'data': np.full(BLOCK_SIZE, i, dtype=np.uint8)
            }
            yield data


    def consumer_fn(idx, data):
        print(time.time(), 'consume', idx)
        time.sleep(TIME_UNIT)
        return len(data)

    start = time.perf_counter()

    items = list(range(NUM_TASKS - 1))
    ds = tf.data.Dataset.from_tensor_slices(items)
    ds = ds.with_options(options).interleave(
        lambda item: tf.data.Dataset.from_generator(
            producer_fn,
            args=(item,),  
            output_signature={
                'idx': tf.TensorSpec(shape=(), dtype=tf.int64),
                'data': tf.TensorSpec(shape=(BLOCK_SIZE,), dtype=tf.uint8)
            },
            name="producer",
        ),
        cycle_length=1,
        # num_parallel_calls=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="producer_flat_map",
    )
    ds = ds.with_options(options).map(
        lambda item: tf.numpy_function(
            consumer_fn,
            inp=[item['idx'], item['data']],  # Pass tensors individually
            Tout=tf.int64,
            name="consumer",
        ),
        # num_parallel_calls=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        name="consumer_map",
    )

    ret = 0
    for row in ds:
        print(time.perf_counter() - start, row)
        ret += row

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(f"Run time: {run_time:.2f} seconds")
    

if __name__ == "__main__":
    
    if not os.path.exists(TF_PROFILER_LOGS):
        os.makedirs(TF_PROFILER_LOGS)
    tf.profiler.experimental.start(TF_PROFILER_LOGS)
    run_bench_second()
    tf.profiler.experimental.stop()
    
    print("Check if log directory exists:", os.path.exists(TF_PROFILER_LOGS))
    print("Contents of the log directory:", os.listdir(TF_PROFILER_LOGS))

