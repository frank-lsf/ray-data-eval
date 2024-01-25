import time
import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("DEBUG")


W = 1000
OUTPUT_BYTES = W * W * 8


def foo(i):
    a = np.random.rand(W * W).reshape((W, W))
    a = np.linalg.matrix_power(a, 100)
    return a.flatten()


start = time.perf_counter()

options = tf.data.Options()
options.autotune.autotune_algorithm = tf.data.experimental.AutotuneAlgorithm.GRADIENT_DESCENT
options.autotune.enabled = True
options.autotune.cpu_budget = 2
# options.autotune.ram_budget = OUTPUT_BYTES * 2

ds = tf.data.Dataset.range(100).with_options(options)
ds = ds.map(
    lambda item: tf.numpy_function(
        foo,
        [item],
        Tout=tf.float64,
    ),
    # num_parallel_calls=1,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


for row in ds:
    print(time.perf_counter() - start, row[0])
